from datetime import datetime
from decimal import Decimal
import itertools
from operator import itemgetter

import ccxt.gdax
from ccxt.websocket_support.exchange import Exchange

import cbpro
from ccxt.base.errors import ExchangeError


class OrderBook(cbpro.OrderBook):
    """An OrderBook class without the print statements."""

    def on_open(self):
        self._sequence = -1

    def on_close(self):
        pass

    def on_sequence_gap(self, gap_start, gap_end):
        self.reset_book()


class WebsocketClient(cbpro.WebsocketClient):
    _tickers_by_product_id = {}

    def __init__(self, *args, **kwargs):
        if "logger" in kwargs:
            self._logger = kwargs["logger"]
            del kwargs["logger"]

        super(WebsocketClient, self).__init__(*args, **kwargs)
        self._order_books_by_product_id = {}
        self._orders_by_id_by_product_id = {}
        self._sequences_by_product_id = {}
        self._trade_ids_seen_by_product_id = {}

    def _full_channel_callback(self, msg):
        """Process a message from the full channel and update the orders cache.

        We will maintain a full image of our orders under
        self._orders_by_id_by_product_id. Each order will have a _type field to
        indicate the last WebSocket message that updated it.

        The following order fields will not be filled in by WebSocket messages:
        * stp
        * time_in_force
        * post_only
        * fill_fees
        * settled
        * specified_funds (it is not clear whether the change message refers to
          funds or specified_funds)

        """
        # We will be strict in the ordering of messages; if the sequence number
        # is wrong, we will delete all orders (for a product) from the cache.
        if msg["type"] == "activate":
            # Strangely enough, according to the example on the documentation,
            # it does not have a sequence number.
            pass
        elif msg["product_id"] not in self._sequences_by_product_id:
            self._sequences_by_product_id[msg["product_id"]] = msg["sequence"]
        else:
            actual = msg["sequence"]
            expected = self._sequences_by_product_id[msg["product_id"]] + 1
            if actual == expected:
                self._sequences_by_product_id[msg["product_id"]] = expected
            else:
                self._logger.error("Expected sequence number {} but got {} "
                                   "instead. Removing {} cache.".format(
                                       expected, actual, msg["product_id"]))
                self._orders_by_id_by_product_id.pop(msg["product_id"], None)
                self._sequences_by_product_id[msg["product_id"]] = actual
                self._trade_ids_seen_by_product_id.pop(msg["product_id"], None)

        orders_by_id = self._orders_by_id_by_product_id.setdefault(
            msg["product_id"], {})

        if msg["type"] == "activate":
            # There's not a lot of documentation on stop orders, so we will
            # preserve their non-standard fields and allow it to transition to
            # any state.
            msg["_type"] = msg["type"]
            msg["id"] = msg["order_id"]
            msg["type"] = "limit" if "price" in msg else "market"
            msg["created_at"] = float(msg["timestamp"])
            msg["created_at"] = datetime.utcfromtimestamp(msg["created_at"])
            msg["created_at"] = msg["created_at"].isoformat()
            msg["status"] = "active"

            del msg["order_id"]
            del msg["timestamp"]

            orders_by_id[msg["id"]] = msg
            return

        if msg["type"] == "match":
            orders = [
                order for order_id, order in orders_by_id.items()
                if order_id in (msg["maker_order_id"], msg["taker_order_id"])
            ]
            if len(orders) == 0:
                return

            trade_ids_seen = self._trade_ids_seen_by_product_id.setdefault(
                msg["product_id"], set())
            if msg["trade_id"] not in trade_ids_seen:
                trade_ids_seen.add(msg["trade_id"])
            else:
                return

            for order in orders:
                if order["_type"] not in ("received", "open", "match",
                                          "change", "activate"):
                    self._logger.error("Unexpected match after {} for order {}"
                                       ". Removing from cache.".format(
                                           order["_type"], order["id"]))
                    del orders_by_id[order["id"]]
                    continue

                order["_type"] = "match"

                filled_size = Decimal(order.get("filled_size", 0))
                filled_size += Decimal(msg["size"])
                order["filled_size"] = str(filled_size)

                executed_value = Decimal(order.get("executed_value", 0))
                increase = Decimal(msg["size"]) * Decimal(msg["price"])
                order["executed_value"] = str(executed_value + increase)

            return

        if msg["order_id"] not in orders_by_id and msg["type"] != "received":
            # It's possible to receive messages for orders not in the cache
            # (especially at the beginning), but it's not an error.
            self._logger.debug("Ignoring {} message as order {} is not in the "
                               "cache.".format(msg["type"], msg["order_id"]))
            return

        order = orders_by_id.setdefault(msg["order_id"], {})

        # If the state transition is wrong, remove the order from the cache.
        _type = order.get("_type")
        if ((_type is None and msg["type"] != "received")
                or (_type == "received" and msg["type"] not in (
                    "open", "done", "match", "change"))
                or (_type == "open" and msg["type"] not in (
                    "done", "match", "change"))
                or (_type == "done")
                or (_type == "match" and msg["type"] not in (
                    "open", "done", "match", "change"))
                or (_type == "change" and msg["type"] not in (
                    "open", "done", "match", "change"))):
            self._logger.error("Unexpected {} after {} for order {}. Removing "
                               "from cache.".format(msg["type"], _type,
                                                    msg["order_id"]))
            del orders_by_id[msg["order_id"]]
            return

        status = order.get("status")
        if status == "open" and msg["type"] == "open":
            self._logger.error("Unexpected open for open order {}. Removing "
                               "from cache.".format(msg["order_id"]))
            del orders_by_id[msg["order_id"]]
            return

        if msg["type"] == "received":
            msg["_type"] = msg["type"]
            msg["id"] = msg["order_id"]
            msg["type"] = msg["order_type"]
            msg["created_at"] = msg["time"]
            msg["filled_size"] = "0"
            msg["executed_value"] = "0"

            del msg["time"]
            del msg["sequence"]
            del msg["order_id"]
            del msg["order_type"]

            order.update(msg)
        elif msg["type"] == "open":
            msg["_type"] = msg["type"]
            msg["id"] = msg["order_id"]
            msg["filled_size"] = Decimal(order["size"])
            msg["filled_size"] -= Decimal(msg["remaining_size"])
            msg["filled_size"] = str(msg["filled_size"])
            msg["status"] = msg["type"]

            del msg["type"]
            del msg["time"]
            del msg["sequence"]
            del msg["order_id"]
            del msg["remaining_size"]

            order.update(msg)
        elif msg["type"] == "done":
            msg["_type"] = msg["type"]
            msg["id"] = msg["order_id"]
            msg["done_at"] = msg["time"]
            msg["done_reason"] = msg["reason"]
            msg["filled_size"] = Decimal(order["size"])
            msg["filled_size"] -= Decimal(msg["remaining_size"])
            msg["filled_size"] = str(msg["filled_size"])
            msg["status"] = "done" if msg["reason"] == "filled" else "canceled"

            del msg["type"]
            del msg["time"]
            del msg["sequence"]
            del msg["order_id"]
            del msg["reason"]
            del msg["remaining_size"]

            order.update(msg)
        elif msg["type"] == "change":
            msg["_type"] = msg["type"]
            msg["id"] = msg["order_id"]

            if "new_size" in msg:
                msg["size"] = msg["new_size"]
                del msg["new_size"]
                del msg["old_size"]

            if "new_funds" in msg:
                msg["funds"] = msg["new_funds"]
                del msg["new_funds"]
                del msg["old_funds"]

            del msg["type"]
            del msg["time"]
            del msg["sequence"]
            del msg["order_id"]

            order.update(msg)
        else:
            self._logger.error("Unexpected message type: " + msg["type"])

    def _ticker_channel_callback(self, msg):
        # These products have zero volume.
        if "sequence" not in msg:
            self._tickers_by_product_id[msg["product_id"]] = msg
            return

        old = self._tickers_by_product_id.get(msg["product_id"], {})
        if msg["sequence"] > old.get("sequence", float("-inf")):
            self._tickers_by_product_id[msg["product_id"]] = msg

    def on_message(self, msg):
        if msg["type"] == "ticker":
            return self._ticker_channel_callback(msg)
        elif msg["type"] in ("received", "open", "done", "match", "change",
                             "activate"):
            if msg["product_id"] in self._order_books_by_product_id:
                self._order_books_by_product_id[msg["product_id"]].on_message(
                    msg)

            if "profile_id" in msg.keys():
                self._full_channel_callback(msg)
        elif msg["type"] == "error":
            # If there is an error that causes messages to be dropped (but does
            # not terminate the connection), the sequence number should allow
            # us to determine whether to carry on.
            return self._logger.error(msg["message"])


class gdax(Exchange, ccxt.gdax):
    """A WebSocket-enabled, drop-in replacement for the ccxt GDAX client.

    The following methods are WebSocket-enabled:
    * fetch_ticker
    * fetch_order_book
    * fetch_orders
    * fetch_open_orders
    * fetch_closed_orders

    When you are finished, you should call close() to close all WebSockets (or
    use a `with` statement).

    """
    def __init__(self, config={}):
        super(gdax, self).__init__(config)
        self.load_markets()

        if "client" not in config:
            config["client"] = WebsocketClient(
                products=[market["id"] for market in self.markets.values()],
                should_print=False,
                auth=len(self.apiKey) > 0,
                api_key=self.apiKey,
                api_secret=self.secret,
                api_passphrase=config.get("apiPassphrase", ""),
                channels=["ticker", "full"],
                logger=self._logger)
            config["client"].start()

        self._client = config["client"]

        self._order_statuses_queried = set()
        self._order_statuses_queried_by_product_id = {}

        self._parent_privateGetOrders = self.privateGetOrders
        self.privateGetOrders = self._privateGetOrders

        self._parent_publicGetProductsIdBook = self.publicGetProductsIdBook
        self.publicGetProductsIdBook = self._publicGetProductsIdBook

        self._parent_publicGetProductsIdTicker = self.publicGetProductsIdTicker
        self.publicGetProductsIdTicker = self._publicGetProductsIdTicker

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _privateGetOrders(self, request):
        order_statuses_queried = self._order_statuses_queried
        if "product_id" in request:
            order_statuses_queried = order_statuses_queried.union(
                self._order_statuses_queried_by_product_id.get(
                    request["product_id"], set()))
            orders_by_ids = [self._client._orders_by_id_by_product_id.get(
                request["product_id"], {})]
        else:
            orders_by_ids = self._client._orders_by_id_by_product_id.values()

        if "status" in request:
            statuses = set([request["status"]])
        else:
            statuses = set(["open", "pending", "active"])

        if "all" in statuses and "all" in order_statuses_queried:
            return [order for orders_by_id in orders_by_ids
                    for order in orders_by_id.values()]
        elif ("all" in order_statuses_queried
                or order_statuses_queried.issuperset(statuses)):
            return [
                order for orders_by_id in orders_by_ids
                for order in orders_by_id.values()
                if order["status"] in statuses
            ]

        response = self._parent_privateGetOrders(request)
        groupby = itertools.groupby(response, key=itemgetter("product_id"))
        for product_id, grouper in groupby:
            orders_by_id = self._client._orders_by_id_by_product_id.setdefault(
                product_id, {})
            for order in grouper:
                if order["id"] in orders_by_id:
                    # The WebSocket version is more up to date.
                    continue

                if order["status"] in "pending":
                    order["_type"] = "received"
                elif order["status"] in ("open", "canceling"):
                    order["_type"] = "open"
                elif order["status"] in ("done", "canceled"):
                    order["_type"] = "done"
                elif order["status"] == "active":
                    order["_type"] = "activate"

                # There's no harm in leaving an order with an unknown status in
                # the cache. If a WebSocket message comes in referring to it
                # and we don't know how to handle the state transition, it will
                # be removed at the end.

                orders_by_id[order["id"]] = order

        if "product_id" in request:
            order_statuses_queried = self._order_statuses_queried_by_product_id
            order_statuses_queried = order_statuses_queried.setdefault(
                request["product_id"], set())
        else:
            order_statuses_queried = self._order_statuses_queried

        if "status" in request:
            order_statuses_queried.add(request["status"])
        else:
            order_statuses_queried.update(("open", "pending", "active"))

        return response

    def _publicGetProductsIdBook(self, request):
        if request["id"] not in self._client._order_books_by_product_id:
            order_book = OrderBook([request["id"]])
            self._client._order_books_by_product_id[request["id"]] = order_book
        else:
            order_book = self._client._order_books_by_product_id[request["id"]]

        if order_book._sequence == -1:
            return self._parent_publicGetProductsIdBook(request)

        # Make a copy of the order book so it doesn't change while we're
        # constructing the return value.
        sequence = order_book._sequence
        bids = order_book._bids.copy()
        asks = order_book._asks.copy()

        level = request.get("level", 1)
        if level == 1 or level == 2:
            bid_levels = bids.items()[-1 if level == 1 else 0:]
            ask_levels = asks.items()[:1 if level == 1 else len(asks)]

            return {
                "sequence": sequence,
                "bids": [
                    [
                        bid_level[0],
                        sum(order["size"] for order in bid_level[1]),
                        len(bid_level[1]),
                    ]
                    for bid_level in bid_levels
                ],
                "asks": [
                    [
                        ask_level[0],
                        sum(order["size"] for order in ask_level[1]),
                        len(ask_level[1]),
                    ]
                    for ask_level in ask_levels
                ],
            }
        elif level == 3:
            return {
                "sequence": sequence,
                "bids": [
                    [order["price"], order["size"], order["id"]]
                    for bid_level in bids.values() for order in bid_level
                ],
                "asks": [
                    [order["price"], order["size"], order["id"]]
                    for ask_level in asks.values() for order in ask_level
                ],
            }
        else:
            raise ExchangeError("{} unexpected level: {}".format(self.id,
                                                                 level))

    def _publicGetProductsIdTicker(self, request):
        if request["id"] not in self._client._tickers_by_product_id:
            ticker = self._parent_publicGetProductsIdTicker(request)
            self._client._tickers_by_product_id[request["id"]] = ticker

        ticker = self._client._tickers_by_product_id[request["id"]]
        return {
            "trade_id": ticker["trade_id"],
            "price": ticker["price"],
            "size": ticker["last_size"],
            "bid": ticker["best_bid"],
            "ask": ticker["best_ask"],
            "volume": ticker["volume_24h"],
            "time": ticker["time"],
        }

    def close(self):
        self._client.close()

    def fetch_order(self, id, symbol=None, params={}):
        if symbol is not None:
            product_id = self.market(symbol)["id"]
            orders_by_ids = [
                self._client._orders_by_id_by_product_id.get(product_id, {}),
            ]
        else:
            orders_by_ids = self._client._orders_by_id_by_product_id.values()

        for orders_by_id in orders_by_ids:
            if id in orders_by_id:
                return self.parse_order(orders_by_id[id])

        return self.parse_order(self.privateGetOrdersId(self.extend({
            "id": id,
        }, params)))
