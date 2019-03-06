from functools import partial
import heapq
import itertools
import operator

from binance.client import Client
from binance.websockets import BinanceClientProtocol, BinanceSocketManager

import ccxt.binance
from ccxt.websocket_support.exchange import Exchange

# Python 3 compatibility.
try:
    import Queue
except ImportError:
    import queue as Queue

_ALL_ORDERS_MAX_LIMIT = 1000
_TICKER_STREAM_NAME = "!ticker@arr"
_USER_DATA_STREAM_NAME = "userDataStream"


def BinanceClientProtocolConstructor(self, *args, **kwargs):
    super(BinanceClientProtocol, self).__init__(*args, **kwargs)


# Workaround for incompatibility with Python 2.
BinanceClientProtocol.__init__ = BinanceClientProtocolConstructor


def ClientConstructor(self, api_key, api_secret, requests_params=None,
                      headers={}):
    self.API_KEY = api_key
    self.API_SECRET = api_secret
    self.session = self._init_session()
    self.session.headers.update(headers)
    self._requests_params = requests_params
    self.ping()


# Allow arena_python to disable gzip by modifying request headers.
Client.__init__ = ClientConstructor


class binance(Exchange, ccxt.binance):
    """A WebSocket-enabled, drop-in replacement for the ccxt Binance client.

    The following methods are WebSocket-enabled:
    * fetch_ticker
    * fetch_tickers
    * fetch_order_book (this can be expensive; see _publicGetDepth() docstring)
    * fetch_balance
    * fetch_orders
    * fetch_open_orders
    * fetch_closed_orders

    When a WebSocket is unable to reconnect, it will be restarted and the
    associated cache will be cleared as it may no longer represent the current
    state.

    When you are finished, you should call close() to close all WebSockets (or
    use a `with` statement). If you will be exiting from Python, you must call
    twisted.internet.reactor.callFromThread(twisted.internet.reactor.stop)
    beforehand.

    """
    def __init__(self, config={}):
        super(binance, self).__init__(config)

        if "manager" not in config:
            headers = config.get("headers", {})
            self._manager = BinanceSocketManager(
                Client(self.apiKey, self.secret, headers=headers))
        else:
            self._manager = config["manager"]

        # A dict of connection key strings identifying individual WebSockets.
        self._keys = {}

        self._order_books_by_symbol = {}
        self._order_book_queues_by_symbol = {}

        # Override internal methods to use our version.
        self._parent_privateGetAccount = self.privateGetAccount
        self.privateGetAccount = self._privateGetAccount

        self._parent_publicGetDepth = self.publicGetDepth
        self.publicGetDepth = self._publicGetDepth

        self._parent_publicGetTicker24hr = self.publicGetTicker24hr
        self.publicGetTicker24hr = self._publicGetTicker24hr

        self._parent_privateGetAllOrders = self.privateGetAllOrders
        self.privateGetAllOrders = self._privateGetAllOrders

        self._parent_privateGetOpenOrders = self.privateGetOpenOrders
        self.privateGetOpenOrders = self._privateGetOpenOrders

        self._parent_privateGetOrder = self.privateGetOrder

        # Some WebSocket callbacks require markets to be loaded.
        self.load_markets()
        self._restart_ticker_socket()
        self._restart_user_socket()
        self._manager.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _restart_ticker_socket(self):
        # This is where we store the latest ticker statistics from the
        # WebSocket. It may not have statistics for all symbols though.
        self._tickers = {}

        if _TICKER_STREAM_NAME in self._keys:
            self._manager.stop_socket(self._keys[_TICKER_STREAM_NAME])

        self._keys[_TICKER_STREAM_NAME] = self._manager.start_ticker_socket(
            self._ticker_socket_callback)

    def _stop_depth_socket(self, symbol):
        stream = self.market(symbol)["id"].lower() + "@depth"
        if stream in self._keys:
            self._manager.stop_socket(self._keys[stream])
            self._order_books_by_symbol.pop(symbol, None)
            self._order_book_queues_by_symbol.pop(symbol, None)

    def _restart_depth_socket(self, symbol):
        self._stop_depth_socket(symbol)

        market = self.market(symbol)
        stream = market["id"].lower() + "@depth"

        self._order_books_by_symbol.pop(symbol, None)
        self._order_book_queues_by_symbol[symbol] = Queue.Queue()

        self._keys[stream] = self._manager.start_depth_socket(
            market["id"], partial(self._depth_socket_callback, symbol))

    def _restart_user_socket(self):
        self._account = {}
        self._orders_by_id_by_symbol = {}
        self._orders_fetched_by_symbol = set()
        self._open_orders_fetched_by_symbol = set()

        if _USER_DATA_STREAM_NAME in self._keys:
            self._manager.stop_socket(self._keys[_USER_DATA_STREAM_NAME])

        self._keys[_USER_DATA_STREAM_NAME] = self._manager.start_user_socket(
            self._user_socket_callback)

    def _ticker_socket_callback(self, message):
        if not isinstance(message, list):
            assert message["e"] == "error"
            self._logger.error(message["m"])
            return self._restart_ticker_socket()

        self._tickers.update(
            (symbol, next(ticker_generator))
            for symbol, ticker_generator in itertools.groupby(
                [
                    {
                        "symbol": ticker["s"],
                        "priceChange": ticker["p"],
                        "priceChangePercent": ticker["P"],
                        "weightedAvgPrice": ticker["w"],
                        "prevClosePrice": ticker["x"],
                        "lastPrice": ticker["c"],
                        "lastQty": ticker["Q"],
                        "bidPrice": ticker["b"],
                        "bidQty": ticker["B"],
                        "askPrice": ticker["a"],
                        "askQty": ticker["A"],
                        "openPrice": ticker["o"],
                        "highPrice": ticker["h"],
                        "lowPrice": ticker["l"],
                        "volume": ticker["v"],
                        "quoteVolume": ticker["q"],
                        "openTime": ticker["O"],
                        "closeTime": ticker["C"],
                        "firstId": ticker["F"],
                        "lastId": ticker["L"],
                        "count": ticker["n"],
                    }
                    for ticker in message
                ],
                key=lambda ticker: self.find_symbol(ticker["symbol"])
            )
        )

    def _depth_socket_callback(self, symbol, message):
        if message["e"] == "error":
            error = "Restarting {} depth socket due to error: {}"
            self._logger.error(error.format(symbol, message["m"]))
            return self._restart_depth_socket(symbol)

        queue = self._order_book_queues_by_symbol[symbol]
        queue.put(message)

        if symbol not in self._order_books_by_symbol:
            self._order_books_by_symbol[symbol] = self._parent_publicGetDepth({
                "symbol": self.market(symbol)["id"],
                "limit": 1000,
            })

        order_book = self._order_books_by_symbol[symbol]
        while not queue.empty():
            message = queue.get()
            first = message["U"]
            final = message["u"]
            last = order_book["lastUpdateId"]

            if final <= last:
                continue
            elif first <= last + 1 and final >= last + 1:
                self._patch_order_book(order_book, {
                    "bids": message["b"],
                    "asks": message["a"],
                })
                order_book["lastUpdateId"] = message["u"]
            else:
                msg = "Restarting {} depth socket due to unexpected update ID."
                self._logger.error(msg.format(symbol))
                return self._restart_depth_socket(symbol)

    def _user_socket_callback(self, message):
        if message["e"] == "error":
            self._logger.error(
                "Restarting user socket due to error: {}".format(message["m"]))
            return self._restart_user_socket()
        elif message["e"] == "outboundAccountInfo":
            return self._outbound_account_info_callback(message)
        elif message["e"] == "executionReport":
            return self._execution_report_callback(message)
        else:
            self._logger.warning("Ignoring {} event as there is no callback "
                                 "for it.".format(message["e"]))

    def _outbound_account_info_callback(self, message):
        if message["E"] < self._account.get("E", float("-inf")):
            return
        elif message["E"] == self._account.get("E", float("-inf")):
            # We're not sure if the message is in order, so refresh the account
            # information with REST.
            self._account = self._parent_privateGetAccount()
        else:
            self._account = {
                "makerCommission": message["m"],
                "takerCommission": message["t"],
                "buyerCommission": message["b"],
                "sellerCommission": message["s"],
                "canTrade": message["T"],
                "canWithdraw": message["W"],
                "canDeposit": message["D"],
                "updateTime": message["u"],
                "balances": [
                    {
                        "asset": balance["a"],
                        "free": balance["f"],
                        "locked": balance["l"],
                    }
                    for balance in message["B"]
                ],
            }

        self._account["E"] = message["E"]

    def _execution_report_callback(self, message):
        symbol = self.find_symbol(message["s"])
        orders_by_id = self._orders_by_id_by_symbol.setdefault(symbol, {})
        order = orders_by_id.setdefault(message["i"], {})

        if message["E"] < order.get("E", float("-inf")):
            return
        elif message["E"] == order.get("E", float("-inf")):
            # We're not sure if the message is in order, so refresh the order
            # with REST.
            orders_by_id[message["i"]] = self._parent_privateGetOrder({
                "symbol": message["s"],
                "orderId": message["i"],
            })
        else:
            orders_by_id[message["i"]] = {
                u"symbol": message["s"],
                u"orderId": message["i"],
                u"clientOrderId": message["c"],
                u"price": message["p"],
                u"origQty": message["q"],
                u"executedQty": message["z"],
                u"cummulativeQuoteQty": message["Z"],
                u"status": message["X"],
                u"timeInForce": message["f"],
                u"type": message["o"],
                u"side": message["S"],
                u"stopPrice": message["P"],
                u"icebergQty": message["F"],
                u"time": message["O"],
                u"updateTime": message["T"],
                u"isWorking": message["w"],
            }

        orders_by_id[message["i"]]["E"] = message["E"]

    def close(self):
        self._manager.close()

    def _privateGetAccount(self, request):
        # These arguments are passed to the underlying GET, so if they're
        # different from the defaults we use, we can't use our cached results.
        if request:
            return self._parent_privateGetAccount(request)

        if not self._account:
            return self._parent_privateGetAccount(request)

        return self._account

    def _publicGetDepth(self, request):
        """Fetch the order book, using the cache if possible.

        While the default limit is 100, the initial snapshot will be fetched
        using a limit of 1000 as it is required by the depth stream. As such,
        the weight of this REST API call is 10 rather than 1.

        Once the initial snapshot has been fetched, subsequent calls of this
        method will use the cache until the underlying WebSocket expires.

        If the WebSocket is much further ahead than the snapshot, the cache
        will be invalidated and the WebSocket restarted. You may need to call
        this method a few times before we have a snapshot that can be updated
        in lockstep with the WebSocket.

        Each symbol will open a separate WebSocket, so it is recommended not to
        fetch unnecessary order books.

        """
        symbol = self.find_symbol(request["symbol"])
        if symbol not in self._order_books_by_symbol:
            if symbol not in self._order_book_queues_by_symbol:
                self._restart_depth_socket(symbol)

            return self._parent_publicGetDepth(request)

        order_book = self._order_books_by_symbol[symbol]
        limit = request.get("limit", 100)
        return {
            "lastUpdateId": order_book["lastUpdateId"],
            "bids": heapq.nlargest(limit, order_book["bids"],
                                   operator.itemgetter(0)),
            "asks": heapq.nsmallest(limit, order_book["asks"],
                                    operator.itemgetter(0)),
        }

    def _publicGetTicker24hr(self, request):
        if "symbol" not in request:
            return list(self._tickers.values())

        symbol = self.find_symbol(request["symbol"])
        if symbol in self._tickers:
            return self._tickers[symbol]
        else:
            return self._parent_publicGetTicker24hr(request)

    def _privateGetAllOrders(self, request):
        if request.setdefault("limit", 500) == 0:
            return []

        # These arguments are passed to the underlying GET, so if they're
        # different from the defaults we use, we can't use our cached results.
        if len(set(request.keys()) - {"symbol", "limit"}) > 0:
            return self._parent_privateGetAllOrders(request)

        limit = _ALL_ORDERS_MAX_LIMIT
        symbol = self.find_symbol(request["symbol"])
        orders_by_id = self._orders_by_id_by_symbol.setdefault(symbol, {})

        # If we have never fetched orders before, we should do it at least once
        # to cache them, let the WebSocket update them and return them on
        # subsequent calls.
        if symbol not in self._orders_fetched_by_symbol:
            response = self._parent_privateGetAllOrders({
                "symbol": request["symbol"],
                "limit": limit,
            })

            if len(response) >= limit:
                message = (
                    "If you have more than {} {} orders in {}, you may not see"
                    " some of them until they are updated through the "
                    "WebSocket."
                ).format(limit, symbol, self.id)
                self._logger.warning(message)

            orders_by_id.update((order["orderId"], order)
                                for order in response)
            self._orders_fetched_by_symbol.add(symbol)

        return sorted(
            orders_by_id.values(),
            key=lambda order: order.get("time", order.get("transactTime"))
        )[-request["limit"]:]

    def _privateGetOpenOrders(self, request):
        # These arguments are passed to the underlying GET, so if they're
        # different from the defaults we use, we can't use our cached results.
        if set(request.keys()) != {"symbol"}:
            return self._parent_privateGetOpenOrders(request)

        symbol = self.find_symbol(request["symbol"])
        orders = self._orders_by_id_by_symbol.setdefault(symbol, {}).values()

        if symbol in self._orders_fetched_by_symbol:
            if len(orders) < _ALL_ORDERS_MAX_LIMIT:
                # If it's below _ALL_ORDERS_MAX_LIMIT, we can be sure we
                # already have all the open orders from a previous
                # fetch_orders() call and they are kept up to date via the
                # WebSocket.
                self._open_orders_fetched_by_symbol.add(symbol)

        # If we have previously fetched open orders before, we've already added
        # them to our cache and they are kept up to date via the WebSocket.
        if symbol in self._open_orders_fetched_by_symbol:
            return [order for order in orders
                    if self.parse_order_status(order["status"]) == "open"]

        # If we're not sure whether we have all the open orders, we will need
        # to perform a REST API call, which does not limit the number of
        # results returned.
        response = self._parent_privateGetOpenOrders({
            "symbol": request["symbol"],
        })
        self._orders_by_id_by_symbol[symbol].update((order["orderId"], order)
                                                    for order in response)
        self._open_orders_fetched_by_symbol.add(symbol)
        return response
