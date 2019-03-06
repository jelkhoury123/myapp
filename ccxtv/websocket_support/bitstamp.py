import collections
import datetime
from functools import partial
import json
import time

from pusherclient import Connection, Pusher

from ccxt.base.errors import NetworkError
import ccxt.bitstamp
from ccxt.websocket_support.exchange import Exchange

# Python 3 compatibility.
try:
    import Queue
except ImportError:
    import queue as Queue


def _epoch(timedelta):
    utcnow = datetime.datetime.utcnow()
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (utcnow + timedelta - epoch).total_seconds()


class bitstamp(Exchange, ccxt.bitstamp):
    """A WebSocket-enabled, drop-in replacement for the ccxt Bitstamp client.

    The following methods are WebSocket-enabled:
    * fetch_order_book
    * fetch_trades

    When a WebSocket is unable to reconnect, it will be restarted and the
    associated cache will be cleared as it may no longer represent the current
    state.

    When you are finished, you should call close() to close all WebSockets (or
    use a `with` statement).

    """
    def __init__(self, config={}):
        super(bitstamp, self).__init__(config)
        self.load_markets()

        self._order_book_queues_by_symbol = {}
        self._orders_by_id = {}
        self._trade_queues_by_symbol = {}

        self._pusher = config.get("pusher", Pusher("de504dc5763aeef9ff52"))
        self._pusher.connection.logger = self._logger

        # Workaround for incompatibility with the latest websocket-client.
        c = self._pusher.connection
        c._on_open = lambda *args: Connection._on_open(*((c,) + args))
        c._on_message = lambda *args: Connection._on_message(*((c,) + args))
        c._on_error = lambda *args: Connection._on_error(*((c,) + args))
        c._on_close = lambda *args: Connection._on_close(*((c,) + args))

        def clear(bitstamp):
            self._order_book_queues_by_symbol.clear()
            self._trade_queues_by_symbol.clear()

            attributes = [
                "_order_books_by_symbol",
                "_trades_by_symbol",
            ]
            for attribute in attributes:
                if hasattr(self, attribute):
                    delattr(self, attribute)

            return c

        # Hack to delete internal attributes if the WebSocket closes.
        c.disconnect = lambda: Connection.disconnect(clear(self))
        c.reconnect = lambda r=None: Connection.reconnect(*(clear(self), r))

        c.bind("pusher:connection_established", self._subscribe_channels)
        self._pusher.connect()

        self._parent_publicGetOrderBookPair = self.publicGetOrderBookPair
        self.publicGetOrderBookPair = self._publicGetOrderBookPair

        self._parent_publicGetTransactionsPair = self.publicGetTransactionsPair
        self.publicGetTransactionsPair = self._publicGetTransactionsPair

        # Since the nonce is expressed in seconds, making private API calls
        # repeatedly may result in invalid nonce errors.
        #
        # ccxt's rate limiter only records the time when throttling happens,
        # not the actual nonce used, so collisions are still possible.
        #
        # As such, we will wrap private API calls with our throttling function.
        # Another advantage over ccxt's rate limiter is public API calls are
        # unaffected.
        self._previous_nonce = None

        def throttle(function):
            def throttled(*args, **kwargs):
                while self.nonce() == self._previous_nonce:
                    time.sleep(1 - (time.time() % 1))

                self._previous_nonce = self.nonce()
                return function(*args, **kwargs)

            return throttled

        names = (name for name in dir(self)
                 if name[:7] == "private" and callable(getattr(self, name)))
        for name in names:
            setattr(self, name, throttle(getattr(self, name)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _subscribe_channels(self, _):
        self._order_books_by_symbol = {}
        self._trades_by_symbol = {}

        for symbol, market in self.markets.items():
            if symbol == "BTC/USD":
                pair = ""
            else:
                pair = "_" + market["id"]

            self._pusher.subscribe("diff_order_book" + pair).bind(
                "data", partial(self._diff_order_book_callback, symbol))
            self._pusher.subscribe("live_trades" + pair).bind(
                "trade", partial(self._live_trades_callback, symbol))

    def _diff_order_book_callback(self, symbol, message):
        queue = self._order_book_queues_by_symbol.setdefault(symbol,
                                                             Queue.Queue())
        queue.put(message)

        if symbol not in self._order_books_by_symbol:
            return

        order_book = self._order_books_by_symbol[symbol]
        while not queue.empty():
            message = json.loads(queue.get())

            # There's no guidance on how far apart the timestamps can be, so we
            # will accept any message as long as they are in order.
            #
            # If we are temporarily disconnected, the WebSocket should be able
            # to catch up with previous messages.
            #
            # If there's an unrecoverable error with the WebSocket (which seems
            # to be the only case where we might miss diffs), the cache will be
            # invalidated as per the method overrides for Pusher.disconnect()
            # and Pusher.reconnect().
            if int(message["timestamp"]) <= int(order_book["timestamp"]):
                continue

            self._patch_order_book(order_book, message)
            order_book["timestamp"] = message["timestamp"]

    def _live_trades_callback(self, symbol, message):
        queue = self._trade_queues_by_symbol.setdefault(symbol, Queue.Queue())
        queue.put(message)

        if symbol not in self._trades_by_symbol:
            return

        trades = self._trades_by_symbol[symbol]
        while not queue.empty():
            message = json.loads(queue.get())
            trade = {
                "date": message["timestamp"],
                "tid": str(message["id"]),
                "price": message["price_str"],
                "type": str(message["type"]),
                "amount": message["amount_str"],
            }

            if trade["date"] >= trades[0]["date"] and trade != trades[0]:
                trades.appendleft(trade)

            # The upper bound of the deque's storage is the maximum number of
            # trades within any 24-hour period.
            if int(trades[-1]["date"]) < _epoch(datetime.timedelta(days=-1)):
                trades.pop()

    def _publicGetOrderBookPair(self, request):
        # These arguments are passed to the underlying GET, so if they're
        # different from the defaults we use, we can't use our cached results.
        if request.get("group", 1) != 1:
            return self._parent_publicGetOrderBookPair(request)

        # The WebSocket isn't open yet.
        if not hasattr(self, "_order_books_by_symbol"):
            return self._parent_publicGetOrderBookPair(request)

        symbol = self.find_symbol(request["pair"])
        if symbol not in self._order_books_by_symbol:
            order_book = self._parent_publicGetOrderBookPair(request)
            self._order_books_by_symbol[symbol] = order_book

        return self._order_books_by_symbol[symbol]

    def _publicGetTransactionsPair(self, request):
        # The WebSocket isn't open yet.
        if not hasattr(self, "_trades_by_symbol"):
            return self._parent_publicGetTransactionsPair(request)

        symbol = self.find_symbol(request["pair"])
        if symbol not in self._trades_by_symbol:
            self._trades_by_symbol[symbol] = collections.deque(
                self._parent_publicGetTransactionsPair({
                    "pair": request["pair"],
                    "time": "day",
                })
            )

        # We will need to get trades that are greater than or equal to this
        # timestamp.
        minimum_timestamp = _epoch({
            "minute": datetime.timedelta(minutes=-1),
            "hour": datetime.timedelta(hours=-1),
            "day": datetime.timedelta(days=-1),
        }[request.get("time", "hour")])

        # Since we'll be bisecting and slicing, we need a random access data
        # structure.
        trades = list(self._trades_by_symbol[symbol])

        if len(trades) > 0:
            low = 0
            high = len(trades)
            while low < high:
                middle = (low + high) // 2
                if minimum_timestamp > int(trades[middle]["date"]):
                    high = middle
                else:
                    low = middle + 1

            trades = trades[:low]

        return trades

    def close(self):
        self._pusher.disconnect()

    def create_order(self, *args, **kwargs):
        order = ccxt.bitstamp.create_order(self, *args, **kwargs)
        self._orders_by_id[int(order["id"])] = order
        return order

    def cancel_order(self, *args, **kwargs):
        cancelation = ccxt.bitstamp.cancel_order(self, *args, **kwargs)
        order = self._orders_by_id.setdefault(int(cancelation["id"]), {})
        order.setdefault("info", {}).update(cancelation)
        return cancelation

    def describe(self):
        return self.deep_extend(super(bitstamp, self).describe(), {
            "has": {
                "fetchOrders": "emulated",
                "fetchClosedOrders": "emulated",
            },
        })

    def fetch_closed_orders(self, symbol=None, since=None, limit=None,
                            params={}):
        """Fetch all closed and canceled orders.

        This relies on fetch_orders(). Please read its docstring to understand
        the performance implications.

        """
        return [
            order for order in self.fetch_orders(symbol, since, limit, params)
            if order["status"] in ("closed", "canceled")
        ]

    def fetch_orders(self, symbol=None, since=None, limit=None, params={}):
        """Fetch all orders.

        This is an API call intensive operation which involves fetching all
        trades for all currencies to collect order IDs and then fetching the
        order status for each of these order ID.

        The reason for this is because there is no API for fetching all orders
        or non-open orders.

        Due to the nature of the nonce, each private API call needs to be at
        least one second apart. As such, you should keep in mind that this
        entire operation takes linear time with the unit in seconds. The growth
        rate is dependent on how many orders there are and how many trades
        filled them.

        """
        if symbol is None:
            symbols = self.markets.keys()
        else:
            symbols = [symbol]

        # Multiple transactions may refer to the same order ID, so we need to
        # keep track of the unique ones.
        order_ids_by_symbol = {}

        # It will be a waste of effort if we abort this operation due to a
        # temporary error, so let's abort only if we encounter a consecutive
        # number of errors.
        retries = 0

        # We need to know which symbol the trade is referring to, as it is not
        # provided in the response and ccxt's inference is not perfect.
        #
        # This will be filled in by fetch_order() later.
        for symbol in symbols:
            offset = 0

            # We don't know how many transactions there are, so we will loop
            # until the number of trades is less than the limit argument.
            while True:
                try:
                    trades = self.fetch_my_trades(symbol, params={
                        "offset": offset,
                        "limit": 1000,
                    })
                    retries = 0
                except NetworkError:
                    retries += 1
                    if retries == 10:
                        raise
                    else:
                        continue

                order_ids = order_ids_by_symbol.setdefault(symbol, set())
                order_ids.update(trade["order"] for trade in trades)

                if len(trades) == 1000:
                    offset += 1000
                else:
                    break

        orders = []
        for symbol, order_ids in order_ids_by_symbol.items():
            for order_id in order_ids:
                while True:
                    try:
                        orders.append(self.fetch_order(order_id, symbol))
                        retries = 0
                        break
                    except NetworkError:
                        retries += 1
                        if retries == 10:
                            raise
                        else:
                            continue

        return self.filter_by_since_limit(orders, since, limit)

    def fetch_order(self, id, symbol=None, params={}):
        """Fetch an order, augmenting it with data from our cache if possible.

        Sometimes fetching an order yields less information than when it was
        created. As such, we override create_order() and cancel_order() to
        cache their return values.

        """
        # These arguments are passed to the underlying POST, so if they're
        # different from the defaults we use, we can't use our cached results.
        if len(params) > 0:
            return ccxt.bitstamp.fetch_order(self, id, symbol, params)

        order_id = int(id)
        order = self._orders_by_id.get(order_id, {})

        status = order.get("status")
        if self.parse_order_status(status) in ("closed", "canceled"):
            return order

        info = order.get("info", {})
        info.update(ccxt.bitstamp.fetch_order(self, id, symbol)["info"])

        market = self.market(order["symbol"]) if order.get("symbol") else None
        artificial = ccxt.bitstamp.parse_order(self, info, market)

        if order.get("type"):
            artificial["type"] = order["type"]

        if artificial.get("side") is None and len(artificial["trades"]) > 0:
            sides = set(trade["side"] for trade in artificial["trades"])
            if len(sides) == 1:
                artificial["side"] = sides.pop()

        self._orders_by_id[order_id] = artificial
        return artificial

    def fetch_trades(self, *args, **kwargs):
        trades = ccxt.bitstamp.fetch_trades(self, *args, **kwargs)

        # There is a bug in ccxt.bitstamp.parse_trade() where it assumes GET
        # /transactions uses amount to indicate side like POST
        # /user_transactions.
        for trade in trades:
            if trade["info"]["type"] == "1":
                trade["side"] = "sell"

        return trades

    def parse_order(self, order, market=None, since=None, limit=None):
        # Some methods in ccxt.bitstamp are passing in extra arguments.
        return ccxt.bitstamp.parse_order(self, order, market)
