import bitmex_websocket

import ccxt.bitmex

# Python 2 compatibility.
try:
    import urllib.parse
except ImportError:
    import urllib
    import urlparse
    urllib.parse = lambda: None
    urllib.parse.urlparse = urlparse.urlparse
    urllib.parse.urlunparse = urlparse.urlunparse


class BitMEXWebsocket(bitmex_websocket.BitMEXWebsocket):
    """A BitMEXWebsocket class that supports subscribing to custom topics.

    This allows us to reuse the same connection for topics other than the ones
    hard-coded in bitmex_websocket.BitMEXWebsocket.

    As a result, __wait_for_account() and __wait_for_symbol() have been
    replaced by stubs. You will need to check whether BitMEXWebsocket.data
    contains the data you're looking for.

    """
    def __init__(self, endpoint, subscriptions, api_key, api_secret):
        self.subscriptions = subscriptions
        bitmex_websocket.BitMEXWebsocket.__init__(self, endpoint, None,
                                                  api_key, api_secret)

    def __get_url(self):
        components = list(urllib.parse.urlparse(self.endpoint))
        components[0] = components[0].replace("http", "ws")
        components[2] = "/realtime?subscribe=" + ",".join(self.subscriptions)
        return urllib.parse.urlunparse(components)

    def __wait_for_account(self):
        pass

    def __wait_for_symbol(self, symbol):
        pass


class bitmex(ccxt.bitmex):
    """A WebSocket-enabled, drop-in replacement for the ccxt BitMEX client.

    The following methods are WebSocket-enabled:
    * fetch_order_book
    * fetch_balance
    * fetch_open_orders

    The following methods are not:
    * fetch_ticker
      * ccxt uses /trade/bucketed by 1-day time intervals to aggregate ticker
        information as of today.
      * The "tradeBin1d" topic only sends a message once the day has ended.
      * The "instrument" topic seems to match the 1d chart on the website, but
        not /trade/bucketed with 1d binSize.

    When you are finished, you should call close() to close all WebSockets (or
    use a `with` statement).

    """
    def __init__(self, config={}):
        super(bitmex, self).__init__(config)
        self.load_markets()

        self._websocket = BitMEXWebsocket(
            "{}/api/{}".format(self.urls["api"], self.version),
            ["margin", "order", "orderBookL2"],
            self.apiKey if len(self.apiKey) > 0 else None,
            self.secret if len(self.secret) > 0 else None
        )

        self._parent_privateGetUserMargin = self.privateGetUserMargin
        self.privateGetUserMargin = self._privateGetUserMargin

        self._parent_publicGetOrderBookL2 = self.publicGetOrderBookL2
        self.publicGetOrderBookL2 = self._publicGetOrderBookL2

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _privateGetUserMargin(self, request):
        if "margin" in self._websocket.data:
            return self._websocket.data["margin"]
        else:
            return self._parent_privateGetUserMargin(request)

    def _publicGetOrderBookL2(self, request):
        if "orderBookL2" in self._websocket.data:
            market_depth = [
                order for order in self._websocket.data["orderBookL2"]
                if order["symbol"] == request["symbol"]
            ]
        else:
            return self._parent_publicGetOrderBookL2(request)

        if "depth" not in request or request["depth"] == 0:
            return market_depth

        # Find the index that bifurcates the list into buys and sells.
        low = 0
        high = len(market_depth)
        while low < high:
            middle = (low + high) // 2
            if market_depth[middle]["side"] != "Buy":
                low = middle + 1
            else:
                high = middle

        return market_depth[
            max(low - request["depth"], 0):
            min(low + request["depth"], len(market_depth))
        ]

    def close(self):
        self._websocket.exit()

    def fetch_open_orders(self, symbol=None, since=None, limit=None,
                          params={}):
        # These arguments are passed to the underlying GET, so if they're
        # different from the defaults we use, we can't use our cached results.
        if since is not None or limit is not None or len(params) > 0:
            return ccxt.bitmex(self, symbol, since, limit, params)

        if "order" not in self._websocket.data:
            return ccxt.bitmex(self, symbol, since, limit, params)

        # The WebSocket will also fetch open orders that existed before it was
        # initialized.
        orders = self.parse_orders(self._websocket.open_orders(""))

        if symbol is not None:
            orders = [order for order in orders
                      if order["symbol"] == self.market(symbol)["id"]]

        return orders
