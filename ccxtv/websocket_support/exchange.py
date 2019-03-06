import heapq
import itertools
import logging
import operator
import sys


class Exchange(object):
    def __init__(self, config={}):
        super(Exchange, self).__init__(config)

        if "logger" not in config:
            handler = logging.StreamHandler(sys.stderr)
            f = "%(created)d:%(levelname)s:%(module)s:%(funcName)s:%(message)s"
            handler.setFormatter(logging.Formatter(f))

            self._logger = logging.getLogger(self.id)
            self._logger.addHandler(handler)
        else:
            self._logger = config["logger"]

    @classmethod
    def _patch_order_book(cls, snapshot, diff):
        bids_to_del = set(bid[0] for bid in diff["bids"] if float(bid[1]) == 0)
        bids_to_patch = dict(
            (price, next(bid_generator))
            for price, bid_generator in itertools.groupby(
                [bid for bid in diff["bids"] if float(bid[1]) > 0],
                key=operator.itemgetter(0)
            )
        )
        snapshot["bids"] = list(heapq.merge(
            [
                bid for bid in snapshot["bids"]
                if bid[0] not in bids_to_del and bid[0] not in bids_to_patch
            ],
            bids_to_patch.values(),
        ))

        asks_to_del = set(ask[0] for ask in diff["asks"] if float(ask[1]) == 0)
        asks_to_patch = dict(
            (price, next(ask_generator))
            for price, ask_generator in itertools.groupby(
                [ask for ask in diff["asks"] if float(ask[1]) > 0],
                key=operator.itemgetter(0)
            )
        )
        snapshot["asks"] = list(heapq.merge(
            [
                ask for ask in snapshot["asks"]
                if ask[0] not in asks_to_del and ask[0] not in asks_to_patch
            ],
            asks_to_patch.values(),
        ))
