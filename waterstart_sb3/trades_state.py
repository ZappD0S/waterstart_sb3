from __future__ import annotations

from functools import cached_property

import numpy as np


class TradesState:
    def __init__(self, trades_sizes: np.ndarray, trades_prices: np.ndarray) -> None:
        self.trades_sizes = trades_sizes
        self.trades_prices = trades_prices

    # TODO: make the naming convention uniform
    @cached_property
    def pos_size(self) -> np.ndarray:
        return self.trades_sizes.sum(0)

    @cached_property
    def available_trades_mask(self) -> np.ndarray:
        # return self.trades_sizes[0] == 0
        return self.trades_sizes[-1] == 0

    # TODO: make the following methods part of the AccountState class?
    @staticmethod
    def _get_validity_mask(sizes_or_trades: np.ndarray) -> np.ndarray:
        trade_open_mask = sizes_or_trades != 0

        mask_cumsum = trade_open_mask.cumsum(0)
        open_trades_counts = mask_cumsum

        expected_open_trades_counts = np.arange(1, open_trades_counts.shape[0] + 1)[
            (slice(None),) + (None,) * (open_trades_counts.ndim - 1)
        ]

        all_following_trades_open = open_trades_counts == expected_open_trades_counts
        return np.all(~trade_open_mask | all_following_trades_open, axis=0)

    def open_trades(
        self, new_pos_size: np.ndarray, open_price: np.ndarray
    ) -> tuple[TradesState, np.ndarray]:
        pos_size = self.pos_size
        trades_sizes = self.trades_sizes
        trades_prices = self.trades_prices
        available_trade_mask = self.available_trades_mask

        open_trade_size = new_pos_size - pos_size
        new_pos_mask = (pos_size == 0) & (open_trade_size != 0)
        # new_pos_mask = (pos_size == 0) & (new_pos_size != 0)
        # new_pos_mask = pos_size.isclose(pos_size.new_zeros(())) & ~new_pos_size.isclose(
        #     new_pos_size.new_zeros(())
        # )

        valid_trade_mask = new_pos_mask | (pos_size * open_trade_size > 0)
        assert np.all(valid_trade_mask | (open_trade_size == 0))
        # assert np.all(valid_trade_mask | np.isclose(new_pos_size, pos_size))
        # assert np.all(
        #     valid_trade_mask | open_trade_size.isclose(open_trade_size.new_zeros(()))
        # )
        open_trade_mask = available_trade_mask & valid_trade_mask

        new_trades_sizes = trades_sizes.copy()
        # new_trades_sizes[:-1] = trades_sizes[1:]
        # new_trades_sizes[-1] = open_trade_size
        new_trades_sizes[1:] = trades_sizes[:-1]
        new_trades_sizes[0] = open_trade_size

        new_trades_sizes = np.where(open_trade_mask, new_trades_sizes, trades_sizes)

        new_trades_prices = trades_prices.copy()
        # new_trades_prices[:-1] = trades_prices[1:]
        # new_trades_prices[-1] = open_price
        new_trades_prices[1:] = trades_prices[:-1]
        new_trades_prices[0] = open_price

        new_trades_prices = np.where(open_trade_mask, new_trades_prices, trades_prices)

        assert not new_trades_prices.isnan().any()
        assert not np.any((new_trades_sizes == 0) != (new_trades_prices == 0))
        assert np.all(
            np.all(new_trades_sizes >= 0, axis=0)
            | np.all(new_trades_sizes <= 0, axis=0)
        )
        assert self._get_validity_mask(new_trades_sizes).all()

        new_account_state = TradesState(
            trades_sizes=new_trades_sizes, trades_prices=new_trades_prices
        )
        assert np.all(
            ~open_trade_mask
            | (new_account_state.pos_size == new_pos_size)
            # ~open_trade_mask
            # | np.isclose(new_account_state.pos_size, new_pos_size)
        )
        return new_account_state, open_trade_mask

    def close_or_reduce_trades(
        self, new_pos_size: np.ndarray
    ) -> tuple[TradesState, np.ndarray]:
        trades_sizes = self.trades_sizes
        trades_prices = self.trades_prices
        pos_size = self.pos_size

        # close_trade_size = new_pos_size - pos_size
        # right_diffs = close_trade_size + trades_sizes.cumsum(0)
        right_diffs: np.ndarray
        right_diffs = trades_sizes.cumsum(0) - new_pos_size  # type: ignore

        left_diffs = np.empty_like(right_diffs)
        left_diffs[1:] = right_diffs[:-1]
        # left_diffs[0] = close_trade_size
        left_diffs[0] = -new_pos_size

        # close_trade_mask = (pos_size != 0) & (pos_size * right_diffs <= 0)
        close_trade_mask = (pos_size != 0) & (pos_size * left_diffs >= 0)
        reduce_trade_mask = left_diffs * right_diffs < 0

        closed_trades_sizes = np.zeros_like(trades_sizes)

        assert not np.any(close_trade_mask & reduce_trade_mask)
        assert np.all(reduce_trade_mask.sum(0) <= 1)

        new_trades_sizes = trades_sizes.copy()
        new_trades_sizes[close_trade_mask] = 0.0
        # new_trades_sizes[reduce_trade_mask] = right_diffs[reduce_trade_mask]
        new_trades_sizes[reduce_trade_mask] = -left_diffs[reduce_trade_mask]

        closed_trades_sizes[close_trade_mask] = trades_sizes[close_trade_mask]
        # closed_trades_sizes[reduce_trade_mask] = -left_diffs[reduce_trade_mask]
        closed_trades_sizes[reduce_trade_mask] = right_diffs[reduce_trade_mask]

        new_trades_prices = trades_prices.copy()
        new_trades_prices[close_trade_mask] = 0.0

        assert not np.any((new_trades_sizes == 0) != (new_trades_prices == 0))
        assert np.all(
            np.all((new_trades_sizes >= 0) & (closed_trades_sizes >= 0), axis=0)
            | np.all((new_trades_sizes <= 0) & (closed_trades_sizes <= 0), axis=0)
        )
        assert self._get_validity_mask(new_trades_sizes).all()

        new_account_state = TradesState(
            trades_sizes=new_trades_sizes, trades_prices=new_trades_prices
        )

        close_or_reduce_mask = np.any(close_trade_mask | reduce_trade_mask, axis=0)
        close_pos_size = closed_trades_sizes.sum(0)
        assert np.all(pos_size == new_account_state.pos_size + close_pos_size)
        # assert np.isclose(
        #     pos_size, new_account_state.pos_size + close_pos_size
        # ).all()

        assert np.all(
            ~close_or_reduce_mask
            | (pos_size * new_pos_size > 0)
            | (new_account_state.pos_size == 0)
        )

        assert np.all(
            ~close_or_reduce_mask
            | (pos_size * new_pos_size < 0)
            | (new_account_state.pos_size == new_pos_size)
            # | np.isclose(new_account_state.pos_size, new_pos_size)
        )

        assert np.all(new_account_state.pos_size * new_pos_size >= 0)

        # assert np.all(
        #     ~close_or_reduce_mask
        #     | (
        #         close_pos_size
        #         == np.where(
        #             close_trade_size.abs() > pos_size.abs(),
        #             pos_size,
        #             -close_trade_size,
        #         )
        #     )
        # )

        return new_account_state, closed_trades_sizes
