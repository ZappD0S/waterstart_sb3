from collections.abc import Sequence
from typing import Optional

import gym
import numpy as np
from gym import spaces
from numpy.lib.stride_tricks import sliding_window_view
from waterstart.inference.utils import compute_min_step_max_arr
from waterstart.training.traindata import TrainingData

from .trades_state import TradesState


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        training_data: TrainingData,
        max_trades: int,
        window_size: int,
        initial_balance: float = 1e5,
        leverage: float = 1.0,
    ):
        super().__init__()

        self._initial_balance = initial_balance
        self._initial_trades_state = TradesState(
            np.zeros((max_trades, training_data.n_traded_sym)),
            np.zeros((max_trades, training_data.n_traded_sym)),
        )

        self._balance = self._initial_balance
        self._trades_state = self._initial_trades_state

        # TODO: data like spreads, time of day and delta to last must already be globally normalized
        self._windowed_market_data = sliding_window_view(
            training_data.market_data, window_size, axis=0
        )

        half_spreads = training_data.spreads / 2
        self._ask_prices = training_data.midpoint_prices + half_spreads
        self._bid_prices = training_data.midpoint_prices - half_spreads
        self._margin_rates = training_data.base_to_dep_rates
        self._quote_to_dep_rates = training_data.quote_to_dep_rates

        self._index: Optional[int] = None

        self._window_size = window_size
        self.leverage = leverage
        self.n_timesteps = training_data.n_timesteps

        self._min_step_max = compute_min_step_max_arr(
            training_data.traded_symbols, training_data.traded_sym_arr_mapper
        )
        self._scaling_idxs_arr = self._build_scaling_idxs_arr(
            training_data.market_data_arr_mapper.scaling_idxs
        )

        self.action_space = spaces.Box(-1, 1, (2, training_data.n_traded_sym))
        self.observation_space = spaces.Dict(
            {
                # midpoint prices, spreads, base_to_dep, quote_to_dep, delta to last trading t, time of day
                "market_data": spaces.Box(
                    -5, 5, shape=self._windowed_market_data.shape[1:]
                ),
                "trades_rel_sizes": spaces.Box(
                    -1, 1, shape=(max_trades, training_data.n_traded_sym)
                ),
                "trades_log_pl": spaces.Box(
                    np.log(0.5),
                    np.log1p(150),
                    shape=(max_trades, training_data.n_traded_sym),
                ),
            }
        )

    @staticmethod
    def _build_scaling_idxs_arr(
        scaling_idxs: Sequence[tuple[int, list[int]]]
    ) -> np.ndarray:
        flat_idxs = [
            ([src_ind] * len(dst_inds), dst_inds) for src_ind, dst_inds in scaling_idxs
        ]

        idxs_arr = np.array(flat_idxs).swapaxes(1, 0)
        assert idxs_arr.shape[0] == 2
        # idxs_arr.shape = (2, -1)
        idxs_arr = idxs_arr.reshape((2, -1))
        return idxs_arr

    def _compute_new_pos_sizes(
        self, action: np.ndarray, pos_margins: np.ndarray
    ) -> np.ndarray:
        signs, fractions = action
        exec_mask = signs != 0

        unused_margin = self._balance - pos_margins.sum()

        new_margins = self._balance * fractions

        required_margin = new_margins[exec_mask].sum()
        available_margin = unused_margin + pos_margins[exec_mask].sum()

        assert available_margin >= 0

        margin_excess = np.maximum(required_margin - available_margin, 0)

        new_margins[exec_mask] -= (
            margin_excess * new_margins[exec_mask] / required_margin
        )
        assert new_margins[exec_mask].sum() <= available_margin
        assert np.all(new_margins >= 0)

        new_pos_sizes = new_margins * self._margin_rates[self._index] * self.leverage

        min_step_max = self._min_step_max

        new_pos_sizes = np.floor(new_pos_sizes / min_step_max.step) * min_step_max.step

        new_pos_sizes = np.where(new_pos_sizes < min_step_max.min, 0, new_pos_sizes)
        new_pos_sizes = np.minimum(new_pos_sizes, min_step_max.max)

        new_pos_sizes = signs * new_pos_sizes
        new_pos_sizes = np.where(exec_mask, new_pos_sizes, self._trades_state.pos_size)

        return new_pos_sizes

    def _compute_profit_loss(
        self, closed_trades_sizes: np.ndarray, closed_trades_prices: np.ndarray
    ) -> float:
        i = self._index

        closed_size = closed_trades_sizes.sum(0)
        close_price = np.where(
            closed_size > 0,
            self._bid_prices[i],
            np.where(closed_size < 0, self._ask_prices[i], 0),
        )

        profit_loss = np.sum(
            closed_trades_sizes
            * (close_price - closed_trades_prices)
            / self._quote_to_dep_rates[i]
        )

        return profit_loss

    def _build_obs(self) -> dict[str, np.ndarray]:
        # NOTE: this is fine because when we get here the index should have already been incremented
        assert self._index is not None
        i = self._index

        virtual_profit_loss = self._compute_profit_loss(
            self._trades_state.trades_sizes, self._trades_state.trades_prices
        )

        dep_cur_pos_sizes = self._trades_state.pos_size / self._margin_rates[i]

        rel_margins = dep_cur_pos_sizes / (self.leverage * self._balance)

        assert rel_margins.sum() <= 1

        log_pl = np.log1p(virtual_profit_loss / dep_cur_pos_sizes)

        market_data = self._windowed_market_data[i - self._window_size + 1]

        src_idxs, dst_idxs = self._scaling_idxs_arr
        scaled_market_data = market_data.copy()
        scaled_market_data[dst_idxs] = np.log(scaled_market_data[dst_idxs]) - np.log(
            scaled_market_data[src_idxs, -1, None]
        )

        return {
            "market_data": scaled_market_data,
            "trades_rel_sizes": rel_margins,
            "trades_log_pl": log_pl,
        }

    def step(self, action: np.ndarray):
        assert self._index is not None
        i = self._index

        done = False

        virtual_profit_loss = self._compute_profit_loss(
            self._trades_state.trades_sizes, self._trades_state.trades_prices
        )

        account_value = self._balance + virtual_profit_loss

        pos_margins = np.abs(self._trades_state.pos_size) / (
            self._margin_rates[i] * self.leverage
        )

        closeout = account_value < 0.5 * pos_margins.sum()

        if closeout:
            new_pos_sizes = np.zeros_like(self._trades_state.pos_size)
            done = True
        else:
            new_pos_sizes = self._compute_new_pos_sizes(action, pos_margins)

        trades_state, closed_trades_sizes = self._trades_state.close_or_reduce_trades(
            new_pos_sizes
        )

        profit_loss = self._compute_profit_loss(
            closed_trades_sizes, self._trades_state.trades_prices
        )

        self._balance += profit_loss
        reward = np.log1p(profit_loss / self._balance)

        # TODO: if new_balance is smaller than a given fraction of initial balance, done = True

        open_sizes: np.ndarray
        open_sizes = new_pos_sizes - trades_state.pos_size  # type: ignore

        open_price = np.where(
            open_sizes > 0,
            self._ask_prices[i],
            np.where(open_sizes < 0, self._bid_prices[i], np.nan),
        )

        self._trades_state, _ = trades_state.open_trades(new_pos_sizes, open_price)

        if i == self.n_timesteps - 1:
            self._index = self._window_size - 1
            done = True
        else:
            self._index += 1

        return self._build_obs(), reward, done, {}

    def reset(self):
        if self._index is None:
            self._index = np.random.randint(self._window_size - 1, self.n_timesteps)

        self._balance = self._initial_balance
        self._trades_state = self._initial_trades_state

        return self._build_obs()

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        pass

    # def seed(self, seed):
    #     ...
