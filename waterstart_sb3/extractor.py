from math import log2

import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MarketDataExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        max_trades: int,
        n_traded_sym: int,
        market_features_dim: int,
        out_features_dim: int,
        window_size: int,
    ):
        super().__init__(observation_space, market_features_dim)
        self.kernel_size = 3
        hidden_dim = 2 ** max(5, round(log2(market_features_dim)))

        self.conv1 = nn.Conv1d(
            market_features_dim, market_features_dim, kernel_size=self.kernel_size
        )
        self.conv2 = nn.Conv1d(
            market_features_dim,
            hidden_dim,
            kernel_size=window_size + 1 - self.kernel_size,
        )
        # TODO: more convolutions?

        self.lin1 = nn.Linear(2 * max_trades * n_traded_sym, hidden_dim)
        self.lin2 = nn.Linear(2 * hidden_dim, out_features_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (2, batch_size, max_trades, n_traded_syms)
        trades_data = torch.stack(
            (observations["trades_rel_sizes"], observations["trades_log_pl"])
        )

        # shape: batch_size, n_traded_syms, max_trades, 2
        trades_data = trades_data.permute(1, 3, 2, 0).flatten(2, -1)

        out1: torch.Tensor = self.conv1(observations["market_data"]).relu()
        out1 = self.conv2(out1).squeeze(2)

        out2: torch.Tensor = self.lin1(trades_data)

        out = torch.cat((out1, out2), dim=1).relu()
        out = self.lin2(out).relu()

        return out
