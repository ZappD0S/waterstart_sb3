from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from waterstart.training.traindata import load_training_data

from .policy import TradingPolicy
from .trading_env import TradingEnv

max_trades = 10
window_size = 100
n_envs = 4

training_data = load_training_data("../financelab/train_data.npz")

env = DummyVecEnv(
    [lambda: TradingEnv(training_data, max_trades=max_trades, window_size=window_size)]
    * n_envs
)

features_extractor_kwargs = {
    "max_trades": max_trades,
    "n_traded_sym": training_data.n_traded_sym,
    "market_features_dim": training_data.n_market_features,
    "out_features_dim": 256,
    "window_size": window_size,
}

policy_kwargs = {
    "net_arch": [50, 50, 20],
    "features_extractor_kwargs": features_extractor_kwargs,
}

model = PPO(TradingPolicy, env, policy_kwargs=policy_kwargs, device="cpu")

model.learn(100_000)
