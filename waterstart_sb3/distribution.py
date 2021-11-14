from stable_baselines3.common.distributions import Distribution

import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.distributions as dist


# Either Categorical (3 possible outcomes, -1, 0, 1) or two
# Bernoulli (0,1 for the execution and -1, 1 for the sign)

# In the exp() before Dirichlet causes trouble, try with Normal + StickBreaking


class DistributionNet(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.lin_dirich = nn.Linear(latent_dim, action_dim)
        self.lin_categ = nn.Linear(latent_dim, 3 * action_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        dirich_action_logits = self.lin_dirich(latent).exp()
        categ_action_logits = self.lin_categ(latent).reshape(3, -1)

        return torch.stack((dirich_action_logits, categ_action_logits))


class MarginFractionsDistribution(Distribution):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return DistributionNet(latent_dim, self.action_dim)

    def proba_distribution(
        self, action_logits: torch.Tensor
    ) -> "MarginFractionsDistribution":
        conc, categ_logits = action_logits.split((1, 3))
        self.frac_dist = dist.Dirichlet(conc)

        self.action_type_dist = dist.TransformedDistribution(
            dist.Categorical(logits=categ_logits.T),
            [dist.transforms.AffineTransform(-1, 1)],
        )

        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        fracs, action_types = actions.unbind()
        return self.frac_dist.log_prob(fracs) + self.action_type_dist.log_prob(
            action_types
        )

    def entropy(self) -> Optional[torch.Tensor]:
        return self.frac_dist.entropy() + self.action_type_dist.entropy()

    def sample(self) -> torch.Tensor:
        return torch.stack((self.frac_dist.sample(), self.action_type_dist.sample()))

    def mode(self) -> torch.Tensor:
        return torch.stack((self.frac_dist.mean, self.action_type_dist.mean))

    def actions_from_params(
        self, action_logits: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, action_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
