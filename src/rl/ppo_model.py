import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)

        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,)
        )

        return mean, log_std


class ValueNetwork(nn.Module):

    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        return x.squeeze(-1)


def build_policy(obs):
    model = PolicyNetwork(action_dim=2)
    params = model.init(jax.random.PRNGKey(0), obs)
    return model, params


def build_value(obs):
    model = ValueNetwork()
    params = model.init(jax.random.PRNGKey(0), obs)
    return model, params
