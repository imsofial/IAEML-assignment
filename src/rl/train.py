import jax
import jax.numpy as jnp
from typing import Dict

def flatten_obs(obs):
    obs_vec = jnp.concatenate([
        obs.distance_to_path.reshape(-1),
        obs.direction_of_path.reshape(-1),
        obs.collision_rays.reshape(-1),
    ])
    mean = jnp.mean(obs_vec)
    std = jnp.std(obs_vec) + 1e-8
    obs_vec = (obs_vec - mean) / std
    return obs_vec


def compute_returns_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    returns = []
    advs = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advs.insert(0, gae)
        returns.insert(0, gae + values[t])
    return jnp.array(returns), jnp.array(advs)

def ppo_loss(policy_apply, policy_params, obs, actions, old_log_probs, advantages, clip_eps=0.2):
    mean, log_std = policy_apply(policy_params, obs)
    std = jnp.exp(log_std)
    log_probs = -0.5 * jnp.sum(((actions - mean)/std)**2 + 2*log_std + jnp.log(2*jnp.pi), axis=-1)
    ratio = jnp.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -jnp.mean(jnp.minimum(unclipped, clipped))
