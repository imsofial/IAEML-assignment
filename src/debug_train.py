import jax
import jax.numpy as jnp

from env.robotaxi import RoboTaxiEnv
from rl.ppo_model import build_policy, build_value
from rl.train import flatten_obs

def main():
    key = jax.random.PRNGKey(0)

    env_params, init_state = RoboTaxiEnv.init_params(
        key, map_id=0, max_steps=100, path_length=10
    )

    key, subkey = jax.random.split(key)
    obs, state = RoboTaxiEnv.reset(subkey, env_params, init_state)

    obs_vec = flatten_obs(obs)

    policy_model, policy_params = build_policy(obs_vec)
    value_model, value_params = build_value(obs_vec)

    mean, log_std = policy_model.apply(policy_params, obs_vec)
    value = value_model.apply(value_params, obs_vec)

    print("Policy mean:", mean)
    print("Value:", value)


if __name__ == "__main__":
    main()
