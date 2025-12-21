import os
import pickle
import jax
import jax.numpy as jnp

from env.robotaxi import RoboTaxiEnv
from rl.ppo_model import build_policy, build_value
from rl.rollout import rollout
from rl.train import compute_returns_advantages, ppo_loss, flatten_obs

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    key = jax.random.PRNGKey(0)

    env = RoboTaxiEnv()
    env_params, init_state = env.init_params(
        key, map_id=0, max_steps=200, path_length=20
    )

    key, subkey = jax.random.split(key)
    obs, state = RoboTaxiEnv.reset(subkey, env_params, init_state)

    obs_vec = flatten_obs(obs)

    policy_model, policy_params = build_policy(obs_vec)
    value_model, value_params = build_value(obs_vec)

    for update in range(50):
        key, subkey = jax.random.split(key)
        # print(state.path_array)

        data = rollout(
            env_params=env_params,
            key=subkey,
            policy_apply=policy_model.apply,
            policy_params=policy_params,
            init_state=init_state,
            value_apply=value_model.apply,
            value_params=value_params,
            max_steps=300
        )

        returns, advs = compute_returns_advantages(
            data["rewards"], data["values"], data["dones"]
        )

        loss = ppo_loss(
            policy_model.apply,
            policy_params,
            data["obs"],
            data["actions"],
            data["log_probs"],
            advs,
        )

        print(f"Update {update}, loss = {loss}")

    checkpoint = {
        "policy_params": policy_params,
        "value_params": value_params,
    }

    with open(os.path.join(CHECKPOINT_DIR, "ppo_checkpoint.pkl"), "wb") as f:
        pickle.dump(checkpoint, f)

    print("Checkpoint saved.")


if __name__ == "__main__":
    main()
