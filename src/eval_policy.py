import pickle
import jax
import jax.numpy as jnp

from env.robotaxi import RoboTaxiEnv
from utils.autoreset import AutoResetWrapper
from rl.train import flatten_obs

def mlp_forward(params, x):
    for (W, b) in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def policy_value(policy_params, value_params, obs_vec):
    mean = mlp_forward(policy_params, obs_vec)
    value = mlp_forward(value_params, obs_vec)[..., 0]
    return mean, value


def load_ckpt(path="checkpoints/ppo_checkpoint.pkl"):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["policy_params"], ckpt["value_params"]


def run_eval(num_episodes=20, max_steps=600, seed=0):
    policy_params, value_params = load_ckpt()

    key = jax.random.PRNGKey(seed)

    env = RoboTaxiEnv()
    env_params, init_state = env.init_params(
        key=key,
        map_id=0,
        max_steps=max_steps,
        path_length=20,
    )

    wrapped_env = AutoResetWrapper(env, env_params, init_state)

    success = 0
    collision = 0
    timeouts = 0
    returns = []
    lengths = []

    for ep in range(num_episodes):
        key, k0 = jax.random.split(key)
        obs, st = RoboTaxiEnv.reset(k0,env_params, init_state)

        ep_ret = 0.0
        ep_len = 0
        last_info = None

        for t in range(max_steps):
            obs_vec = flatten_obs(obs)

            mean, _ = policy_value(policy_params, value_params, obs_vec)

            act = jnp.clip(mean, -1.0, 1.0)

            key, ks = jax.random.split(key)
            obs, st, r, done, info = wrapped_env.step(ks, st, act)

            ep_ret += float(r)
            ep_len += 1
            last_info = info

            if bool(done):
                break

        returns.append(ep_ret)
        lengths.append(ep_len)

        goal = False
        col = False

        if isinstance(last_info, dict):
            goal = bool(last_info.get("goal", False))
            col = bool(last_info.get("collision", False))

        if goal:
            success += 1
        elif col:
            collision += 1
        else:
            timeouts += 1

        print(
            f"ep={ep:02d} len={ep_len:4d} return={ep_ret:8.3f} "
            f"goal={goal} collision={col}"
        )

    print("\n==== SUMMARY ====")
    print(f"episodes: {num_episodes}")
    print(f"success:   {success} ({100*success/num_episodes:.1f}%)")
    print(f"collision: {collision} ({100*collision/num_episodes:.1f}%)")
    print(f"timeout:   {timeouts} ({100*timeouts/num_episodes:.1f}%)")
    print(f"avg_len:   {sum(lengths)/len(lengths):.1f}")
    print(f"avg_ret:   {sum(returns)/len(returns):.3f}")


if __name__ == "__main__":
    run_eval(num_episodes=20, max_steps=600, seed=0)
