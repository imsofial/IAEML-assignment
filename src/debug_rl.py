import jax
import jax.numpy as jnp

from env.robotaxi import RoboTaxiEnv
from utils.autoreset import AutoResetWrapper

def make_env(key):
    env_params, init_state = RoboTaxiEnv.init_params(
        key=key,
        map_id=1,
        max_steps=300,
        path_length=100,
        discretization_scale=1,
        fps=30,
    )
    env = AutoResetWrapper(RoboTaxiEnv, env_params, init_state)
    return env, env_params

def test_reset(env, env_params, key):
    obs, state = env.reset(key, env_params, None)

    print("RESET CHECK")
    print("obs type:", type(obs))
    print("obs fields:", obs.__dict__.keys())
    print("state type:", type(state))
    print("state.time:", state.time)
    print("-" * 30)

    return obs, state

def test_single_step(env, env_params, obs, state, key):
    key, subkey = jax.random.split(key)

    action = jnp.array([0.5, 0.1])  # acc, steer

    obs2, state2, reward, done, info = env.step(
        subkey, state, action, env_params
    )

    print("STEP CHECK")
    print("action:", action)
    print("obs fields:", obs2.__dict__.keys())
    print("reward:", reward)
    print("done:", done)
    print("info:", info)
    print("time:", state2.time)
    print("-" * 30)

    return obs2, state2

def test_rollout(env, env_params, key, steps=200):
    obs, state = env.reset(key, env_params, None)

    for t in range(steps):
        key, subkey = jax.random.split(key)

        action = jax.random.uniform(
            subkey, (2,), minval=-1.0, maxval=1.0
        )

        obs, state, reward, done, info = env.step(
            subkey, state, action, env_params
        )

        if t % 50 == 0:
            print(
                f"t={t} | reward={reward:.3f} | done={bool(done)} | vel={info['velocity']:.2f}"
            )

    print("ROLLOUT FINISHED")

def main():
    key = jax.random.PRNGKey(0)

    env, env_params = make_env(key)

    obs, state = test_reset(env, env_params, key)
    obs, state = test_single_step(env, env_params, obs, state, key)
    test_rollout(env, env_params, key)


if __name__ == "__main__":
    main()
