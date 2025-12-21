import jax
import jax.numpy as jnp
from env.robotaxi import RoboTaxiEnv
from rl.train import flatten_obs

def rollout(env_params, key, policy_apply, policy_params, value_apply, value_params, init_state, max_steps: int = 200):
    obs_list, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    obs, state = RoboTaxiEnv.reset(key, env_params, init_state)
    obs_vec = flatten_obs(obs)

    for t in range(max_steps):
        # --- Policy step ---
        mean, log_std = policy_apply(policy_params, obs_vec)
        std = jnp.exp(log_std)
        key, subkey = jax.random.split(key)
        action = mean + std * jax.random.normal(subkey, shape=mean.shape)
        log_prob = -0.5 * jnp.sum(((action - mean)/std)**2 + 2*log_std + jnp.log(2*jnp.pi))

        # --- Value step ---
        value = value_apply(value_params, obs_vec).squeeze()  # scalar

        # --- Save step ---
        obs_list.append(obs_vec)
        actions.append(action)
        rewards.append(0.0) 
        dones.append(False)
        log_probs.append(log_prob)
        values.append(value)

        # --- Environment step ---
        obs, state, reward, done, info = RoboTaxiEnv.step(key, state, action, env_params)
        obs_vec = flatten_obs(obs)

        rewards[-1] = reward
        dones[-1] = done

        if done:
            obs, state = RoboTaxiEnv.reset(key, env_params, init_state)
            obs_vec = flatten_obs(obs)

    return {
        "obs": jnp.stack(obs_list),
        "actions": jnp.stack(actions),
        "rewards": jnp.array(rewards),
        "log_probs": jnp.stack(log_probs),
        "values": jnp.array(values),
        "dones": jnp.array(dones)
    }
