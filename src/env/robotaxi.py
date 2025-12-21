import jax 
import jax.numpy as jnp
from typing import Tuple, Dict, Any
import chex
from functools import partial

from env.base import BaseEnv, BaseEnvParams, BaseEnvState
from env.base import RADIUS, NUM_RAY_SENSORS, FOV, OBSTACLE_COST, FPS


class RoboTaxiState(BaseEnvState):
    velocity:  chex.Scalar
    heading:  chex.Scalar

class RoboTaxiEnv(BaseEnv):
    @staticmethod
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5,
        num_ray_sensors: float = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: float = FPS,
    ) -> Tuple[BaseEnvParams, RoboTaxiState]:
        
        env_params, base_state = BaseEnv.init_params(
            key,
            map_id,
            max_steps,
            path_length,
            discretization_scale,
            perception_radius,
            num_ray_sensors,
            fov,
            fps,
        )
        state = RoboTaxiState(
            **base_state.__dict__,
            velocity = jnp.array(0.0), 
            heading = jnp.array(0.0),
        )   
        return env_params, state

    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey, 
        env_params: BaseEnvParams, 
        init_state: BaseEnvState | None
    ) -> Tuple[chex.Array, BaseEnvState]:
        if init_state is not None:
            state = init_state
        else: 
            raise Exception("init state is None")
        obs = BaseEnv.get_observation(state, env_params)
        return obs, state

    @partial(jax.jit,static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: BaseEnvState,
        action: chex.Array,  # (2,)
        env_params: BaseEnvParams,
    ) -> Tuple[
        chex.Array, RoboTaxiState, chex.Scalar | chex.Array, chex.Scalar | chex.Array, Dict[Any, Any]
    ]:
    
        acc = action[0]
        steer = action[1]

        dt = env_params.step_size
        friction = 0.1
        max_speed = 5.0

        velocity = env_state.velocity + acc * dt
        velocity *= (1.0 - friction)
        velocity = jnp.clip(velocity, -max_speed, max_speed) # ppo 

        heading = env_state.heading + steer * dt

        direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])
        new_agent_pos = env_state.agent_pos + direction * velocity * dt

        # from BaseEnv
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )

        # Check Done.
        goal_done = BaseEnv._check_goal(env_state.agent_pos, env_state.goal_pos)
        collision_done = BaseEnv._check_collisions(env_state.agent_pos, obstacles)
        time_done = env_state.time >= env_params.max_steps_in_episode

        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        # Calculate reward
        dist_prev = jnp.linalg.norm(env_state.agent_pos - env_state.goal_pos)
        dist_curr = jnp.linalg.norm(new_agent_pos - env_state.goal_pos)

        reward = (dist_prev - dist_curr) * dt
        reward = reward + goal_done.astype(jnp.float32)
        reward = reward - collision_done.astype(jnp.float32)

        # kinematic obstacles
        kinematic_obstacles = BaseEnv._move_kinematic_obstacles(env_state, env_params)

        # Increment time
        new_time = env_state.time + 1

        # path array. update every second.
        pred = jnp.mod(new_time, env_params.fps)
        path_array = jax.lax.cond(
            pred,
            lambda _: BaseEnv._find_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        state_dict = env_state.__dict__.copy()
        state_dict["time"] = new_time
        state_dict["agent_pos"] = new_agent_pos
        state_dict["agent_forward_dir"] = direction
        state_dict["velocity"] = velocity
        state_dict["heading"] = heading
        state_dict["kinematic_obstacles"] = kinematic_obstacles
        state_dict["path_array"] = path_array

        new_state = RoboTaxiState(**state_dict)

        obs = BaseEnv.get_observation(new_state, env_params)
        info = {
        "time": new_time,
        "velocity": velocity,
        "heading": heading,
        }
        return obs, new_state, reward, done, info