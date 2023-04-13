import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments.environment import Environment
from gymnax.visualize import Visualizer
from flax.core.frozen_dict import FrozenDict
from model import NN

# @partial(jax.jit, static_argnums=(0, 3, 4))
def generate_data(env: Environment,
                key: jax.random.PRNGKey,
                model_params: FrozenDict, 
                model: NN,
                n_actions: int,
                discount: float
                ):
    
    state_seq, reward_seq = [], []
    key, subkey_reset = jax.random.split(key)
    obs, env_state = env.reset(subkey_reset)
    while True:
        state_seq.append(env_state)
        key, subkey_act, subkey_step = jax.random.split(key, 3)

        policy_log_probs, _ = model.apply(model_params, obs)
        policy_probs = jnp.exp(policy_log_probs)
        assert policy_probs.shape == (n_actions,), f"{policy_probs.shape}, {n_actions}"
        action = jax.random.choice(subkey_act, n_actions, p=policy_probs)

        next_obs, next_env_state, reward, done, _ = env.step(
            subkey_step, env_state, action)
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    return state_seq, reward_seq
    
    # key, subkey_reset = jax.random.split(key)
    # state_feature, state = env.reset(subkey_reset)

    # initial_val = {"next_is_terminal": False,
    #                't': 0,
    #                "discounted_return": 0,
    #                "key": key,
    #                "model_params": model_params,
    #                "state_feature": state_feature,
    #                "state": state}
    
    # state_seq = []
    # reward_seq = []
    
    # def condition_function(val):
    #     return jnp.logical_not(val["next_is_terminal"])

    # def body_function(val):
    #     val["key"], subkey_policy, subkey_mdp = jax.random.split(val["key"], 3)

    #     # (n_actions), (1,)
    #     policy_log_probs, _ = model.apply(val["model_params"], val["state_feature"])
    #     policy_probs = jnp.exp(policy_log_probs)
    #     assert policy_probs.shape == (n_actions,), f"{policy_probs.shape}, {n_actions}"
    #     action = jax.random.choice(subkey_policy, n_actions, p=policy_probs)
        
    #     val["state_feature"], val["state"], reward, val["next_is_terminal"], _ = env.step(subkey_mdp, val["state"], action)
    #     val["discounted_return"] += (discount**val['t']) * reward
    #     val['t'] += 1
    #     state_seq.append(val["state"])
    #     reward_seq .append(reward)
    #     return val

    # val = jax.lax.while_loop(condition_function, body_function, initial_val)
    
    # return val, state_seq, reward_seq
    

def create_vis(env: Environment,
                env_params,
                key: jax.random.PRNGKey,
                model_params: FrozenDict, 
                model: NN,
                n_actions: int,
                discount: float
                ):
    
    state_seq, reward_seq = generate_data(env,
                key,
                model_params, 
                model,
                n_actions,
                discount
                )
    
    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    return vis