"""
Temporal Difference Learning Example. Also includes Q-learning and policy iteration using temporal difference learning.
"""

# %%

from copy import copy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from policies.greedy_policies import GreedyPolicy
from policies.random_policies import RandomPolicy


def td_value_estimation(env, policy, num_episodes=100, discount_factor=1.0, alpha=0.1):
    value_function = {key: 0 for key in range(env.observation_space.n)}
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = policy(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update the value function using TD(0) update rule
            value_function[observation] += alpha * (
                reward
                + discount_factor * value_function[next_observation]
                - value_function[observation]
            )
            if done:
                break
            observation = next_observation
    return value_function


def td_action_value_function_estimation(
    env, policy, num_episodes=100, discount_factor=1.0, alpha=0.1, q_learning = False
):
    action_value_function = {
        state: {action: 0 for action in range(env.action_space.n)}
        for state in range(env.observation_space.n)
    }
    for episode in range(num_episodes):
        done = False
        observation, _ = env.reset()
        action = policy(observation)
        while not done:
            next_observation, reward, terminated, truncated, _ = env.step(action)

            next_action = policy(next_observation)
            done = terminated or truncated

            # Update the action-value function using TD(0) update rule
            if q_learning:
                update_term = max(action_value_function[next_observation].values())
            else:    
                update_term = action_value_function[next_observation][next_action]
            action_value_function[observation][action] += alpha * (
                reward
                + discount_factor * update_term
                - action_value_function[observation][action]
            )
            if done:
                break
            action = next_action
            observation = next_observation
    return action_value_function


def td_pseudo_policy_iteration(
    env, num_episodes=100, max_iterations=100, discount_factor=1.0, alpha=0.1
):
    policy = RandomPolicy(env.action_space)

    for iteration in range(max_iterations):
        action_value_function = td_action_value_function_estimation(
            env, policy, num_episodes, discount_factor, alpha
        )
        new_policy = GreedyPolicy(action_value_function, env.action_space)
        if new_policy == policy:
            break
        else:
            policy = new_policy
    return new_policy

def reset_action_value_function(action_value_function):
    for state in action_value_function:
        for action in action_value_function[state]:  
            action_value_function[state][action] = 0
    return action_value_function

def initialize_action_value_function(env):
    action_value_function = {
        state: {action: 0 for action in range(env.action_space.n)}
        for state in range(env.observation_space.n)
    }
    return action_value_function

def td_policy_iteration(
    env, initial_policy=None, num_episodes=100, discount_factor=1.0, alpha=0.1,
    policy_update_steps = None, q_learning=False, cache_policies=False
):
    """
    Performs policy iteration using temporal difference learning.

    Args:
        env (gym.Env): The environment to interact with.
        initial_policy (GreedyPolicy): The initial policy to use. Is expected to have an update method that accepts the action-value function and updates the policy accordingly.
        num_episodes (int): The number of episodes to run.
        discount_factor (float): The discount factor for future rewards.
        alpha (float): The learning rate.
        policy_update_steps (tuple): A tuple of two integers. The first integer specifies the number of steps after which to update the policy within an episode. The second integer specifies the number of episodes after which to update the policy. Default is (0, 10), which means the policy will be updated every 10 episodes and not within an episode. 

    Returns:
        GreedyPolicy: The final policy.
    """
    if policy_update_steps is None:
        policy_update_steps = (0, 10)
    if policy_update_steps[0] < 0 or policy_update_steps[1] < 0:
        raise ValueError("Policy update steps must be non-negative integers.")
    if 0 < policy_update_steps[0] < 100:
        print("Warning: Policy update steps within an episode is set to a low value. This may lead to unstable learning.")
    # Start with a random policy

    action_value_function = initialize_action_value_function(env)
    if initial_policy is None:
        policy = GreedyPolicy(action_value_function, env.action_space)
    else:
        policy = copy(initial_policy)
    if cache_policies:
        policy_cache = [ copy(policy) ]
        iteration_cache = [(0, 0)]
    step_number = 0
    for i in range(num_episodes):
        done = False
        observation, _ = env.reset()
        action = policy(observation)
       
        while not done:
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # if reward == 0 and done:
            #     reward = -100
            next_action = policy(next_observation)
            if q_learning:
                update_term = max(action_value_function[next_observation].values())
            else:    
                update_term = action_value_function[next_observation][next_action]
            # Update the action-value function using TD(0) update rule
            action_value_function[observation][action] += alpha * (
                reward
                + discount_factor * update_term
                - action_value_function[observation][action]
            )
            if done:
                break
            action = next_action
            observation = next_observation

            # Update the policy to be greedy with respect to the updated action-value function
            step_number += 1
            if policy_update_steps[0] > 0 and step_number % policy_update_steps[0] == 0:
                policy.update_policy(action_value_function)
                ##action_value_function = initialize_action_value_function(env)
                #action_value_function = reset_action_value_function(action_value_function)
        if policy_update_steps[1] > 0 and (i + 1) % policy_update_steps[1] == 0:
            policy.update_policy(action_value_function)
            #action_value_function = initialize_action_value_function(env)
            if cache_policies:
                policy_cache.append(copy(policy))
                iteration_cache.append((i, step_number))
            #action_value_function = reset_action_value_function(action_value_function)
        # Update the policy to be greedy with respect to the updated action-value function
    if cache_policies:
        return policy, policy_cache, iteration_cache
    return policy



# %%
n = 4
map = generate_random_map(size=n)
is_slippery = True
# %%
env = gym.make("FrozenLake-v1", desc=map, render_mode=None, is_slippery=is_slippery)
initial_policy = GreedyPolicy({}, env.action_space, epsilon=0.01)

# random_policy = RandomPolicy(env.action_space)
# value_function = td_value_estimation(env, random_policy, num_episodes=1000)
# action_value_function = td_action_value_function_estimation(env, random_policy, num_episodes=10000)
# iterated_policy = td_pseudo_policy_iteration(env, num_episodes=1000, max_iterations=100, discount_factor=.9)
iterated_policy, policies, iterations = td_policy_iteration(env, initial_policy=initial_policy, 
                                      num_episodes=1000, discount_factor=1.0,
                                      policy_update_steps=(0, 10), alpha=0.1,
                                      q_learning=False, cache_policies=True)
# %%
def evaluate_policy(env, policy, n_tests=100):
    total_reward = 0
    n_tests = 100
    for i in range(n_tests):
        episode_over = False
        observation, info = env.reset()
        while not episode_over:
        
            action = iterated_policy(
                observation
            )  
            observation, reward, terminated, truncated, _ = env.step(action)


            total_reward += reward
            episode_over = terminated or truncated
    return total_reward / n_tests

average_rewards = {}
for i, policy in zip(iterations, policies):

    average_rewards[i] = evaluate_policy(env, policy, n_tests=10)

# %%
pd.Series(average_rewards).plot()
# %%
values = td_value_estimation(env, iterated_policy, num_episodes=1000)
plt.imshow(
    np.array(list(values.values())).reshape(n, n),
)
for row_index, map_row in enumerate(map):
    for column_index, cell in enumerate(map_row):
        if cell in ("S", "H", "G"):
            plt.text(
                column_index,
                row_index,
                cell,
                ha="center",
                va="center",
                color=["black", "red", "green"][["S", "H", "G"].index(cell)],
                fontsize=14,
                fontweight="bold",
            )

plt.colorbar()
plt.show()
env.close()
# %%
env = gym.make("FrozenLake-v1", desc=map, render_mode="human", is_slippery=is_slippery)
for i in range(2):
    episode_over = False
    total_reward = 0
    observation, info = env.reset()
    while not episode_over:
        action = iterated_policy(
            observation,
        )  
        print(action)
        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)


        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
env.close()
# %%
