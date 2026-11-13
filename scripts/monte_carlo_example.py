# %%
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from algorithms.monte_carlo import policy_iteration_monte_carlo

# %%



map = generate_random_map(size=4)
# %%
env = gym.make('FrozenLake-v1',  desc=map, render_mode=None, is_slippery=False)
iterated_policy = policy_iteration_monte_carlo(env, num_episodes=100, max_iterations=10,discount_factor=.9)
total_reward = 0
n_tests = 100
for i in range(n_tests):
    episode_over = False
    observation, info = env.reset()
    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        action = iterated_policy(observation, epsilon=0.1)  # Random action for now - real agents will be smarter!
        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        total_reward += reward
        episode_over = terminated or truncated

print(f"Average Reward {total_reward/n_tests}")
env.close()
# %%
env =gym.make('FrozenLake-v1', desc=map, render_mode="human", is_slippery=True)

total_reward = 0
for i in range(2):
    observation, info = env.reset()
    episode_over = False
    while not episode_over:
     
        action = iterated_policy(observation, epsilon=0.1) 
        print(action)
        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)


        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
env.close()
# %%
