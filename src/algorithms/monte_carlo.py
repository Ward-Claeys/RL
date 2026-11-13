# %%s
import numpy as np

from policies.general_functions import (
    create_sample_episode,
    get_action_state_returns_from_sample,
)
from policies.greedy_policies import GreedyPolicy


# %%
def estimate_action_values_monte_carlo(
    policy,
    env,
    num_episodes: int = 100,
    max_iterations: int = 1000,
    discount_factor: float = 1.0,
    first_visit: bool = True,
):
    returns = {}
    action_value_function = {}
    for i in range(num_episodes):
        # Generate the episode using the policy
        actions, states, rewards = create_sample_episode(policy, env, max_iterations)

        # Calculate returns and update value function
        new_returns = get_action_state_returns_from_sample(
            states, actions, rewards, discount_factor, first_visit
        )
        for state in new_returns:
            if state not in returns:
                returns[state] = {}
            for action in new_returns[state]:
                if action not in returns[state]:
                    returns[state][action] = []
                returns[state][action].extend(new_returns[state][action])

    action_value_function = {
        state: {
            action: np.mean(returns[state][action])
            if len(returns[state][action]) > 0
            else 0
            for action in returns[state]
        }
        for state in returns
    }

    return action_value_function


# %%


def policy_iteration_monte_carlo(
    env,
    discount_factor: float = 1.0,
    num_episodes: int = 100,
    max_iterations: int = 1000,
    first_visit: bool = True,
    start_policy=None,
    epsilon: float = 0.1,
):
    """Perform policy iteration to find the optimal policy."""
    # Initialize a starting policy
    if start_policy is not None:
        policy = start_policy
    else:
        policy = lambda state: env.action_space.sample()  # Start with a random policy
    last_action_value_function = None
    # Iterate until convergence
    for _ in range(max_iterations):
        # Evaluate the current policy
        action_value_function = estimate_action_values_monte_carlo(
            policy,
            env,
            num_episodes=num_episodes,
            discount_factor=discount_factor,
            first_visit=first_visit,
        )

        # Get the new policy based on the updated action-value function
        new_policy = GreedyPolicy(
            action_value_function, env.action_space, epsilon=epsilon
        )

        # If the policy hasn't changed, we've converged
        if last_action_value_function is not None:
            equal = True
            for key in action_value_function:
                if key not in last_action_value_function:
                    equal = False
                    break
                else:
                    for action in action_value_function[key]:
                        if action not in last_action_value_function[key]:
                            equal = False
                            break
                        elif not np.allclose(
                            action_value_function[key][action],
                            last_action_value_function[key][action],
                            atol=1e-5,
                        ):
                            equal = False
                            break
                        else:
                            continue

            if equal:
                break
        policy = new_policy
        last_action_value_function = action_value_function

    return new_policy
