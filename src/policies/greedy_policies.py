import numpy as np


def get_greedy_policy(action_value_function):
    """Get a greedy policy based on the action-value function."""
    greedy_policy_dict = {}
    for state, actions in action_value_function.items():
        action_values = np.array(list(actions.values()))
        act = np.array(list(actions.keys()))
        max_value = max(action_values)
        greedy_policy_dict[state] = act[action_values == max_value]

    rng = np.random.default_rng()  # Create a random number generator instance

    def greedy_policy(state):
        if state not in greedy_policy_dict:
            raise ValueError(f"State {state} not found in the action-value function.")
        if len(greedy_policy_dict[state]) > 1:
            return np.squeeze(
                rng.choice(greedy_policy_dict[state])
            ).item()  # Randomly choose among the best actions

        return np.squeeze(greedy_policy_dict[state]).item()

    return greedy_policy


def get_epsilon_greedy_policy(action_value_function, all_actions, epsilon: float = 0.1):
    """Get a greedy policy based on the action-value function."""
    greedy_policy_dict = {}
    for state, actions in action_value_function.items():
        action_values = np.array(list(actions.values()))
        act = np.array(list(actions.keys()))
        max_value = max(action_values)
        greedy_policy_dict[state] = act[action_values == max_value]

    adjusted_epsilon = (
        epsilon * (len(all_actions) / (len(all_actions) - 1))
    )  # Adjust epsilon based so we can just draw from all states and not have to worry about the greedy action being drawn.

    rng = np.random.default_rng()  # Create a random number generator instance

    def greedy_policy(state):
        if state not in greedy_policy_dict:
            return np.squeeze(
                rng.choice(all_actions)
            ).item()  # If the state is not in the action-value function, choose a random action
        if (
            rng.random() < adjusted_epsilon
        ):  # With probability epsilon, choose a random action
            return np.squeeze(rng.choice(all_actions)).item()

        if len(greedy_policy_dict[state]) > 1:
            return np.squeeze(
                np.random.choice(greedy_policy_dict[state])
            ).item()  # Randomly choose among the best actions
        return np.squeeze(greedy_policy_dict[state]).item()

    return greedy_policy
