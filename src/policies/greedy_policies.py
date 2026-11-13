import numpy as np


class GreedyPolicy:
    def __init__(self, action_value_function, action_space, epsilon: float = 0.1):
        """
        Initializes a GreedyPolicy instance.

        Args:
            action_value_function (dict): A dictionary representing the action-value function.
            action_space (gym.spaces.Discrete): The action space of the environment.
            epsilon (float): The probability of choosing a random action (exploration).

        """
        self.action_value_function = action_value_function
        self.action_space = action_space
        self.epsilon = epsilon
        self.greedy_policy_dict = self._create_greedy_policy_dict(
            self.action_value_function
        )
        self.rng = np.random.default_rng()  # Create a random number generator instance
        self.n = action_space.n

    def _create_greedy_policy_dict(self, action_value_function):
        greedy_policy_dict = {}
        for state, actions in action_value_function.items():
            action_values = np.array(list(actions.values()))
            act = np.array(list(actions.keys()))
            max_value = max(action_values)
            greedy_policy_dict[state] = act[action_values == max_value]
        return greedy_policy_dict

    def update_policy(self, new_action_value_function):
        """
        Updates the action-value function and the greedy policy dictionary.
        This is useful for policy iteration so you don't have to create a new instance of the GreedyPolicy class every time you want to update the policy.

        Args:
            new_action_value_function (dict): A new action-value function to update the policy.

        """
        self.action_value_function = new_action_value_function
        self.greedy_policy_dict = self._create_greedy_policy_dict(
            self.action_value_function
        )
    
    def __call__(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        adjusted_epsilon = (
            epsilon * (self.n / (self.n - 1))
        )  # Adjust epsilon based so we can just draw from all states and not have to worry about the greedy action being drawn.

        if state not in self.greedy_policy_dict:
            return self.action_space.sample()  # If the state is not in the action-value function, choose a random action
        choose_greedy = (
            self.rng.random() > adjusted_epsilon
        )  # With probability epsilon, choose a random action
        if choose_greedy:
            if len(self.greedy_policy_dict[state]) > 1:
                return np.squeeze(
                    self.rng.choice(self.greedy_policy_dict[state])
                ).item()  # Randomly choose among the best actions

            return np.squeeze(self.greedy_policy_dict[state]).item()
        else:
            return self.action_space.sample()  # Randomly choose among all actions

    def __equals__(self, other):
        if not isinstance(other, GreedyPolicy):
            return False
        return (
            self.greedy_policy_dict == other.greedy_policy_dict
            and self.action_space == other.action_space
            and self.epsilon == other.epsilon
        )
