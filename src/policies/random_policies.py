import numpy as np


class RandomPolicy:
    def __init__(self, action_space):
        """
        Initializes a RandomPolicy instance.

        Args:
            action_space (gym.spaces.Discrete): The action space of the environment.

        """
        self.action_space = action_space
        self.rng = np.random.default_rng()  # Create a random number generator instance

    def __call__(self, state):
        return self.action_space.sample()  # Randomly choose among all actions
