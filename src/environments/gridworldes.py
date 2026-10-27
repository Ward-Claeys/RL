from typing import Literal


class BaseEnvironment:
    """Base class for environments. This class should be inherited by specific environment implementations.
    It provides a common interface for interacting with the environment, including methods for resetting the environment,
    taking steps, and calculating rewards. Subclasses should implement the specific logic for these methods based on the environment's dynamics and rules."""

    def __init__(self, config):
        self.config = config

    def reset(self):
        """Reset the environment to its initial state"""
        raise NotImplementedError("The reset method must be implemented by subclasses.")

    def reward(self, state, action, next_state):
        """Calculate the reward for transitioning from the current state to the next state after taking an action."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def step(self, action):
        """Take a step in the environment based on the given action and return the next state. Does not update the environment's state."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def perform(self, action):
        """Perform the given action in the environment. Updates the environment's state and returns the next state and the reward received."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class GridWorldEnvironment(BaseEnvironment):
    """A simple grid world environment where an agent can move in four directions (up, down, left, right) on a grid.
    The environment has a defined start state, goal states, and illegal states that the agent cannot enter.
    The agent receives rewards based on its actions and the states it reaches.

    Attributes:
        grid_size (int): The size of the grid (grid_size x grid_size). Default is 5 if not specified in the config.
        start_state (tuple): The starting position of the agent in the grid. Default is (0, 0) if not specified in the config.
        goal_states (list): A list of goal states that provide a reward when reached. Default is the bottom-right corner of the grid if not specified in the config.
        illegal_states (list): A list of states that the agent cannot enter. Default is an empty list if not specified in the config.
        step_penalty (float): The penalty for each step taken by the agent. Default is 0 if not specified in the config.
        goal_reward (float): The reward for reaching a goal state.  Default is 1 if not specified in the config.
        action_set (Literal): A literal type representing the set of possible actions.  ["up", "down", "left", "right"].

    """

    def __init__(self, config):
        """Initialize the GridWorldEnvironment with the provided configuration.
        Attributes can be set through the config dictionary.
        Args:
            config (dict): A dictionary containing configuration parameters for the environment.


        """
        super().__init__(config)
        self.grid_size = config.get("grid_size", 5)
        self.start_state = config.get("start_state", (0, 0))  # Starting position
        self.goal_states = config.get(
            "goal_states", [(self.grid_size - 1, self.grid_size - 1)]
        )  # If no goal state is provided, default to the bottom-right corner
        self.illegal_states = config.get(
            "illegal_states", []
        )  # Defaults to no illegal states if not provided in the config

        self.step_penalty = config.get("step_penalty", 0)  # Default step penalty is 0
        self.goal_reward = config.get(
            "goal_reward", 1
        )  # Default reward for reaching the goal is 1
        self.action_set = Literal["up", "down", "left", "right"]  # Possible actions

        # Set the intial state to the starting position
        self.reset()

        self.all_states = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.illegal_states
        ]

    def reward(self, next_state):
        if next_state in self.goal_states:
            return self.goal_reward  # Reward for reaching the goal
        else:
            return (
                self.step_penalty
            )  # Small negative reward for each step to encourage faster solutions

    def step(self, action: str):

        if action not in self.action_set.__args__:
            raise ValueError(
                f"Invalid action: {action}. Valid actions are: {self.action_set.__args__}"
            )

        if action == "up":
            next_state = self._do_up()
        elif action == "down":
            next_state = self._do_down()
        elif action == "left":
            next_state = self._do_left()
        elif action == "right":
            next_state = self._do_right()

        # Stay in the same state if the next state is illegal
        if next_state in self.illegal_states:
            next_state = self.state
        return next_state

    def _do_up(self):
        next_state = (min(self.grid_size - 1, self.state[0] + 1), self.state[1])
        return next_state

    def _do_down(self):
        next_state = (max(0, self.state[0] - 1), self.state[1])
        return next_state

    def _do_left(self):
        next_state = (self.state[0], max(0, self.state[1] - 1))
        return next_state

    def _do_right(self):
        next_state = (self.state[0], min(self.grid_size - 1, self.state[1] + 1))
        return next_state

    def perform(self, action: str):
        next_state = self.step(action)
        reward = self.reward(next_state)
        self.state = next_state
        return next_state, reward

    def reset(self):
        self.state = self.start_state  # Reset to starting position
        return self.state
