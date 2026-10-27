import numpy as np


def global_random_policy(env):
    """A global random policy that selects actions uniformly at random. Assumes all actions are possible in all states.
    Expects enb to have an attribute 'action_set' which is a Literal of possible actions."""
    action_set = env.action_set

    def policy(state):
        return np.random.choice(action_set.__args__)

    return policy
