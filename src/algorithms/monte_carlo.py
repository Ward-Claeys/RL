# %%
import matplotlib.pyplot as plt
import numpy as np


def create_sample_episode(policy, env, max_iterations: int = 1000):
    env.reset()
    actions, states, rewards = [], [], []
    for i in range(max_iterations):
        state = env.state
        action = policy(state)
        next_state, reward = env.perform(action)

        actions.append(action)
        states.append(state)
        rewards.append(reward)

        if next_state in env.goal_states:
            break

    return actions, states, rewards


def estimate_values(
    policy,
    env,
    num_episodes: int = 100,
    max_iterations: int = 1000,
    discount_factor: float = 1.0,
    first_visit: bool = True,
):

    state = env.reset()
    if not hasattr(env, "goal_states"):
        raise AttributeError(
            "The environment must have a 'goal_state' attribute for this function to work."
        )
    if hasattr(env, "all_states"):
        returns = {tuple(state): [] for state in env.all_states}
    else:
        print(
            "Warning: The environment does not have an 'all_states' attribute. Will iteratively build the states as encountered."
        )
        returns = {tuple(state): []}

    for i in range(num_episodes):
        # Generate the episode using the policy
        _, states, rewards = create_sample_episode(policy, env, max_iterations)

        # Calculate returns and update value function
        new_returns = get_state_returns_from_sample(
            states, rewards, discount_factor, first_visit
        )
        for state in new_returns:
            if state not in returns:
                returns[state] = []
            returns[state].extend(new_returns[state])
    value_function = {
        state: np.mean(returns[state]) if len(returns[state]) > 0 else 0
        for state in returns
    }

    return value_function


def get_state_returns_from_sample(
    states, rewards, discount_factor: float = 1.0, first_visit: bool = True
):
    G = 0
    returns = {}
    for t in range(len(states) - 1, -1, -1):
        G = discount_factor * G + rewards[t]
        if tuple(states[t]) not in returns:
            returns[tuple(states[t])] = []
        if first_visit:
            if tuple(states[t]) not in [tuple(s) for s in states[:t]]:
                returns[tuple(states[t])].append(G)
        else:
            returns[tuple(states[t])].append(G)
    return returns


def get_action_state_returns_from_sample(
    states, actions, rewards, discount_factor: float = 1.0, first_visit: bool = True
):
    G = 0
    returns = {}
    for t in range(len(states) - 1, -1, -1):
        G = discount_factor * G + rewards[t]

        if tuple(states[t]) not in returns:
            returns[tuple(states[t])] = {actions[t]: []}
        elif actions[t] not in returns[tuple(states[t])]:
            returns[tuple(states[t])][actions[t]] = []
        if first_visit:
            if tuple(states[t]) not in [tuple(s) for s in states[:t]]:
                returns[tuple(states[t])][actions[t]].append(G)
        else:
            returns[tuple(states[t])][actions[t]].append(G)
    return returns


def estimate_action_values(
    policy,
    env,
    num_episodes: int = 100,
    max_iterations: int = 1000,
    discount_factor: float = 1.0,
    first_visit: bool = True,
):
    if not hasattr(env, "goal_states"):
        raise AttributeError(
            "The environment must have a 'goal_state' attribute for this function to work."
        )
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


def policy_iteration(
    env,
    iterate_policy,
    discount_factor: float = 1.0,
    num_episodes: int = 100,
    max_iterations: int = 1000,
    first_visit: bool = True,
):
    """Perform policy iteration to find the optimal policy."""
    # Initialize a random policy
    policy = global_random_policy(env)

    # Iterate until convergence
    for _ in range(max_iterations):
        # Evaluate the current policy
        action_value_function = estimate_action_values(
            policy,
            env,
            num_episodes=num_episodes,
            discount_factor=discount_factor,
            first_visit=first_visit,
        )

        # Get the new policy based on the updated action-value function
        new_policy = iterate_policy(action_value_function)

        # If the policy hasn't changed, we've converged
        if new_policy == policy:
            break

        policy = new_policy

    return policy


def plot_value_heatmap(value_function, env):
    """Plot a heat map for a value function indexed by (row, column) states."""
    if not value_function:
        raise ValueError("Cannot plot an empty value function.")

    states = list(value_function)
    rows = [state[0] for state in states]
    columns = [state[1] for state in states]
    heatmap = np.full((max(rows) + 1, max(columns) + 1), np.nan)

    for state, value in value_function.items():
        heatmap[state[0], state[1]] = value

    fig, ax = plt.subplots()
    image = ax.imshow(heatmap, cmap="viridis", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Value")
    ax.set_title("Value Function")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(heatmap.shape[1]))
    ax.set_yticks(range(heatmap.shape[0]))

    goal_states = getattr(env, "goal_states", [(None, None)])
    start_state = getattr(env, "start_state", (None, None))
    illegal_states = getattr(env, "illegal_states", [])
    fontsize = 11
    for (row, column), value in np.ndenumerate(heatmap):
        if (row, column) in illegal_states:
            continue  # Skip illegal states
        elif (row, column) in goal_states:
            ax.add_patch(
                plt.Rectangle((column - 0.5, row - 0.5), 1, 1, color="green", alpha=1)
            )
            ax.text(
                column,
                row,
                "G",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )
        elif row == start_state[0] and column == start_state[1]:
            ax.add_patch(
                plt.Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    edgecolor="white",
                    alpha=1,
                    facecolor="none",
                )
            )
            ax.text(
                column,
                row,
                f"Start\n{value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=fontsize,
            )
        elif not np.isnan(value):
            ax.text(
                column, row, f"{value:.2f}", ha="center", va="center", color="white"
            )

    for illegal_state in illegal_states:
        ax.add_patch(
            plt.Rectangle(
                (illegal_state[1] - 0.5, illegal_state[0] - 0.5),
                1,
                1,
                color="red",
                alpha=1,
            )
        )
        ax.text(
            illegal_state[1],
            illegal_state[0],
            "X",
            ha="center",
            va="center",
            color="white",
            fontsize=fontsize,
        )

    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    from environments.gridworldes import GridWorldEnvironment
    from policies.greedy_policies import get_epsilon_greedy_policy, get_greedy_policy
    from policies.random_policies import global_random_policy

    # Create an instance of the GridWorldEnvironment
    config = {
        "grid_size": 5,
        "goal_states": [(4, 4)],
        "start_state": (0, 0),
        "illegal_states": [(2, 2), (2, 3), (2, 4)],
    }  # Example configuration  "illegal_states":
    env = GridWorldEnvironment(config)

    random_policy = global_random_policy(env)

    # Estimate values using the simple policy
    value_function = estimate_values(
        random_policy,
        env,
        num_episodes=100,
        max_iterations=1000,
        discount_factor=0.9,
        first_visit=False,
    )

    action_value_function = estimate_action_values(
        random_policy,
        env,
        num_episodes=100,
        max_iterations=1000,
        discount_factor=0.9,
        first_visit=False,
    )
    # Print the estimated value function

    fig, ax = plot_value_heatmap(value_function, env)
    ax.set_title("Value Function Heatmap Random Policy")
    # Get the greedy policy based on the action-value function
    greedy_policy = get_greedy_policy(action_value_function)
    value_function = estimate_values(
        greedy_policy,
        env,
        num_episodes=10,
        max_iterations=100,
        discount_factor=0.9,
        first_visit=False,
    )
    fig, ax = plot_value_heatmap(value_function, env)
    ax.set_title("Value Function Heatmap Greedy Policy")
    # %%
    iterated_policy = policy_iteration(
        env,
        get_greedy_policy,
        discount_factor=0.9,
        max_iterations=100,
        first_visit=False,
    )

    value_function = estimate_values(
        iterated_policy,
        env,
        num_episodes=10,
        max_iterations=100,
        discount_factor=0.9,
        first_visit=False,
    )
    fig, ax = plot_value_heatmap(value_function, env)
    ax.set_title("Value Function Heatmap Policy Iteration")

    # %%
    epsilon_greedy_policy = lambda x: get_epsilon_greedy_policy(
        x, env.action_set.__args__, epsilon=0.05
    )
    iterated_policy = policy_iteration(
        env,
        epsilon_greedy_policy,
        num_episodes=100,
        discount_factor=0.9,
        max_iterations=100,
        first_visit=False,
    )

    value_function = estimate_values(
        iterated_policy,
        env,
        num_episodes=100,
        max_iterations=100,
        discount_factor=0.9,
        first_visit=False,
    )
    fig, ax = plot_value_heatmap(value_function, env)
    ax.set_title("Value Function Heatmap Policy Iteration")
    # %%
