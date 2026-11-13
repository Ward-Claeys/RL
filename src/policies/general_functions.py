def create_sample_episode(policy, env, max_iterations: int = 1000):
    observation, _ = env.reset()
    actions, states, rewards = [], [], []
    for i in range(max_iterations):
        states.append(observation)
        action = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

        actions.append(action)

        rewards.append(reward)

        if terminated or truncated:
            break

    return actions, states, rewards


def get_state_returns_from_sample(
    states, rewards, discount_factor: float = 1.0, first_visit: bool = True
):
    G = 0
    returns = {}
    for t in range(len(states) - 1, -1, -1):
        G = discount_factor * G + rewards[t]
        if states[t] not in returns:
            returns[states[t]] = []
        if first_visit:
            if states[t] not in [s for s in states[:t]]:
                returns[states[t]].append(G)
        else:
            returns[states[t]].append(G)
    return returns


def get_action_state_returns_from_sample(
    states, actions, rewards, discount_factor: float = 1.0, first_visit: bool = True
):
    G = 0
    returns = {}
    for t in range(len(states) - 1, -1, -1):
        G = discount_factor * G + rewards[t]

        if states[t] not in returns:
            returns[states[t]] = {actions[t]: []}
        elif actions[t] not in returns[states[t]]:
            returns[states[t]][actions[t]] = []
        if first_visit:
            if (states[t], actions[t]) not in [
                (s, a) for s, a in zip(states[:t], actions[:t])
            ]:
                returns[states[t]][actions[t]].append(G)
        else:
            returns[states[t]][actions[t]].append(G)
    return returns
