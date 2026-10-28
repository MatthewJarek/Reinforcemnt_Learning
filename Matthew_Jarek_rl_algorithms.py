# --- Your code here ---#
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Implement the policy (\epsilon-greedy)
def epsilon_greedy(epsilon: float, Q: np.ndarray[float], S: int):
    probability = np.random.uniform(size = 1)

    if probability < epsilon:
        return int(np.random.choice(np.linspace(0, len(Q[S])-1, len(Q[S]))))
    else:
        # Get argmax_a(Q(S,a))
        return argmax(Q[S])

# Make my own argmax that breaks ties randomly
def argmax(a):
    if len(np.argwhere(a == np.max(a)). flatten()) > 1:
        # If there is a tie, choose a random argmax
        action = int(np.random.choice(np.argwhere(a == np.max(a)).flatten()))
    # If no ties then take the argmax
    else: action = int(np.argmax(a))
    return action

def sarsa(env: gym.Env, alpha=0.5, gamma=1.0, epsilon=0.1, num_episodes=500) -> tuple[np.ndarray[float], np.ndarray[float]]:
    # Implement SARSA.

    # Initialize Q(s,a) for all s \in S^+, a \in A(s) arbitrarily except that Q(terminal,\cdot)=0
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    Q = np.zeros((num_states,num_actions), float)
    total_rewards = np.zeros(0)

    # Loop for each episode
    for episode in range(num_episodes):
        total_reward = 0

        # Initialize S
        S, info = env.reset()

        # Choose A from S using policy derived from Q (e.g., \epsilon-greedy)
        A = epsilon_greedy(epsilon, Q, S)
        terminated, truncated = (0,0)

        # Loop for each step of episode
        while not (terminated or truncated):

            # Take action A, observe R, S'
            Sp, reward, terminated, truncated, info = env.step(A)

            # Choose action A' from S' using policy derived from Q (e.g., \epsilon-greedy)
            Ap = epsilon_greedy(epsilon, Q, Sp)

            if not (terminated or truncated):
                # Q(S,A) <- Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]
                Q[S, A] = Q[S, A] + alpha*(reward + gamma*(Q[Sp, Ap])- Q[S, A])
            else:
                Q[S, A] = Q[S, A] + alpha*(reward - Q[S, A])
            total_reward += reward

            # S <- S'; A <- A' 
            S = Sp
            A = Ap

        total_rewards = np.append(total_rewards, total_reward)


    # Return the final Q-table and episode rewards.
    return Q, total_rewards

def q_learning(env: gym.Env, alpha=0.5, gamma=1.0, epsilon=0.1, num_episodes=500) -> tuple[np.ndarray[float], np.ndarray[float]]:
    # Implement Q-learning.

    # Initialize Q(s,a) for all s \in S^+, a \in A(s) arbitrarily except that Q(terminal,\cdot)=0
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    Q = np.zeros((num_states,num_actions), float)
    total_rewards = np.zeros(0)

    # Loop for each episode
    for episode in range(num_episodes):
        total_reward = 0

        # Initialize S
        S, info = env.reset()
        terminated, truncated = (0,0)

        # Loop for each step of episode
        while not (terminated or truncated):

            # Choose A from S using policy derived from Q (e.g., \epsilon-greedy)
            A = epsilon_greedy(epsilon, Q, S)

            # Take action A, observe R, S'
            Sp, reward, terminated, truncated, info = env.step(A)

            if not (terminated or truncated):
                # Q(S,A) <- Q(S,A) + \alpha[R + \gamma max_a(Q(S',a)) - Q(S,A)]
                Q[S, A] = Q[S, A] + alpha*(reward + gamma*(np.max(Q[Sp]))- Q[S, A])
            else:
                Q[S, A] = Q[S, A] + alpha*(reward - Q[S, A])
            total_reward += reward

            # S <- S'
            S = Sp

        total_rewards = np.append(total_rewards, total_reward)

    # Return the final Q-table and episode rewards.
    return Q, total_rewards
