import numpy as np
from src.continuous_warehouse import ContinuousWarehouseEnv

class DiscretizedQLearningAgent:
    def __init__(self, env, grid_size=10, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.env = env
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((grid_size, grid_size, env.action_size))

    def discretize(self, state):
        x, y = state[0], state[1]
        x_bin = int((x - self.env.xlim[0]) / (self.env.xlim[1] - self.env.xlim[0]) * self.grid_size)
        y_bin = int((y - self.env.ylim[0]) / (self.env.ylim[1] - self.env.ylim[0]) * self.grid_size)
        x_bin = np.clip(x_bin, 0, self.grid_size - 1)
        y_bin = np.clip(y_bin, 0, self.grid_size - 1)
        return x_bin, y_bin

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_size)
        x_bin, y_bin = self.discretize(state)
        return np.argmax(self.q_table[x_bin, y_bin])

    def update(self, state, action, reward, next_state, done):
        x_bin, y_bin = self.discretize(state)
        nx_bin, ny_bin = self.discretize(next_state)
        best_next = np.max(self.q_table[nx_bin, ny_bin])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[x_bin, y_bin, action]
        self.q_table[x_bin, y_bin, action] += self.alpha * td_error

    def train(self, episodes=2000):
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
            rewards.append(ep_reward)
        return rewards

if __name__ == "__main__":
    env = ContinuousWarehouseEnv()
    agent = DiscretizedQLearningAgent(env)
    rewards = agent.train(episodes=2000)
    # Plotting code can be added here
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    plt.plot(uniform_filter1d(rewards, size=100), label="Discretized Q-learning")
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling avg)")
    plt.legend()
    plt.show()
