import numpy as np
from src.continuous_warehouse import ContinuousWarehouseEnv

class TileCoder:
    def __init__(self, xlim, ylim, num_tilings=8, tiles_per_dim=4):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.xlim = xlim
        self.ylim = ylim
        self.tile_width = (xlim[1] - xlim[0]) / (tiles_per_dim - 1)
        self.offsets = [
            (i * self.tile_width / num_tilings, i * self.tile_width / num_tilings)
            for i in range(num_tilings)
        ]
        self.total_tiles = num_tilings * tiles_per_dim * tiles_per_dim

    def encode(self, x, y):
        features = np.zeros(self.total_tiles)
        for t in range(self.num_tilings):
            x_offset, y_offset = self.offsets[t]
            x_bin = int(((x - self.xlim[0] + x_offset) / (self.xlim[1] - self.xlim[0])) * self.tiles_per_dim)
            y_bin = int(((y - self.ylim[0] + y_offset) / (self.ylim[1] - self.ylim[0])) * self.tiles_per_dim)
            x_bin = np.clip(x_bin, 0, self.tiles_per_dim - 1)
            y_bin = np.clip(y_bin, 0, self.tiles_per_dim - 1)
            idx = t * self.tiles_per_dim * self.tiles_per_dim + x_bin * self.tiles_per_dim + y_bin
            features[idx] = 1
        return features

class TileCodedQLearningAgent:
    def __init__(self, env, num_tilings=8, tiles_per_dim=4, epsilon=0.1, alpha=0.1, gamma=0.99):
        self.env = env
        self.tile_coder = TileCoder(env.xlim, env.ylim, num_tilings, tiles_per_dim)
        self.epsilon = epsilon
        self.alpha = alpha / num_tilings
        self.gamma = gamma
        self.n_actions = env.action_size
        self.weights = np.zeros((self.n_actions, self.tile_coder.total_tiles))

    def get_features(self, state):
        x, y = state[0], state[1]
        return self.tile_coder.encode(x, y)

    def q_values(self, state):
        features = self.get_features(state)
        return np.dot(self.weights, features)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_values(state))

    def update(self, state, action, reward, next_state, done):
        features = self.get_features(state)
        next_q = np.max(self.q_values(next_state))
        target = reward + self.gamma * next_q * (not done)
        prediction = np.dot(self.weights[action], features)
        td_error = target - prediction
        self.weights[action] += self.alpha * td_error * features

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
    agent = TileCodedQLearningAgent(env)
    rewards = agent.train(episodes=2000)
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    plt.plot(uniform_filter1d(rewards, size=100), label="Tile-coded Q-learning")
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling avg)")
    plt.legend()
    plt.show()
