# 6D 확장 실험 스크립트 (보너스)
import numpy as np
from src.continuous_warehouse import ContinuousWarehouseEnv
from src.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class Warehouse6DEnv(ContinuousWarehouseEnv):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.state_size = 6
        self.load_lim = (0.0, 1.0)
        self.battery_lim = (0.0, 1.0)
    def reset(self):
        base = super().reset()
        load = self.rng.uniform(*self.load_lim)
        battery = self.rng.uniform(*self.battery_lim)
        self.state = np.concatenate([base, [load, battery]])
        self.step_count = 0
        return self.state.copy()
    def step(self, action):
        # 6D 상태에서 4D만 부모에 넘김
        base_state = self.state[:4]
        load, battery = self.state[4], self.state[5]
        # 부모의 state를 4D로 임시 설정
        self.state = base_state
        next_base, reward, done, info = super().step(action)
        # load: 고정, battery: 감소
        battery = max(battery - 0.01, 0.0)
        # 속도는 load, battery에 따라 감소
        v = next_base[3] * (1 - 0.5 * load) * (0.5 + 0.5 * battery)
        # 6D로 재조립
        self.state = np.array([next_base[0], next_base[1], next_base[2], v, load, battery])
        # 보상: load가 높을수록 goal 도달시 보상 증가
        dist_goal = np.linalg.norm(self.state[:2] - self.goal)
        if dist_goal < self.goal_radius:
            reward = 1.0 + load
            done = True
        return self.state.copy(), reward, done, info
    def normalize_state(self, state=None):
        if state is None:
            state = self.state
        base = super().normalize_state(state[:4])
        load = (state[4] - self.load_lim[0]) / (self.load_lim[1] - self.load_lim[0])
        battery = (state[5] - self.battery_lim[0]) / (self.battery_lim[1] - self.battery_lim[0])
        return np.concatenate([base, [load, battery]])

def q_table_size():
    bins = 10
    return bins ** 6

if __name__ == "__main__":
    print(f"Q-table size for 6D: {q_table_size():,}")
    env = Warehouse6DEnv(seed=42)
    agent = DQNAgent(env)
    rewards = agent.train(episodes=1000)
    plt.plot(uniform_filter1d(rewards, size=100), label="DQN 6D")
    plt.xlabel("Episode")
    plt.ylabel("Reward (rolling avg, 100)")
    plt.legend()
    plt.title("DQN in 6D Warehouse")
    plt.show()
