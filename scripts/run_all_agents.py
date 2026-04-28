# 실험 실행 예시 스크립트
# 이 스크립트는 세 가지 에이전트의 학습 곡선을 한 번에 비교합니다.

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from src.continuous_warehouse import ContinuousWarehouseEnv
from src.discretized_q import DiscretizedQLearningAgent
from src.tile_coded_q import TileCodedQLearningAgent
from src.dqn_agent import DQNAgent

EPISODES = 2000

# Discretized Q-learning
env1 = ContinuousWarehouseEnv(seed=0)
agent1 = DiscretizedQLearningAgent(env1)
rewards1 = agent1.train(episodes=EPISODES)

# Tile-coded Q-learning
env2 = ContinuousWarehouseEnv(seed=1)
agent2 = TileCodedQLearningAgent(env2)
rewards2 = agent2.train(episodes=EPISODES)

# DQN
env3 = ContinuousWarehouseEnv(seed=2)
agent3 = DQNAgent(env3)
rewards3 = agent3.train(episodes=EPISODES)

plt.figure(figsize=(10,6))
plt.plot(uniform_filter1d(rewards1, size=100), label="Discretized Q-learning")
plt.plot(uniform_filter1d(rewards2, size=100), label="Tile-coded Q-learning")
plt.plot(uniform_filter1d(rewards3, size=100), label="DQN")
plt.xlabel("Episode")
plt.ylabel("Reward (rolling avg, 100)")
plt.legend()
plt.title("Learning Curve Comparison")
plt.show()
