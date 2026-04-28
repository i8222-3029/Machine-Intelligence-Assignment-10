# DQN Ablation 실험 스크립트
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.continuous_warehouse import ContinuousWarehouseEnv
from src.dqn_agent import DQNAgent

EPISODES = 2000

# Full DQN (replay + target)
env1 = ContinuousWarehouseEnv(seed=10)
agent1 = DQNAgent(env1, use_replay=True, use_target=True)
rewards1 = agent1.train(episodes=EPISODES)

# DQN without experience replay
env2 = ContinuousWarehouseEnv(seed=11)
agent2 = DQNAgent(env2, use_replay=False, use_target=True)
rewards2 = agent2.train(episodes=EPISODES)

# DQN without target network
env3 = ContinuousWarehouseEnv(seed=12)
agent3 = DQNAgent(env3, use_replay=True, use_target=False)
rewards3 = agent3.train(episodes=EPISODES)

plt.figure(figsize=(10,6))
plt.plot(uniform_filter1d(rewards1, size=100), label="Full DQN (replay+target)")
plt.plot(uniform_filter1d(rewards2, size=100), label="No Replay")
plt.plot(uniform_filter1d(rewards3, size=100), label="No Target Net")
plt.xlabel("Episode")
plt.ylabel("Reward (rolling avg, 100)")
plt.legend()
plt.title("DQN Ablation Study")
plt.show()
