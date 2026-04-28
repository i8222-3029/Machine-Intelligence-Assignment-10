import numpy as np

class ContinuousWarehouseEnv:
    def __init__(self, seed=None):
        # State: (x, y, theta, v)
        self.xlim = (0.0, 4.0)
        self.ylim = (0.0, 4.0)
        self.theta_lim = (0.0, 2 * np.pi)
        self.v_lim = (0.0, 1.0)
        self.state = None
        self.goal = np.array([3.5, 3.5])
        self.hazard = np.array([2.0, 2.0])
        self.goal_radius = 0.5
        self.hazard_radius = 0.5
        self.dt = 0.1
        self.max_steps = 200
        self.step_count = 0
        self.rng = np.random.RandomState(seed)
        self.action_map = {
            0: np.array([0, 1]),   # North
            1: np.array([0, -1]),  # South
            2: np.array([1, 0]),   # East
            3: np.array([-1, 0]),  # West
        }
        self.action_size = 4
        self.state_size = 4

    def reset(self):
        self.state = np.array([
            self.rng.uniform(0.5, 1.0),
            self.rng.uniform(0.5, 1.0),
            self.rng.uniform(0, 2 * np.pi),
            self.rng.uniform(0.0, 0.2)
        ])
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        x, y, theta, v = self.state
        # Action: change heading and velocity
        dtheta = 0.0
        dv = 0.0
        if action == 0:  # North
            dtheta = 0.0
            dv = 0.05
        elif action == 1:  # South
            dtheta = np.pi
            dv = 0.05
        elif action == 2:  # East
            dtheta = np.pi / 2
            dv = 0.05
        elif action == 3:  # West
            dtheta = 3 * np.pi / 2
            dv = 0.05
        # Update heading and velocity
        theta = (theta + dtheta) % (2 * np.pi)
        v = np.clip(v + dv + self.rng.normal(0, 0.01), self.v_lim[0], self.v_lim[1])
        # Move
        dx = v * np.cos(theta) * self.dt + self.rng.normal(0, 0.01)
        dy = v * np.sin(theta) * self.dt + self.rng.normal(0, 0.01)
        x = np.clip(x + dx, self.xlim[0], self.xlim[1])
        y = np.clip(y + dy, self.ylim[0], self.ylim[1])
        self.state = np.array([x, y, theta, v])
        self.step_count += 1
        # Reward
        dist_goal = np.linalg.norm(self.state[:2] - self.goal)
        dist_hazard = np.linalg.norm(self.state[:2] - self.hazard)
        done = False
        reward = -0.01  # step penalty
        if dist_goal < self.goal_radius:
            reward = 1.0
            done = True
        elif dist_hazard < self.hazard_radius:
            reward = -1.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True
        return self.state.copy(), reward, done, {}

    def render(self):
        # Optional: implement visualization if needed
        pass

    def normalize_state(self, state=None):
        if state is None:
            state = self.state
        x = (state[0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        y = (state[1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        theta = state[2] / (2 * np.pi)
        v = (state[3] - self.v_lim[0]) / (self.v_lim[1] - self.v_lim[0])
        return np.array([x, y, theta, v])
