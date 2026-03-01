import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class GravityEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, gravity_type="normal", gravity_strength=0.8):
        super(GravityEnv, self).__init__()
        self.gravity_strength = gravity_strength
        
        self.window_size = 600
        self.grid_size = 20
        self.cell_size = self.window_size // self.grid_size
        
        # State: [px, py, vx, vy, tx, ty, gx, gy]
        # p=pos, v=vel, t=target, g=gravity
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, 0, 0, -1, -1]),
            high=np.array([self.window_size, self.window_size, 10, 10, self.window_size, self.window_size, 1, 1]),
            dtype=np.float32
        )
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        self.render_mode = render_mode
        self.gravity_type = gravity_type
        self.window = None
        self.clock = None
        
        self.reset()

    def _get_gravity(self):
        s = self.gravity_strength
        if self.gravity_type == "normal":
            return np.array([0.0, s])
        elif self.gravity_type == "reverse":
            return np.array([0.0, -s])
        elif self.gravity_type == "zero":
            return np.array([0.0, 0.0])
        elif self.gravity_type == "lateral_right":
            return np.array([s, 0.0])
        elif self.gravity_type == "lateral_left":
            return np.array([-s, 0.0])
        return np.array([0.0, s])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent_pos = np.array([50.0, 50.0])
        self.agent_vel = np.array([0.0, 0.0])
        self.target_pos = np.array([550.0, 550.0])
        
        self.gravity = self._get_gravity()
        
        self.steps = 0
        self.max_steps = 500
        self.prev_dist = np.linalg.norm(self.agent_pos - self.target_pos)
        
        observation = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def _get_obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.agent_vel[0], self.agent_vel[1],
            self.target_pos[0], self.target_pos[1],
            self.gravity[0], self.gravity[1]
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        # Action forces
        force = 1.0
        if action == 0: # Up
            self.agent_vel[1] -= force
        elif action == 1: # Down
            self.agent_vel[1] += force
        elif action == 2: # Left
            self.agent_vel[0] -= force
        elif action == 3: # Right
            self.agent_vel[0] += force
            
        # Apply Gravity
        self.agent_vel += self.gravity
        
        # Friction/Damping
        self.agent_vel *= 0.95
        
        # Update Position
        self.agent_pos += self.agent_vel
        
        # Check boundaries
        terminated = False
        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = (self.prev_dist - dist) * 0.1 # Progress-based reward
        self.prev_dist = dist
        
        if (self.agent_pos[0] < 0 or self.agent_pos[0] > self.window_size or
            self.agent_pos[1] < 0 or self.agent_pos[1] > self.window_size):
            terminated = True
            reward = -20.0 # Reduced crash penalty
            
        # Check target reach
        if dist < 20:
            terminated = True
            reward = 100.0 # Success reward
            
        # Time limit
        truncated = self.steps >= self.max_steps
        
        observation = self._get_obs()
        info = {"distance": dist}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((30, 30, 30))
        
        # Draw target
        pygame.draw.circle(canvas, (0, 255, 0), self.target_pos.astype(int), 15)
        
        # Draw agent
        pygame.draw.circle(canvas, (0, 150, 255), self.agent_pos.astype(int), 10)
        
        # Draw gravity arrow (indicator)
        start_pt = (50, 50)
        end_pt = (50 + self.gravity[0] * 40, 50 + self.gravity[1] * 40)
        pygame.draw.line(canvas, (255, 255, 0), start_pt, end_pt, 3)
        
        if self.render_mode == "human":
            if self.window is not None:
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
