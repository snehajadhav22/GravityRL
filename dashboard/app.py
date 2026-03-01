import sys
import os
import torch
import pygame
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gravity_env import GravityEnv
from agents.dqn import DQNAgent

def run_demo():
    env = GravityEnv(render_mode="human", gravity_type="normal")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, epsilon_start=0.0) # No exploration
    
    model_path = "models/dqn_gravity.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found. Running with random agent.")

    running = True
    gravity_modes = ["normal", "reverse", "zero", "lateral_right", "lateral_left"]
    current_mode_idx = 0
    
    print("\n--- GravityRL Demo Controls ---")
    print("Space: Change Gravity Mode")
    print("R: Reset Agent")
    print("ESC/Q: Quit")
    
    while running:
        state, info = env.reset()
        env.gravity_type = gravity_modes[current_mode_idx]
        done = False
        
        while not done and running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_SPACE:
                        current_mode_idx = (current_mode_idx + 1) % len(gravity_modes)
                        print(f"Switched Gravity to: {gravity_modes[current_mode_idx]}")
                        done = True # Restart env with new gravity
                    if event.key == pygame.K_r:
                        done = True # Reset
            
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            env.render()
            
            # Simple UI overlay would be nice, but Pygame render already shows gravity arrow
            
    env.close()

if __name__ == "__main__":
    run_demo()
