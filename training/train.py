import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gravity_env import GravityEnv
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer

def train():
    # Parameters
    episodes = 1200
    batch_size = 64
    buffer_capacity = 10000
    update_target_every = 10
    
    # Initialize Env & Agent
    env = GravityEnv(render_mode=None, gravity_type="normal")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon_decay=0.995)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    rewards_history = []
    loss_history = []
    
    print("Starting training...")
    for episode in tqdm(range(episodes)):
        # Randomly change gravity every few episodes to encourage adaptation
        if episode % 50 == 0:
            gravity_modes = ["normal", "reverse", "zero", "lateral_right", "lateral_left"]
            mode = random.choice(gravity_modes)
            env.gravity_type = mode
            print(f"\nSwitched gravity mode to: {mode}")
            
        state, info = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps_in_episode = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                loss = agent.update(replay_buffer.sample(batch_size))
                episode_loss += loss
                steps_in_episode += 1
                
        if episode % update_target_every == 0:
            agent.update_target_network()
            
        rewards_history.append(episode_reward)
        if steps_in_episode > 0:
            loss_history.append(episode_loss / steps_in_episode)
        else:
            loss_history.append(0)
            
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1} | Total Reward: {episode_reward:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    agent.save("models/dqn_gravity.pth")
    print("Training finished. Model saved.")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("results/reward_curve.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Episode")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.savefig("results/loss_curve.png")
    plt.close()

if __name__ == "__main__":
    train()
