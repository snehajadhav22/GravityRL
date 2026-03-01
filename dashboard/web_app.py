import streamlit as st
import numpy as np
import torch
import sys
import os
import time
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gravity_env import GravityEnv
from agents.dqn import DQNAgent

st.set_page_config(page_title="GravityRL Dashboard", layout="wide")

st.title("🚀 GravityRL: Deep Q-Learning in Dynamic Gravity")
st.markdown("""
This dashboard runs the **GravityRL** agent directly in your browser. 
The agent uses a Deep Q-Network to adapt to real-time changes in physics.
""")

# Sidebar controls
st.sidebar.header("🛠️ Simulation Controls")
gravity_mode = st.sidebar.selectbox(
    "Select Gravity Mode",
    ["normal", "reverse", "zero", "lateral_right", "lateral_left"]
)

model_path = "models/dqn_gravity.pth"
load_model = st.sidebar.checkbox("Load Trained Model", value=os.path.exists(model_path))

st.sidebar.divider()
st.sidebar.markdown("**Hardware:** CPU")
st.sidebar.markdown("**Status:** Running" if load_model else "**Status:** Random Agent")

# Metrics placeholders
col1, col2, col3 = st.columns(3)
reward_metric = col1.empty()
dist_metric = col2.empty()
step_metric = col3.empty()

# Simulation display
sim_placeholder = st.empty()

def run_simulation():
    env = GravityEnv(render_mode="rgb_array", gravity_type=gravity_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, epsilon_start=0.0)
    
    if load_model and os.path.exists(model_path):
        agent.load(model_path)
        st.sidebar.success("Model Loaded Successfully!")
    
    state, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # User might have changed gravity mode in selectbox mid-way
        env.gravity_type = gravity_mode
        
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Capture frame
        frame = env.render()
        img = Image.fromarray(frame)
        
        # Update UI
        sim_placeholder.image(img, caption=f"Agent navigating in {gravity_mode} gravity", use_container_width=True)
        reward_metric.metric("Total Reward", f"{total_reward:.2f}")
        dist_metric.metric("Distance to Target", f"{info['distance']:.2f}")
        step_metric.metric("Step", f"{env.steps}")
        
        time.sleep(0.01) # Smoothness
        
    if terminated and reward > 50:
        st.balloons()
        st.success("Target Reached! 🥳")
    elif terminated:
        st.error("Agent Crashed! 💥")
    else:
        st.warning("Time Limit Reached! ⏳")

if st.sidebar.button("▶️ Start Simulation"):
    run_simulation()
else:
    st.info("Adjust the settings in the sidebar and click 'Start Simulation' to begin.")

# Display Training Results if available
if os.path.exists("results/reward_curve.png"):
    st.divider()
    st.header("📈 Training Performance")
    c1, c2 = st.columns(2)
    c1.image("results/reward_curve.png", caption="Reward Curve")
    c2.image("results/loss_curve.png", caption="Loss Curve")
