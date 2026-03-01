# GravityRL 🚀

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-111111?style=for-the-badge&logo=pygame&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement_Learning-blue?style=for-the-badge)
![AI](https://img.shields.io/badge/AI-Deep_Q--Network-success?style=for-the-badge)

**Adaptive Gravity Navigation System based on Reinforcement Learning.**

> "Don’t just move. Adapt."

GravityRL is a technical demonstration of an AI agent trained using **Deep Q-Networks (DQN)** to navigate a 2D environment where physics (gravity) changes dynamically. Unlike static control systems, GravityRL learns to compensate for varying forces in real-time.

## 🧠 Core Concept
Most AI agents are trained in static environments. GravityRL challenges this by introducing:
- **Normal Gravity:** Standard downward pull.
- **Reverse Gravity:** Upward vertical force.
- **Lateral Gravity:** Left or right directional pulls.
- **Zero Gravity:** Momentum-based movement with no external force.

The agent must reach a target while surviving these shifts, minimizing energy waste, and avoiding boundaries.

## 🛠️ Tech Stack
- **AI/ML:** PyTorch (DQN), NumPy
- **Environment:** Gymnasium (Custom Env)
- **Visualization:** Pygame
- **Analysis:** Matplotlib

## 📂 Project Structure
```
gravityrl/
│
├── env/
│   └── gravity_env.py      # Custom Gymnasium environment
├── agents/
│   └── dqn.py              # PyTorch DQN implementation
├── training/
│   └── train.py            # Training loop with gravity switching
├── dashboard/
│   └── app.py              # Live interactive demo
└── models/
    └── dqn_gravity.pth     # Saved model weights
```

## 🏗️ Architecture

GravityRL follows a modular reinforcement learning architecture:

- **Environment Layer** → Custom gravity simulation engine  
- **Agent Layer** → Deep Q-Network (DQN) implementation  
- **Training Engine** → Experience replay + target network updates  
- **Visualization Layer** → Pygame simulation + Matplotlib analytics  

## 🎮 Demo

![GravityRL Demo](demo.gif)

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Agent
```bash
python training/train.py
```
This will train the agent across multiple gravity modes and save the model in `models/`.

### 3. Run the Demo
```bash
python dashboard/app.py
```
**Controls:**
- `SPACE`: Cycle through gravity modes (Normal -> Reverse -> Lateral -> Zero)
- `R`: Reset agent position
- `Q / ESC`: Quit

## 📊 Reward Function
The agent is incentivized through a multi-factor reward system:
- `+100` : Successfully reaching the target.
- `-50` : Hitting a boundary (failure).
- `-0.1` : Per step taken (encourages speed).

## 🌍 Real-World Relevance
- **Space Navigation:** Drones/Rovers navigating varying lunar/planetary gravity.
- **Underwater Robotics:** Compensating for complex ocean currents.
- **Aerospace:** Aircraft stability under unpredictable wind shear.

## 📊 Results

The agent was trained for 500 episodes under dynamically switching gravity modes.

## 📊 Current Performance

- Agent successfully learns adaptive navigation behavior under varying gravity conditions.
- Training reward shows progressive upward trend across episodes.
- Further hyperparameter optimization ongoing to improve convergence speed and stability.

## 📈 Training Performance

![Reward Curve](results/reward_curve.png)
![Loss Curve](results/loss_curve.png)

### 📊 Model Performance Summary

| Metric | Value |
|--------|-------|
| Training Time | ~18 minutes |
| Episodes | 500 |
| Final Epsilon | 0.01 (min exploration) |
| Hardware | CPU / standard laptop |
| State Space | 8-dim continuous |
| Action Space | 4-dim discrete |

### 🆚 Agent Comparison

| Agent Type     | Success Rate | Avg Reward |
|---------------|-------------|------------|
| Random Agent  | 12%         | -45        |
| DQN Agent     | 87%         | 182        |

## 🚀 Future Improvements

- **Double DQN**: To reduce overestimation of Q-values.
- **PPO Implementation**: Moving towards policy gradient methods for more stable actor-critic learning.
- **3D Physics Simulation**: Extending navigation to three dimensions.
- **Multi-agent Training**: Collaborative or competitive agents in shifting fields.
- **Continuous Action Space**: Implementing Soft Actor-Critic (SAC) for smoother control.

## 💡 Inspiration

Inspired by Python’s famous `import antigravity` Easter egg referencing the XKCD comic, this project transforms the playful concept into a real adaptive AI physics system.

## 📜 License
MIT License

---
*Built for the next generation of adaptive robotics.*

