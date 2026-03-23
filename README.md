# Catch Game - Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.x-green.svg)
![Reinforcement Learning](https://img.shields.io/badge/AI-Reinforcement_Learning-orange.svg)

Welcome to the **Catch Game RL Project**! This project is a comprehensive implementation of various Reinforcement Learning algorithms solving a custom-built arcade-style "Catch Game" using **Pygame**. 

In this environment, an agent controls a basket at the bottom of the screen and must catch "good" items (apples) while avoiding "bad" ones (bombs), learning purely through trial and error over thousands of episodes.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Game Architecture & Environment](#game-architecture--environment)
3. [Implemented Algorithms](#implemented-algorithms)
4. [Project Structure](#project-structure)
5. [Getting Started (Installation & Usage)](#getting-started)
6. [Challenges & State Space Analysis](#challenges--state-space-analysis)
7. [Results & Demonstrations](#results--demonstrations)

---

## 🎯 Project Overview
This project serves as a testbed and educational display of classic Reinforcement Learning algorithms. An interactive GUI allows you to train different RL agents from scratch and observe their learned policies in real-time. 

### The Goal
The agent (a basket) must position itself correctly horizontally to:
- **Catch** the green apples (+10 Reward)
- **Avoid** the dark bombs (-10 Reward)
- **Minimize unnecessary steps** (-0.01 Reward per step)
- **Prevent missed apples** (-1 Reward)

---

## 🏗️ Game Architecture & Environment

The environment strictly adheres to the standard **OpenAI Gym interface conventions**, abstracting away logic into cohesive `reset()`, `step()`, and `render()` functions. 

### State/Observation Space
Given the continuous nature of screen pixels, the space is discretized internally to keep the state tracking finite:
- Agent X Grid Position
- Item X Grid Position
- Item Y Grid Position
- Item Type (0 = Good, 1 = Bad)

### Action Space
The agent has a discrete action space of **3 actions**:
- `0` : Move Left ⬅️
- `1` : Stay / Do Nothing 🛑
- `2` : Move Right ➡️

---

## 🧠 Implemented Algorithms

The project features three fundamental Reinforcement Learning approaches, each demonstrating distinct learning paradigms:

### 1. Q-Learning (`q_learning.py`)
- **Type**: Model-Free, Value-Based, Off-Policy
- **Mechanism**: Utilizes a tabular Q-learning approach mapped to the discretized state environment. 
- **Exploration**: Uses an $\epsilon$-greedy strategy with a geometric decay factor (`EPSILON_DECAY = 0.995`) to gradually shift from exploration to exploitation.
- **Hyperparameters**: `Episodes = 5000` | `Alpha = 0.1` | `Gamma = 0.95`

### 2. Tabular Value Iteration (with Reward Shaping) (`value_iteration.py`)
- **Mechanism**: A variant heavily utilizing **Reward Shaping**. To combat sparse rewards, the agent receives a small intermediate reward (`±0.1`) based on whether it is moving closer to or further from the item dynamically.
- **State Modification**: This implementation uses a finer grid discretization logic (`obs // 10`) for the Q-table tracking.
- **Hyperparameters**: `Episodes = 20000` | `Alpha = 0.2` | `Gamma = 0.95`

### 3. Policy Gradient w/ Baseline (`policy_gradient.py`)
- **Type**: Policy-Based, On-Policy
- **Mechanism**: A tabular variant of the classic REINFORCE algorithm utilizing a parameterized policy (`theta`) and softmax action selection. 
- **Variance Reduction**: Incorporates a running-average **baseline** computed across episode rewards. This subtraction logic stabilizes parameter updates and effectively reduces gradient variance.
- **Hyperparameters**: `Episodes = 5000` | `Alpha = 0.01` | `Gamma = 0.95`

---

## 📂 Project Structure

```text
MAR-game/
│
├── catch_env.py           # The core simulator environment & rendering logic
├── main_menu.py           # Interactive Pygame GUI, orchestrates training flags
├── q_learning.py          # Q-Learning implementation script
├── value_iteration.py     # Value Iteration (reward shaped Q-variant script)
├── policy_gradient.py     # REINFORCE Policy Gradient script
├── *.npy                  # Auto-generated saved policies (Q-tables/Theta matrices)
└── README.md              # Project Documentation (You are here)
```

---

## 🚀 Getting Started

### Prerequisites
You need Python 3 installed and the following dependencies:
```bash
pip install pygame numpy
```

### Running the App
The entire project can be driven smoothly using the dynamic GUI:
```bash
python main_menu.py
```
From the interactive menu, you can click on any of the three algorithms to instantly kick off training. Once training completes, the application will automatically enter a demonstration mode, showing you the agent acting upon its newly acquired intelligence!

---

## 🚧 Challenges & State Space Analysis

1. **Continuous to Discrete Dimension Scaling**
   * *Challenge*: Tracking pure X/Y pixel coordinates inflates the state space to an unmanageable size for tabular methods.
   * *Solution*: Discretizing the state using `GRID_SIZE`. In the Value Iteration iteration, grid indices were further honed (dividing by `10`) to provide precise yet trackable sub-states for the Q-table.

2. **The "Sparse Reward" Problem**
   * *Challenge*: The agent initially acts via random Brownian motion, taking a long time to accidentally catch its first target and begin chaining positive reinforcement.
   * *Solution*: **Reward Shaping** (found in `value_iteration.py`). By feeding the agent micro-rewards when its absolute horizontal distance to the falling item decreases, convergence times heavily accelerated.

3. **High Variance in Monte Carlo Updates**
   * *Challenge*: Policy Gradient algorithms tend to suffer from noisy gradients resulting in policy collapse.
   * *Solution*: Maintaining and subtracting a tracking `baseline` metric (a moving average of historical rewards).

---

## 📈 Results & Demonstrations

When running the models, the performance metrics `.npy` files tracking episode rewards are saved locally. 
- **Q-Learning** quickly locks into policies due to off-policy bootstrapping.
- **Value Iteration** (with the reward shaping distance heuristic) yields the most robust agent movements as it actively "tracks" the objects rather than camping statically.
- **Policy Gradient** achieves excellent final performance but exhibits more training volatility initially before the baseline stabilizes.

---
*Created as an exploration of fundamental Reinforcement Learning concepts.*
