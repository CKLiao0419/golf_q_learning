# Q-Learning Golf

[中文版](README_zh-TW.md) | English

A simple reinforcement learning project that trains an agent to play mini-golf using Q-Learning.

## Overview

This project implements a Q-Learning agent that learns to hit a golf ball into a hole. The agent explores different angles and forces to find optimal shots based on the hole's position.

## Requirements
```bash
pip install pygame numpy matplotlib
```

## Usage

Run the training and demo:
```bash
python main.py
```

The program will:
1. Train the agent for 12 million episodes
2. Display a learning curve showing reward progress
3. Enter demo mode where you can watch the trained agent play

Press `Q` to quit the demo.

## Project Structure

- `main.py` - Training loop and demo mode
- `src/agent.py` - Q-Learning agent implementation
- `src/env.py` - Golf environment with physics simulation
- `src/config.py` - Game parameters and hyperparameters

## Q-Learning Algorithm

The agent uses the Q-Learning update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r - Q(s, a)]$$

Where:
- $Q(s, a)$ is the Q-value for state $s$ and action $a$
- $\alpha$ is the learning rate (0.17)
- $r$ is the immediate reward

Since this is an episodic task (one shot per episode), the target simplifies to just the reward $r$ without considering future states.

## Reward Function

$$\text{reward} = \max(0, 100 - \frac{d}{3}) + \begin{cases} 
1000 & \text{if ball in hole} \\
0 & \text{otherwise}
\end{cases}$$

Where $d$ is the Euclidean distance between the ball and the hole.

## State and Action Space

- **State Space**: Grid-based discretization of hole positions (66 × 100 grid cells)
- **Action Space**: 663 actions (51 angles × 13 force levels)
  - Angles: -45° to 45° in 51 steps
  - Force: 0.2 to 1.0 in 13 steps

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate (α) | 0.17 |
| Discount Factor (γ) | 0.95 |
| Initial Epsilon (ε) | 1.0 |
| Min Epsilon | 0.01 |
| Epsilon Decay | 0.9999997 |

## Results

After training, the agent learns to consistently hit the ball close to or into the hole from a fixed starting position, adapting its shot based on the hole's location.

## License

MIT