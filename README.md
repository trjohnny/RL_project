# Reinforcement Learning Project: Panda Training Framework
This repository is a framework used to train RL agents in a Panda environment [[1]](#1). The main purpose of it is to test different algorithms on different tasks to solve complex problems in a 3-dimensional space.

## Download from source

```
git clone https://github.com/trjohnny/RL_project.git
```

## Usage

```
py main.py --env [environment] --algo [algorithm] --verbose [1,2 or 3] --episodes [episodes]
```

* environment: listed here ----> https://github.com/qgallouedec/panda-gym
* algo: A2C (Actor Critic) | A2C_DISCRETE (Actor Critic with discrete actions aggregations) | DDPG (Deep Deterministic Policy Gradient) | A2C_N_STEP_AHEAD (Actor Critic TD(n) )
* verbose: 1 (print every 100 episodes) | 2 (every 10) | 3 (print every episode)
* episodes: the number of episodes to run

## Follow up

Iperparameter optimization is still under review but will be ready soon :)

## References
<a id="1">[1]</a> 
Gallouédec, Quentin and Cazin, Nicolas and Dellandréa, Emmanuel and Chen, Liming
panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning
4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS, 2021
