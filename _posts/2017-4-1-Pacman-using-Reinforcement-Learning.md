---
layout: post
title: Pacman using Reinforcement Learning
---

This project uses techniques of reinforcement learning, specifically value iteration and Q-learning to play the classic game of Pacman. The agents designed are tested on a simulated robot controller (Crawler) and then on Pacman.

<!-- ![pacman_gridworld]({{ site.baseurl }}/images/pacman_gridworld.png) -->

## Value Iteration

The first agent we are concerned with is the value iteration agent - an offline planner. The agent takes a Markov Decision Process on construction and runs value iteration for a desired number of iterations. Value iteration computes k-step estimates of the optimal values, V<sub>k</sub>. 

To display the GUI, run 
``` 
python gridworld.py -a value -i 5
```

Upon running the command, the output looks like:
![value-iteration]({{ site.baseurl}}/images/value-iteration.png)

The agent then selects the best action for a particular state according to the values and computes the Q-value of each (state, action) pair. 

![q-values]({{ site.baseurl}}/images/q-values.png)


