---
layout: post
title: Pacman using Reinforcement Learning
---
<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

This project uses techniques of reinforcement learning, specifically value iteration and Q-learning to play the classic game of Pacman. The agents designed are tested on a simulated robot controller (Crawler) and then on Pacman. The code for the project can be found [here](https://github.com/simpeijie/CS188-Artificial-Intelligence/tree/master/reinforcement).

<!-- ![pacman_gridworld]({{ site.baseurl }}/images/pacman_gridworld.png) -->

## Background

### Markov Decision Process

The first agent we are concerned with is the value iteration agent - an offline planner. The agent has full knowledge of both the reward function and transition function, i.e. all the information they need to precompute optimal actions in the world encoded by the Markov Decision Process in question. 

A transition function $T(s, a, s')$ is a probability function which represents the probability that an agent taking an action, $a \in A$ from a state $s \in S$ ends up in a state $s' \in S$. 

A reward function $R(s, a, s')$ is a function that returns either a positive or negative number depending on whether or not an agent in a state $s \in S$ takes an action $a \in A$ and ends up in state $s' \in S$ that benefits it. The objective is to acquire the maximum reward possible before arriving at some terminal state.  

### The Bellman Equation

The equation that is used for MDP's is the Bellman Equation. There are two mathematical quantities that are important in understanding this equation. 
* The optimal value of a state, $s$, $V^*(s)$ - the optimal value of $s$ is the expected value of the utility an optimally-behaving agent that starts in $s$ receives .

* The optimal value of a q-state $(s,a)$, $Q^*(s,a)$ - the optimal value of $(s,a)$ is the expected value of the utility an agent receives after starting in $s$, taking $a$, and acting optimally henceforth.

The Bellman Equation is defined as follows: 

$$ V^*(s) = \underset{a}{\text{max}} \sum_{s'} T(s, a, s')[R(s, a, s') + \gamma V^*(s')]$$

where $\gamma$ is the discount.

### Value Iteration

Next, let's introduce a [dynamic programming algorithm](https://en.wikipedia.org/wiki/Dynamic_programming) that computes a value based off of values previously computed, known as value iteration. We are essentially computing the value $V_{k+1}$ with $V_k$, where $k$ represents a timestep.

We first initialize $V_0 (s) = 0$, then follow the update rule until convergence:

$$ \forall s \in S, V^*(s) = \underset{a}{\text{max}} \sum_{s'} T(s, a, s')[R(s, a, s') + \gamma V^*(s')]$$ 

Convergence is defined such that $\forall s \in S, V_k(s) = V_{k+1}(s) = V^*(s)$.

### Temporal Difference Learning 

In contrast to offline planning, online planning does not have prior knowledge of the transition function and reward function. The agent must try exploration, during which it performs actions and receives feedback in the form of the successor states it arrives in and the corresponding rewards it reaps.

Before delving into Q-learning, we first have to understand temporal difference learning. Temporal difference learning keeps track of an exponential moving average of the value at each state and at each timestep, it obtains a sample value of taking an action $\pi(s)$ from state $s$ and ending up in $s'$. The sample value is calculated as such:

$$sample = R(s, \pi(s), s') + \gamma V^\pi(s')$$

Then, we incorporate the sample (new estimate for $V^\pi(s)$) into the existing model of $V^\pi(s)$

$$V^\pi(s) = (1 - \alpha)V^\pi(s) + \alpha \cdot sample$$

where $alpha$ is the learning rate that specifies the weight to assign the existing mdoel and the weight to assign the new sampled estimate. 

## Q-Learning

TD learning does not enable us to find the optimal policy for our agent, which requires knowledge of the q-values of states. Unlike TD learning, Q-learning learns the q-values of states directly, bypassing the need to know any values, transition functions, or reward functions. The update rule for Q-learning is as follows:

$$Q_{k+1}(s, a) = \sum_{s'} T(s, a, s')[R(s, a, s') + \gamma \underset{a'}{\text{max}} Q_k(s', a')]$$

which is a slight modification over the update rule for value iteration. With this update rule, Q-learning is derived essentially the same way as TD learning, by acquiring q-value samples: 

$$sample = R(s, a, s') + \gamma \underset{a'}{\text{max}} + \gamma Q(s', a')$$

and the exponential moving average is:

$$Q(s, a) = (1 - \alpha) Q(s, a) + \alpha \cdot sample$$

## Implementation

With the knowledge in the first part of this project, I implemented value iteration to show the optimal value, $V^*(s)$ after $k$ steps.  

To display the GUI, run 
``` 
python gridworld.py -a value -i 5
```

<img src="{{ site.baseurl }}/images/value-iteration.png" width="300" class="center-image">

The agent then selects the best action for a particular state according to the values and computes the Q-value of each (state, action) pair. 

<img src="{{ site.baseurl }}/images/q-values.png" width="300" class="center-image">

# Q-Learning

The value iteration agent does not learn from experience. It attempts to construct an MDP model to arrive at a complete policy before interacting with a real environment. When it interacts with the environment, it simply follows the precomputed policy. However, in the real world, the MDP is not available. 

Therefore, a Q-learning agent, which does very little on construction, but instead learns by trial and error from interatctions with the environment is more appropriate for this setting.

Here's a demo of the agent learning for 5 episodes. 
<img src="{{ site.baseurl }}/images/q-learning.gif" width="300" class="center-image">

To watch the Q-learner learn under manual control, run  
``` 
python gridworld.py -a q -k 5 -m
```

The Q-learning agent is complete with the implementation of epsilon-greedy action selection, meaning it chooses random actions an epsilon fraction of the time, and follows its current best Q-values otherwise. It works on 
