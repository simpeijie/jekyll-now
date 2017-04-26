---
layout: post
title: Pacman using Reinforcement Learning
---
This project uses techniques of reinforcement learning, specifically value iteration and Q-learning to play the classic game of Pacman. The agents designed are tested on a simulated robot controller (Crawler) and then on Pacman. The code for this project can be found [here](https://github.com/simpeijie/CS188-Artificial-Intelligence/tree/master/reinforcement).

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

In contrast to offline planning, online planning does not have prior knowledge of the transition function and reward function. The agent must try exploration, during which it performs actions and receives feedback in the form of the successor states it arrives in and the corresponding rewards it reaps. In the real world, temporal difference learning and Q-learning are more useful since an MDP will not always be available. 

Before delving into Q-learning, we first have to understand temporal difference learning. Temporal difference learning keeps track of an exponential moving average of the value at each state and at each timestep, it obtains a sample value of taking an action $\pi(s)$ from state $s$ and ending up in $s'$. The sample value is calculated as such:

$$sample = R(s, \pi(s), s') + \gamma V^\pi(s')$$

Then, we incorporate the sample (new estimate for $V^\pi(s)$) into the existing model of $V^\pi(s)$

$$V^\pi(s) = (1 - \alpha)V^\pi(s) + \alpha \cdot sample$$

where $\alpha$ is the learning rate that specifies the weight to assign the existing mdoel and the weight to assign the new sampled estimate. 

### Q-Learning

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

where the option `-i` specifies the number of timesteps/iterations to run.

<img src="{{ site.baseurl }}/images/value-iteration.png" width="300" class="center-image">

The agent then selects the best action for a particular state according to the values and computes the q-value of each (state, action) pair. 

<img src="{{ site.baseurl }}/images/q-values.png" width="300" class="center-image">

Here, each quarter of a square represents the q-value of taking an action $a$ that is either North, South, East or West from the current state $s$.

With Q-learning update in place, we can watch the agent learn under manual control, using the keyboard:

```
python gridworld.py -a q -k 5 -m
```
<img src="{{ site.baseurl }}/images/q-learning.gif" width="300" class="center-image">

The Q-learning agent is complete with the implementation of epsilon-greedy action selection, meaning it chooses random actions an epsilon fraction of the time, and follows its current best q-values otherwise. Choosing a random action may result in choosing the best action. Hence, the agent will choose any random legal action rather than a random sub-optimal action. 

The epsilon-greedy action selection is coded generically so that it works on gridworld as well as on the Q-learning crawler robot. To see the robot in action, we run:

```
python crawler.py
```
<iframe width="540" height="315" class="center-image" src="http://www.youtube.com/embed/SROJrjSZg0Q" frameborder="0" allowfullscreen></iframe><b>

The crawler starts off undecisive and wobbly but as it moves, it learns the best actions to progress after which it moves quicker and more steadily. The same applies to Pacman, who in the beginning of the game, perishes but later wins the game consistently. 

There are two phases in which Pacman will play this game. The first is training, where Pacman will begin to learn about the values of positions and actions. Once training is complete, he will enter testing mode during which he will play the game for a certain number of rounds and how well he does is measured by his win rate. To see Pacman in action after 2000 games of training, run:

```
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid 
``` 

<iframe width="540" height="315" class="center-image" src="http://www.youtube.com/embed/zmx-SJnsiQM" frameborder="0" allowfullscreen></iframe><b>

One thing to note though is that Pacman fails to win on larger layouts because each board configuration is a separate state with separate q-values. He has no way to generalize that running into a ghost is bad for all positions. Hence, this approach will not scale. 

The approach that is more scalable is approximate Q-learning. An approximate Q-learning agent learns weights of features of states where many states may share the same features. The agent assumes the existence of a feature function $f(s, a)$ over state and action pairs, which yields a vector $f_1(s, a) .. f_i(s, a) .. f_n(s, a)$ of feature values. 

The approximate Q-function takes the following form

$$Q(s, a) = \sum_{i=1}^n f_i(s, a) \cdot w_i$$

where each weight $w_i$ is associated with a particular feature $f_i(s, a)$. The weight vectors are updated the same one updates q-values:

$$difference = (r + \gamma \underset{a'}{\text{max}} Q(s', a')) - Q(s, a)$$
$$w_i = w_i + \alpha \cdot difference \cdot f_i(s, a)$$ 

## Conclusion

To summarize, a value iteration agent is not a reinforcement learning agent. It takes an MDP on construction and computes the best action according to the value function. That is, it ponders its MDP model to arrive at a complete policy before ever interacting with a real environment. A Q-learning agent, on the other hand, interacts with the environment it's in learns by trial and error and computes values when necessary. 

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