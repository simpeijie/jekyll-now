---
layout: post
title: Pacman-Multiagent
---
In this project, I design agents (Pacman and ghosts) for the classic version of Pacman as well as implement both minimax with alpha-beta pruning and expectimax search. The point of this project is to get Pacman to reach his goals in scenarios where there are more than one adversary. The code is available [here](https://github.com/simpeijie/CS188-Artificial-Intelligence/tree/master/multiagent).

## Background

### Zero-sum Game

A zero-sum game is a type of game in which actions are deterministic and an agent's gain is directly equivalent to its opponent's loss and vice versa. For more info, visit [here](https://en.wikipedia.org/wiki/Zero-sum_game).

### Minimax 

The first game we will look at is minimax, which runs under the motivating assumption that the opponent we face behaves optimally, and will always perform the move that is worst for us.

We define the value of a state as the optimal score attainable by the agent which controls the state. We represent the value with the function $V(s)$ where $s$ is the state an agent is in and the function is defined as follows (in the context of Pacman with adversaries):

$$\forall \text{agent-controlled states, } V(s) = \underset{s' \in successors(s)}{\text{max}} V(s')$$

$$\forall \text{opponent-controlled states, } V(s) = \underset{s' \in successors(s)}{\text{max}} V(s')$$

$$\forall \text{terminal states, } V(s) = \text{known}$$

That is, at each state that Pacman controls, he will seek for the maximum permissible value by the ghosts, whose purpose is to minimize the values Pacman can achieve. 

### Alpha-beta Pruning

Minimax is simple and optimal, but it is inefficient. Its execution is similar to that of depth-first search with a time complexity of $O(b^m)$, where $b$ is the branching factor and $m$ is the tree depth. An optimization that can be done is alpha-beta pruning. 

The basic idea of alpha-beta pruning is best illustrated with an example. 

<img src="{{ site.baseurl }}/images/ab-pruning.png" width="600" class="center-image">

*Note: Upward and downward-pointing triangles are the maximizers and minimizers, respectively.*

Here, we start by iterating through nodes with values 3, 12 and 8, assigning the value min(3, 12, 8) = 3 to the leftmost minimizer. Likewise, both the middle minimizer and rightmost minimizer are assigned 2. Finally, the root maximizer will get max(3, 2, 2) = 3. 

However, as soon as we visit the child of the middle minimizer with value 2, we no longer need to look at the middle minimizer's other children. Since we've encountered a child of the middle minimizer with a value of 2, we know that no matter what values the other children hold, the value of the middle minimizer can be at most 2. Therefore, we can prune the search tree and do not have to examine its remaining children.

The advantage of alpha-beta pruning is the reduction in time complexity of minimax to $O(b^{m/2})$, effectively doubling the "solvable" depth.

### Expectimax

Minimax allows us to respond optimally to optimal opponents and because of this, it has some natural constraints on the situations to which it can respond. It is often overly pessimistic in situations where optimal responses of an agent's actions are not guaranteed. 

Expectimax is similar to minimax but instead of considering the worst case scenario, we consider the average case. While minimizers simply computee the minimum utility over their children, chance nodes compute the expected utility or expected value. The rule for determining values of nodes with expectimax is as follows:

$$\forall \text{agent-controlled states, } V(s) = \underset{s' \in successors(s)}{\text{max}} V(s')$$

$$\forall \text{chance states, } V(s) = \underset{s' \in successors(s)}{\sum} p(s')V(s')$$

$$\forall \text{terminal states, } V(s) = \text{known}$$

$p(s')$ refers to the probability that a given probabilistic action results in state $s'$, or the probability that an opponent chooses an action that results in $s'$.

## Implementation

When Pacman believes that his death is unavoidable, he will try to end the game as soon as possible because of the constant penalty for living. Sometimes, this is the wrong thing to do with random ghosts, but minimax agents always assume the worst:

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
