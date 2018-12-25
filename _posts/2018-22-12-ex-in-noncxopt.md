---
layout: post
title:  "Exercises in Non Convex Optimization"
date:   2018-12-22 03:00:26 +0530
categories: non-convex-optimization
---

These are my solutions of chapter 2 exercises of this wonderful [primer on non-convex optimization.][PrateekJainBook]

### Ex 2.1: Show that strong smoothness does not imply convexity by constructing a nonconvex function $$ f: R^p \rightarrow R $$ that is 1-SS

Let $$ f(x) = - \left\Vert x \right\Vert_2^2$$, which is non-convex.

So $$\nabla f(x) = -2x$$

$$ f(y) - f(x) - \langle \nabla f(x) , y - x \rangle = \left\Vert x \right\Vert_2^2  - \left\Vert y \right\Vert_2^2 - \langle \nabla f(x) , y - x \rangle $$

$$ = \left\Vert x \right\Vert_2^2  - \left\Vert y \right\Vert_2^2 - 2\left\Vert x \right\Vert_2^2 + 2\langle x , y \rangle $$

$$ = - \left\Vert x-y \right\Vert_2^2 <= 0 <= 1/2 \left\Vert x-y \right\Vert_2^2 $$

### Ex 2.2. Show that if a differentiable function f has bounded gradients i.e., $$ \left\Vert \nabla f(x) \right\Vert^2 \leq G $$ for all $$ x \in R^d $$ , then $$f$$ is Lipschitz. What is its Lipschitz constant?

From mean value theorem, there exists a point $$c$$ on the line between $$x$$ and $$y$$ such that $$\nabla f(c) = \frac {f(y) - f(x)}{y-x} $$

So, $$ \left\Vert \nabla f(c) \right\Vert^2 = \left\Vert\frac{f(y) - f(x)}{y-x}\right\Vert_2^2$$

So, $$ \left\Vert f(y) - f(x) \right\Vert \leq \sqrt{G} \left\Vert y-x \right\Vert_2 $$

Lipschitz constant is $$ \sqrt{G} $$


### Ex 2.3. Show that for any point $$ z \not\in B_2(r) $$, the projection onto the ball is given by $$\Pi_{B_2(r)}(z) = r.\frac{z}{|z|} $$

Geometrically, the closest point in the set $$ B_2(r) $$ to a point outside it will be on the surface of the set and in the same direction. So, direction of the unit vector is given by $$ \hat{e} = \frac{z}{\vert z \rvert} $$, and magnitude is $$ r $$

So, the projection is $$ \Pi_{B_2(r)}(z) = r\hat{e} = r\frac{z}{\lvert z \rvert} $$


### Proof by Contradiction:

Assume there's a point $$ \hat{z} \neq r\frac{z}{\lvert z \rvert} $$ and is the projection of $$ z $$ on to the L2-ball.

From Projection lemma, which states: For any set (convex or not) $$ C \subset R^p $$ and $$ z \in R^p $$ , let $$ \hat{z} = \Pi_C(z) $$ . Then for all $$ x \in C $$, $$ \Vert\hat {z} − z\Vert_2 \leq \Vert x − z\Vert $$.

So, $$ \Vert\hat{z} - z\Vert_2 \leq \Vert r\frac{z}{\lvert z \rvert} - z\Vert_2 $$

So, $$ \Vert\hat{z} - z\Vert_2 \leq \Vert z\Vert_2 \Vert \frac{r}{\lvert z \rvert} - 1\Vert_2 $$

And, $$ r \lt \lvert z \rvert $$. This is a contradiction


### Ex 2.4. Show that a horizon-oblivious setting of $$\eta_t = \frac{1}{\sqrt{t}} $$ while executing the Projected Gradient Descent algorithm with a convex function with bounded gradients also ensures convergence.


Define the potential function $$ \Phi_t = f(x^t) - f(x^\star) $$, where $$x^\star$$ is the minima. From convexity of $$f$$, we can upperbound $$\Phi_t$$.


\begin{align} 
\Phi_t &= f(x^t) - f(x^\star) \\\\ 
&<= \langle \nabla f(x), x^t - x^\star \rangle \\\\ &<= \frac{1}{2\eta_t} \left( \Vert x^t - x^\star \Vert_2^2 + \eta_t^2G^2 - \Vert z^{t+1} - x^\star \Vert_2^2 \right)
\end{align}

where $$ z^{t+1} $$ is the projection of the update on $$x^t$$ onto the convex constraint set.

We also know from the Projection Lemma, $$ \Vert z^{t+1} - x^\star \Vert_2^2 >= \Vert x^{t+1} - x^\star \Vert_2^2 $$

So, $$ \Phi_t <= \frac{1}{2\eta_t} \left(\Vert x^{t} - x^\star \Vert_2^2 - \Vert x^{t+1} - x^\star \Vert_2^2 \right) + \frac{\eta_t G^2}{2}$$

Refer Pg. 21 of [this book][PrateekJainBook].

The above form doesn't allow for easy telescoping operation, so we will have to make some necessary assumptions for this. One such assumption can be on the diameter of the convex constraint set $$diam(X) <= D$$.

\begin{align}
\frac{1}{T}\sum_t \Phi_t \\\\
&<= \frac{1}{2T}(\Vert x^\star \Vert_2^2 + \sum_{t=2}^{T} ( \Vert x^t - x^\star \Vert_2^2(\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}}) + \eta_t G^2 ) \\\\ &<=  \frac{1}{2T}(\Vert x^\star \Vert_2^2 + \sum_{t=2}^{T} ( D^2(\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}})) + \sum_{t=1}^T \eta_t G^2 )) \\\\ &<= \frac{1}{2T}(\Vert x^\star \Vert_2^2 + D^2(\frac{1}{\eta_T} - \frac{1}{\eta_{1}})) + \sum_{t=1}^T\eta_t G^2 )) \\\\ &<= \frac{1}{2T}(\Vert x^\star \Vert_2^2 + D^2(\sqrt{T} - 1)) + \sqrt{T}G^2 )) \\\\ &<= \frac{1}{2\sqrt{T}}(\frac{\Vert x^\star \Vert_2^2}{\sqrt{T}} + D^2 + G^2 )) 
\end{align}

(Using $$x^1 = 0$$, $$ \eta_1 = 1$$, $$\eta_T = \frac{1}{\sqrt{T}} $$, and $$\sum_{i=1}^{n}\frac{1}{\sqrt{i}} <= 2sqrt(n) - 1 $$ )


### Ex 2.5. Show that if $$ f : R^p \rightarrow R $$ is a strongly convex function that is differentiable, then there is a unique point $$ x^* \in R^p $$ that minimizes the function value $$f$$ i.e., $$ f (x^*)$$ = min$$_{x \in R^p} f(x)$$

Proof by Contradiction:
Let $$x^ *$$ and $$y^*$$ be two points that minimize $$f$$, i.e. $$ f(x^*) = f(y^*) = f^* $$
From strong convexity, $$ f(\theta x^* + (1-\theta)y^*) < \theta f(x^*) + (1-\theta)f(y^*) $$
$$ = f* $$.
This is a contradiction.

### Ex 2.6. Show that the set of sparse vectors $$B_0(s) \subset R^p $$ is non-convex for any $$s < p$$. What happens when $$s = p$$ ?

Take $$ x_1 $$ s.t. $$ \Vert x_1 \Vert_0 = s$$ And the $$s$$ non zero elements are the first $$s$$ elements. And take $$ x_2 $$ s.t. $$ \Vert x_2 \Vert_0 = 1 $$, with the last element as non-zero.

A convex sum of $$ x_1 $$ and $$ x_2 $$ is not in $$ B_0(s) \subset R^p $$, if $$ s < p $$. Thus $$ B_0(s) $$ is non-convex. $$B_0(p) $$ however is convex.

###  Exercise 2.7. Show that $$ B_{rank(r)} \subseteq R_{n×n}$$, the set of $$n × n$$ matrices with rank at most $$r$$, is non-convex for any $$r < n$$. What happens when $$r = n$$ 

The sum of two singular matrices can be non-singular. So $$ B_{rank(r)} \subset R_{n×n} $$ is non-convex, for $$ r< n$$ and convex for $$ r = n$$.



[PrateekJainBook]: http://www.prateekjain.org/publications/all_papers/JainK17_FTML.pdf