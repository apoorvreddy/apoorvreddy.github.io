---
layout: post
title:  "Random Fourier Features for Kernel Machines"
date:   2019-08-24 03:00:26 +0530
categories: kernel methods
---

We will explore in some detail the 2007 NIPS paper, [Random Features for Large Scale Kernel Machines][RFF] by Ali Rahimi and Ben Recht in this post. I have been intrigued by this beautiful paper ever since it was awarded the NIPS test of time award in 2017. Personally, I have learnt many interesting mathematical tools and ideas from this paper. This post would be helpful to people trying to understand this paper, as I have tried to fill in many gaps in knowledge and jumps in logic and proofs.

## Dual Representation of Linear Models

A linear model $$ y = w^T \phi(x) $$ has an equivalent dual representation, $$ y = \sum_{i=1}^N \alpha_i k(x, x_i) $$. Here, $$\alpha_i$$ are scalars, while the function $$k(x, x_i)$$ is a kernel function which measures similarity between the two vectors $$x$$ and $$x_i$$, by computing the inner product $$<\phi(x), \phi(x_i)>$$ . Here $$x_i$$ are the observations from the training set.

Consider a linear regression model whose objective function $$J(w)$$ can be written as follows:

$$J(w) = \frac{1}{2} \sum_{i=1}^N \{w^T \phi(x_i) - y_i\}^2 + \frac{\lambda}{2} w^Tw \tag{1}$$

Setting the gradient $$\nabla_w J(w) = 0$$, we get

$$ w = -\frac{1}{\lambda} \sum_{i=1}^N \{w^T \phi(x_i) - y_i \}\phi(x_i) \tag{2}$$

Let 

$$\alpha_i = -\frac{1}{\lambda} \{ w^T \phi(x_i) - y_i\} \tag{3}$$

So, 

$$w = \sum_{i=1}^N \alpha_i \phi(x_i) = \Phi^T \alpha \tag{4}$$

Substituting for $$w$$, we get,

$$ y = w^T \phi(x) = \alpha^T \Phi \phi(x) = \sum_{i=1}^N \alpha_i \phi(x_i)^T \phi(x) = \sum_{i=1}^N \alpha_i k(x, x_i) \tag{5}$$

You can thus see that the linear model in $$w$$ can now be expressed as a linear combination of functions of the training observations. But what about $$\alpha_i$$ ? It is still a function of $$w$$ which we haven't got rid of yet.

Observe,

$$\alpha_1 = -\frac{1}{\lambda} \{\alpha^T \Phi \phi(x_1) - y_1\}$$ 

$$\dots$$

$$\alpha_N = -\frac{1}{\lambda} \{\alpha^T \Phi \phi(x_N) - y_N\}$$

So,

$$\alpha = -\frac{1}{\lambda} [ \Phi \Phi^T \alpha - Y] \tag{6}$$ 

Thus,

$$\alpha = (K + \lambda I_N)^{-1} Y \tag{7}$$

where $$K = \Phi \Phi^T$$ is the kernel matrix aka the Gram matrix. This is where the problem lies. As you can see, computing the kernel matrix and its inverse gets prohibitive in both memory and time beyond even a 10000 observations. The solution proposed by Rahimi and Recht, tries to solve this problem for a large number of observations.


## Fourier Transform of a Gaussian Kernel

$$\begin{eqnarray}
p(w) &=& \mathcal{F} (k(x, y)) \\
	 &=& \frac{1}{(2\pi)^d} \int_{R^d} e^{- \frac{||x-y||_2^2}{2}} e^{-j w^T (x-y)} dx dy \\
	 &=& \frac{1}{(2\pi)^d} \int_{R^d} e^{- \frac{||z||_2^2}{2}} e^{-j w^T z } dz \\
	 &=& (2\pi)^{-d/2}e^{-\frac{||w||_2^2}{2}} \tag{8}
\end{eqnarray}$$


Observe that $$p(w)$$ also happens to be a probability distribution of the form $$\mathcal{N}(0, I_d)$$.

<details>

Using the identities,

$$ \mathcal{F}(f'(x)) = jw\mathcal{F}(f(x)) $$

$$ \mathcal{F}(x f(x)) = j \mathcal{F}'(f(x)) $$

We have,

$$f(z) = e^{- \frac{||z||_2^2}{2}} $$

$$\begin{eqnarray}

\mathcal{F}(f'(z)) &=& jw \mathcal{F}(f(z)) \\
\mathcal{F}(-z f(z)) &=& jw \mathcal{F}(f(z)) \\
-j \mathcal{F}'(f(z)) &=& jw \mathcal{F}(f(z)) \\
\mathcal{F}'(f(z)) &=& - w \mathcal{F}(f(z)) \\
\frac{\mathcal{F}'(f(z))}{\mathcal{F}(f(z))} &=& -w \\
\int \frac{\mathcal{F}'(f(z))}{\mathcal{F}(f(z))} &=& - \int w dw \\
\ln(\mathcal{F}(f(z))) &=& - \frac{||w||_2^2}{2} + C \\
\mathcal{F}(f(z)) &=& e^{-\frac{||w||_2^2}{2}}e^C \\
\end{eqnarray}$$


Solving for $$C$$, by putting $$w=0$$,

$$p(0) = C = \int_{R^d} e^{-||z||^2/2} dz = (2\pi)^{d/2} $$

</details>


## Bochner's Theorem

Bochner's Theorem generalizes the above result and states that the Fourier Transform of a normalized, shift invariant kernel ($$k(x, y) = k(x-y)$$) is a probability distribution. Therefore, we can sample $$w$$ vectors from $$p(w)$$.

From the inverse Fourier transform,

$$\begin{eqnarray}
k(x-y) &=& \int_{R^d} p(w) e^{j w^T (x-y)} dw \\
       &=& \mathbb{E}_w [e^{j w^T (x-y)}] \\
       &=& \mathbb{E}_{w \sim p(w)} [\gamma_w(x) \gamma_w(y)^*] 
\end{eqnarray}
\tag{9}$$

where,

$$ \gamma_w(x) = e^{j w^T x} $$

and

$$ \gamma_w(-x) = \gamma_w(x)^* $$ 

is the complex conjugate.

Observe that both $$k$$ and $$p(w)$$ are real and even, therefore the complex part of the exponential can be dropped,

$$\begin{eqnarray}
k(x-y) &=& \mathbb{E}_{w \sim p(w)} [\gamma_w(x) \gamma_w(y)^*] \\
	   &=& \mathbb{E}_w [\cos(w^T x) \cos(w^T y) + \sin(w^T x) \sin(w^T y)] \tag{10}
\end{eqnarray}$$

Therefore,

$$\begin{eqnarray}
k(x-y) &\approx& \frac{1}{D} \sum_{i=1}^D \cos(w_i^T x) \cos(w_i^T y) +  \sin(w_i^T x) \sin(w_i^T y)\\
		&=& z_w(x)^T z_w(y) \tag{11}
\end{eqnarray}$$

where $$z_w(x) = \sqrt{\frac{1}{D}}[\cos(w_1^T x), \sin(w_1^T x), \dots, \cos(w_D^T x), \sin(w_D^T x)]^T $$

The RHS is thus an unbiased estimate of the kernel function. And the variance of the approximation can be decreased by increasing $$D$$.

This is the key insight from the paper. Now that the kernel can be approximated as a dot product between random features, we can thus fit linear models in the random feature subspace instead using the primal formulation.

## Hoeffding Bound

Observe,

$$ |z_{w_i} (x)| = | \frac{1}{\sqrt{D}}[\cos(w_i^T x), \sin(w_i^T x)]^T | <= \frac{1}{\sqrt{D}} \tag{12}$$

Hoeffding's bound provides an upper bound on the probability of deviation of a sum of independent variables $$X_i$$ from their mean.

$$ P(|\sum_{i=1}^N X_i - \mathbb{E}[X]| >= Nt) <= 2\exp(-\frac{2 N^2 t^2}{\sum_{i=1}^N (b_i - a_i)^2}) \tag{13}$$

where $$a_i$$ and $$b_i$$ are the lower and upper bounds on $$X_i$$.

Using the above identity,

$$\begin{eqnarray} P( | \sum_{i=1}^D {z_{w_i} (x) z_{w_i} (y)} - k(x,y)| >= \epsilon) &<=& 2\exp(-\frac{2\epsilon^2}{\sum_{i=1}^D (2\sqrt{1/D})^2}) \\
&<=& 2\exp(-\frac{D\epsilon^2}{2}) \tag{14}
\end{eqnarray}$$

The paper proves a slightly looser Hoeffding bound of $$ 2\exp(-\frac{D\epsilon^2}{4})$$ by taking the approximation $$ k(x-y) = \mathbb{E}_w [\cos(w^T x) \cos(w^T y)] $$


## Covering Number Argument

The paper goes on to prove a much stronger argument about the faithfulness of this approximation, essentially proving that it holds across all of the metric space. To understand the covering number argument and the proof, we will need to understand some mathematical vocabulary.

<h3> Metric Space </h3>

A 2-tuple $$(X, d_X)$$ where $$X$$ is a set and $$d_X$$ is a metric defined on this set. 

<h3> Diameter of a Metric Space $$(X, d_X)$$ </h3>

$$diam(\mathcal{M}) = \max_{x,y} \{ d_X(x, y) : x, y \in X\}$$

<h3> Spherical Balls in $$(X, d_X)$$ </h3>

$$B_r(x) = \{y \in X : ||y-x||_{d_X} <= r\} $$

In $$(\mathbb{R}^d, l_2)$$,

$$B_r(x) = \{y \in \mathbb{R}^d : ||y-x||_2 <= r\} $$


<h3> Covering Number </h3>

This is the minimum number of spherical balls of radius $$r$$ required to completely cover a given space $$ \mathcal{M} \subset \mathbb{R}^d$$. These balls can be centred in a subset $$ C \subset \mathbb{R}^d $$, where $$C$$ may or may not be a subset of $$\mathcal{M}$$.

$$N_r = \min_{|C|} : \cup_{x \in C} B_r(x) = \mathcal{M}$$

If $$C \subset \mathcal{M}$$, it is an internal covering, else an external covering.

An upper bound for the covering number of a metric space $$\mathcal{M}$$ has been proved [here][MFOL], which is 

$$N_r <= \bigg(\frac{2 diam(\mathcal{M})}{r} \bigg)^d \tag{15}$$

<h3> Covering and Packing Radius </h3>

Let, $$ C \subset \mathcal{M}$$. So, the covering radius $$r_c(\mathcal{M}, C)$$ is the smallest possible number such that $$\mathcal{M}$$ can be covered by spherical balls of radius $$r_c$$ centred in $$C$$.

$$r_c(\mathcal{M}, C) = \min_{r} : \cup_{x \in C} B_r(x) = \mathcal{M} $$

The packing radius is the maximum radius of spherical balls in $$\mathcal{M}$$, such that all balls are disjoint. Or in other words half of the smallest distance between any two points in $$\mathcal{M}$$

$$r_p(\mathcal{M}) = \max_r : \cap_{x \in \mathcal{M}} B_r(x) = \phi $$

<h3> Epsilon Net </h3>

An $$\epsilon$$-packing is a set $$X \subset \mathcal{M}$$ which has packing radius $$\geq \epsilon/2$$. And an $$\epsilon$$-covering is a set $$X \subset \mathcal{M}$$ which has a covering radius $$\leq \epsilon$$. An $$\epsilon$$-net is both $$\epsilon$$-packing and $$\epsilon$$-covering.

This means, the $$\epsilon$$-net is a set $$X \subset \mathcal{M}$$, s.t.

$$r_p(X) \geq \epsilon/2$$

and,

$$r_c(\mathcal{M}, X) \leq \epsilon$$

Therefore, as $$X \subset \mathcal{M}$$,

$$r_p(X) = \epsilon/2$$

$$r_c(\mathcal{M}, X) = \epsilon$$

<h3> Markov's Inequality </h3>

For a non-negative random variable $$X$$,

$$P(X > \epsilon) \leq \frac{\mathbb{E}(X)}{\epsilon}$$

<h3> Union Bound </h3>

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Therefore,

$$P(A \cup B) <= P(A) + P(B) $$

Generalizing,

$$P(\sum_{i=1}^N A_i)  <= \sum_{i=1}^N P(A_i)$$

<h3> Lispchitz constant of a function </h3>

The Lipschitz constant $$L_f$$ of a function $$f$$ is an indicator of the smoothness of the function. Smoother functions have lower values of $$L_f$$

For a function $$f : \mathbb{R}^d \rightarrow \mathbb{R}^k$$

$$L_f = \sup_{x, y} \frac{||f(x) - f(y)||_2}{||x - y||_2}$$

If f is differentiable,

$$L_f = \sup_{x} |f'(x)| $$

<h3> Covering Number Claim : Uniform Convergence of Fourier Features </h3>

\textbf{Claim}: Let $$\mathcal{M}$$ be a compact subset of $$\mathbb{R}^d$$ with diameter $$diam(\mathcal{M})$$. Then, for the mapping $$z$$, we have

$$P[\sup_{x, y \in \mathcal{M}} |z(x)^T z(y) - k(x,y) | \geq \epsilon] \leq 2^8 \bigg(\frac{\sigma_p diam(\mathcal{M})}{\epsilon}\bigg)^2 \exp(-\frac{D\epsilon^2}{4(d+2)})$$


where,
 $$\sigma_p^2 = \mathbb{E}[w^T w]$$ is the second moment of the Fourier transform of $$k$$.


Firstly, let's parse this claim.

On the LHS, we are computing the probability of the maximum approximation error for any two points $$x, y \in \mathcal{M}$$ to be greater than some $$\epsilon (>0)$$. On the RHS, we are claiming an upper bound on this probability as a function of the $$\epsilon$$ and $$D$$. The $$diam(\mathcal{M})$$ and $$\sigma_p$$ are constants. Thus the probability that the approximation error is large (for large $$\epsilon$$) is extremely small, and drops exponentially in $$D$$ and $$\epsilon^2$$ and holds throughout the space $$\mathcal{M}$$.

### Proof

The proof is interesting and beautiful to say the least. At an intuitive level, this is how the proof works.

- Define $$\Delta = x - y$$, where $$\Delta \in \mathcal{M}_\Delta$$
- Let $$f(x,y) = z(x)^T z(y) - k(x,y) = s(\Delta) - k(\Delta) $$ be a function from $$\mathcal{M_\Delta} \rightarrow \mathbb{R}$$
- Observe
	- $$ 0 \leq k(x,y) \leq 1 $$
	- $$ |s(\Delta)| \leq 1 $$
	- Therefore, $$ \vert f(\Delta) \vert \leq 2 $$
	- $$ \mathbb{E}[f(\Delta)] = 0 $$
	- $$ diam(\mathcal{M}_\Delta) = 2diam(\mathcal{M})$$, as the $$\Delta_\max = x_\max - (- x_\max) = 2 x_\max$$, where $$x_\max$$ is any vector on the surface of the enclosing sphere of $$\mathcal{M}$$
- Construct an $$\epsilon$$-net over $$\mathcal{M}_\Delta$$ which can cover $$\mathcal{M}_\Delta$$ using $$T$$-balls of radius $$r$$, with centres at $$\{\Delta_i\}_1^T$$. What this means, is that any $$\Delta \in \mathcal{M}_\Delta$$ is within $$r \leq \epsilon$$ distance from at least one of the anchor points of the $$\epsilon$$-net.
- Since $$f$$ is differentiable. From a 1st order Taylor expansion, $$f(\Delta) \approx f(\Delta_j) + f'(\Delta_j)(\Delta - \Delta_j)$$, where $$\Delta_j$$ is the closest anchor point in the $$\epsilon$$-net to $$\Delta$$.
	- Observe,
		- $$|f(\Delta)| \leq \beta \equiv |f(\Delta_j) + f'(\Delta_j)(\Delta - \Delta_j)| \leq \beta $$
		- $$|\Delta - \Delta_j| \leq r$$
	- And let,
		- $$ | f(\Delta_j) | \leq \gamma $$
		- $$L_f$$ be the Lipschitz constant of $$f$$.

	- Therefore, $$\gamma + L_f r \leq \beta$$ . Suitable values are $$\gamma \leq \frac{\beta}{2}$$ and $$L_f \leq \frac{\beta}{2r}$$

- Now, for $$\vert f(\Delta)\vert \leq \beta $$ to hold throughout $$\mathcal{M}_\Delta$$, we require all $$ \vert f(\Delta_i) \vert \leq \beta , i \in [1, T]$$.

- $$L_f = \| \nabla f(\Delta^*) \|$$, where $$\Delta^* = \arg \max_{\Delta \in \mathcal{M}_\Delta} \| \nabla f(\Delta) \|$$
	- $$\begin{eqnarray}
	\mathbb{E}[L_f^2] &=& \mathbb{E}[\|\nabla s(\Delta^*) - \nabla k(\Delta^*) \|^2] \\
					  &=& \mathbb{E}[\| \nabla s(\Delta^*)\|^2] + \mathbb{E}[\|\nabla k(\Delta^*)^2\|] - 2\mathbb{E}[\nabla k(\Delta^*)]\mathbb{E}[\nabla s(\Delta^*)] \\
					  &=& \mathbb{E}[\| \nabla s(\Delta^*)\|^2] - \mathbb{E}[\|\nabla k(\Delta^*)^2\|] \\
					  &\leq& \mathbb{E}[\| \nabla s(\Delta^*)\|^2] \\
					  &\leq& \mathbb{E}[\|w\|^2] \\
					  &=& \sigma_p^2
	  \end{eqnarray}$$

- So, 
	$$\begin{eqnarray}
	P(\vert f(\Delta) \vert \leq \beta) &=& P(\vert f(\Delta_1) \vert \leq \beta/2 \wedge \dots \wedge \vert f(\Delta_T) \vert \leq \beta/2 \wedge (L_f \leq \frac{\epsilon}{2r})) \\
	&=& 1 - P(\vert f(\Delta_1) \vert \geq \beta/2 \vee \dots \vee \vert f(\Delta_T) \vert \geq \beta/2 \vee (L_f \geq \frac{\epsilon}{2r})) \\
	&=& 1 - P(\cup_{i \in [1, T]} \vert f(\Delta_i) \vert \geq \beta/2) - P(L_f \geq \frac{\epsilon}{2r})) \\
	\end{eqnarray} \tag{16}$$

- From Union and Hoeffding Bounds,
	- $$\begin{eqnarray}
	P(\cup_{i \in [1, T]} \vert f(\Delta_i) \vert \geq \beta/2) &\leq& \sum_{i=1}^T P(\vert f(\Delta_i) \vert \geq \beta/2) \\
	&=& 2T\exp\bigg(-\frac{D\beta^2}{8}\bigg)
	  \end{eqnarray} \tag{17}$$

- So, from Markov's Inequality, $$P[L_f^2 \geq t] \leq \frac{\mathbb{E}[L_f^2]}{t}$$
	- Therefore, $$P[L_f \geq \frac{\beta}{2r}] \leq \bigg(\frac{2r \sigma_p}{\beta} \bigg)^2 \tag{18}$$

- Putting together 16, 17 and 18,
	- $$P(\vert f(\Delta) \vert \leq \beta) \geq 1 - 2T \exp\bigg(-\frac{D\beta^2}{8}\bigg) - \bigg(\frac{2r \sigma_p}{\beta} \bigg)^2$$

- A result from [functional analysis][MFOL], gives a bound for the covering number $$T \leq (2 diam(\mathcal{M}_\Delta) /r )^d = (4 diam(\mathcal{M})/r)^d$$.

- So, $$P(\vert f(\Delta) \vert \leq \beta) \geq 1 - 2 \bigg(\frac{4 diam(\mathcal{M})}{r} \bigg)^d \exp\bigg(-\frac{D\beta^2}{8}\bigg) - \bigg(\frac{2r \sigma_p}{\beta} \bigg)^2 $$

	- The right hand side is of the form $$1 - k_1 r^{-d} - k_2 r^2$$
	- The right hand side is maximized when $$r = (\frac{d}{2})^\frac{1}{d} (\frac{k_1}{k_2})^\frac{d}{d+2}$$
	- So $$P(\vert f(\Delta) \vert \leq \beta) \geq 1 - k_1^\frac{2}{d+2} k_2^\frac{d}{d+2} (c^{-d} + c^2 ) $$ where $$c = (\frac{d}{2})^\frac{1}{d}$$
	- $$c^{-d} + c^2 \leq 2$$ , Therefore, $$P(\vert f(\Delta) \vert \leq \beta) \geq 1 - 2 k_1^\frac{2}{d+2} k_2^\frac{d}{d+2} $$
- 
	- Assuming $$\bigg( \frac{diam(\mathcal{M}) \sigma_p}{\beta} \bigg) > 1$$

	- $$\begin{eqnarray}
		P(\vert f(\Delta) \vert \geq \beta) &\leq& 2 k_1^\frac{2}{d+2} k_2^\frac{d}{d+2} \\
											&\leq& 2^\frac{7d+4}{d+2} \bigg( \frac{diam(\mathcal{M}) \sigma_p}{\beta} \bigg)^\frac{2d}{d+2} \exp\bigg(- \frac{D \beta^2}{4(d+2)} \bigg) \\
											&\leq& 2^8 \bigg( \frac{diam(\mathcal{M}) \sigma_p}{\beta} \bigg)^2 \exp\bigg(- \frac{D \beta^2}{4(d+2)} \bigg)
	\end{eqnarray}$$

	which completes the proof.

### Implementation of Kernel Regression using Random Fourier Features

- Draw random vectors $$w_i$$ from the Fourier Transform of the kernel we're trying to approximate. See eqn 8 for an example.
	- $$w_i \sim p(w), i \in [1, D], w_i \in \mathbb{R}^d$$
- Project input $$x$$ onto random Fourier space, $$\phi(x) = \sqrt{\frac{1}{D}}[\cos(w_1^T x), \sin(w_1^T x), \dots, \cos(w_D^T x), \sin(w_D^T x)]^T $$
- Solve least squares for $$ y =  \beta^T \phi(x)$$
	- either using SGD or using the normal solutions method.


### References:

* [Random Features for Large Scale Kernel Machines][RFF]

* [Chapter 6 (Kernel Methods), Pattern Recognition and Machine Learning, Christopher Bishop][Bishop]

* [Proposition 5, On the Mathematical Foundations of Learning] [MFOL]

* [Metric Spaces][metricwiki]

* [Covering Numbers][covering]

* [Epsilon Nets] [epsilon]

[RFF]: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

[Bishop]: http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf

[MFOL]: https://www.ams.org/journals/bull/2002-39-01/S0273-0979-01-00923-5/S0273-0979-01-00923-5.pdf

[metricwiki]: https://en.wikipedia.org/wiki/Metric_space

[covering]: https://en.wikipedia.org/wiki/Covering_number

[epsilon]: https://en.wikipedia.org/wiki/Delone_set
