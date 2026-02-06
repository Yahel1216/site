---
title: "Approximating the Cut-Norm - Part 2"
date: 2026-01-12
slug: cut-norm-2
draft: false
katex: true
series: "Cut-Norm"
description: "Using Grothendick's identity to obtain a randomized approximate algorithm for computing the Cut-Norm of a matrix"
tags: ["optimization", "semi-definite-programming"]
categories: ["Optimization", "Complexity"]
---
> [!note] Prerequisites
> *   **Linear Algebra:** Eigenvalues, PSD matrices, Tensor products.
> *   **Convex Optimization:** Basic familiarity with Semidefinite Programming (SDP).
> *   **Probability:** Expectations, Markov's inequality.
> *   **Graph Theory:** Basic definitions, Cuts, Regularity.

We are interested in computing the cut norm of a matrix, defined by $$\\|A\\|\_C=\max_{I\subset R,J\subset S}\left|\sum_{i\in I,j\in J}a_{i,j}\right|$$
where $A=(a_{i,j})\_{i\in R,j\in S}$. We've seen this is a hard problem, and it is often equivalent to computing the $\infty\mapsto 1$ norm, defined by $$\\|A\\|\_{\infty\mapsto 1}=\max_{x\in \set{\pm 1}^R, y\in \set{\pm1}^S} \sum_{i\in R,j\in S} a_{i,j}\cdot x_i\cdot y_j$$
We've already seen that computing the latter norm can be done by solving an integer quadratic program, which has a relaxation to a quadratically constrained quadratic program given by $$\max \sum_{i,j}a_{i,j}\cdot \langle u_i, v_j\rangle \quad\text{subject to}\quad \\|u_i \\|^2 = \\|v_j\\|^2=1$$ where the optimization is over vectors $u_1,\ldots,u_n$ and $v_1,\ldots ,v_m$. We've seen that this problem can be solved using semi-definite programming, and we've seen one method to round the solution, giving an approximation factor of $\approx 0.03$. In this post, we'll see another method, which is much cleaner, and uses randomized rounding of this SDP.

## Randomized Improvement
This method is based on Grothendieck's inequality, which has turned out to be a fundamental result in computer science. Let us first recall some definitions:

> [!note] Reminder
> A Hilbert space is a vector space with an inner product, which is also **complete**. Every finite dimensional inner product space is complete. Completeness means that Cauchy sequences converge (if $\left\lVert x_{n}-x_{m} \right\rVert\to 0$ then the sequence $(x_n)$ has a limit point).

> [!note] Reminder
> Let $\mathcal{H}$ denote a Hilbert space and $u,v\in \mathcal{H}$, then $\left\langle u,v \right\rangle^{k}=\left\langle u^{\otimes k},v^{\otimes k} \right\rangle$ where $u^{\otimes k},v^{\otimes k}\in \mathcal{H}^{\otimes k}$ are elements in the tensor product space. For more background on tensor products see this [post](/posts/fmm-2) where the tensor product of vector spaces is discussed, and for the proof of the assertion made here see this [post](/posts/kernel-3/#as-tensor-products) where the polynomial kernel is discussed.

The main identity we need is:

> [!important] Grothendieck's Identity
> For every two vectors $u,v\in \mathcal{H}$ in a Hilbert space, if $z$ is chosen randomly and uniformly at random from the unit sphere of $\mathcal{H}$, then
> $$\frac{\pi}{2}\mathbb{E}_{z}[\mathrm{sign}(\left\langle u,z \right\rangle)\cdot \mathrm{sign}(\left\langle v,z \right\rangle)]=\arcsin(\left\langle u,v \right\rangle).$$

In other words, by randomly projecting $u$ and $v$ onto the same unit vector, we are able to estimate their inner product.

> [!important] Lemma
> Let $c=\sinh^{-1}(1)=\ln (1+\sqrt{2})$. For any set $\lbrace u_{i}\rbrace_{i=1}^{n}$ and $\lbrace v_{j}\rbrace_{j=1}^{m}$ of **unit vectors** in a Hilbert space $\mathcal{H}$, there is a set $\lbrace u_{i}'\rbrace_{i=1}^{n}$ and $\lbrace v_{j}'\rbrace_{j=1}^{m}$ of unit vectors in a Hilbert space $\mathcal{H}'$, such that for $z$ chosen randomly and uniformly from the unit sphere of $\mathcal{H}'$ it holds for every $i,j$:
> $$\frac{\pi}{2}\mathbb{E}\_{z}[\mathrm{sign}(\left\langle u_{i}',z \right\rangle)\cdot \mathrm{sign}(\left\langle v_{j}',z \right\rangle)]=c \cdot \left\langle u_{i},v_{j} \right\rangle.$$

**Proof.** By the identity, the expectation is just $\arcsin(\left\langle u_{i}',v_{j}' \right\rangle)$. So we need to choose $u_{i}',v_{j}'$ such that $\left\langle u_{i}',v_{j}' \right\rangle=\sin(c\cdot \left\langle u_{i},v_{j} \right\rangle)$. Using the Taylor expansion of the sine function,
$$\sin(c\cdot \left\langle u,v \right\rangle)=\sum_{k=0}^{\infty}(-1)^{k}\frac{c^{2k+1}}{(2k+1)!}\cdot(\left\langle u,v \right\rangle)^{2k+1}=\sum_{k=0}^{\infty}(-1)^{k}\frac{c^{2k+1}}{(2k+1)!}\cdot \left\langle u^{\otimes 2k+1},v^{\otimes2k+1} \right\rangle.$$
Define $\mathcal{H}'=\bigoplus_{k=0}^{\infty}\mathcal{H}^{\otimes 2k+1}$, and define $u'=\bigoplus_{k=0}^{\infty}(-1)^{k}\sqrt{\frac{c^{2k+1}}{(2k+1)!}}u^{\otimes2k+1}$  and $v'=\bigoplus_{k=0}^{\infty}\sqrt{\frac{c^{2k+1}}{(2k+1)!}}v^{\otimes2k+1}$. Then
$$\left\langle u',v' \right\rangle=\sin(c\cdot \left\langle u,v \right\rangle),$$
and also
$$\left\lVert u' \right\rVert^{2}=\left\langle u',u' \right\rangle=\sum_{k=0}^{\infty}(-1)^{2k}\frac{c^{2k+1}}{(2k+1)!}\left\lVert u^{\otimes2k+1} \right\rVert^{2}=\sum_{k=0}^{\infty}\frac{c^{2k+1}(\left\lVert u \right\rVert^{2})^{2k+1}}{(2k+1)!}=\sinh(c \cdot \left\lVert u \right\rVert^{2}).$$
Similarly, $\left\lVert v' \right\rVert^{2}=\sinh(c\cdot \left\lVert v \right\rVert^{2})$, and since we assumed $\left\lVert u \right\rVert=\left\lVert v \right\rVert=1$, we obtain that $\left\lVert u' \right\rVert=\left\lVert v' \right\rVert=\sinh(c)=\sinh(\sinh^{-1}(1))=1$, as required. $\blacksquare$

The existence is proved constructively but gives an infinite dimensional Hilbert space. However, we are essentially interested in a small sub-space of dimension at most $n+m$.
Thus, given $\lbrace u_{i}\rbrace_{i=1}^{n}$ and $\lbrace v_{j}\rbrace_{j=1}^{m}$ we can find $\lbrace u_{i}'\rbrace $ and $\lbrace v_{j}'\rbrace $ satisfying the above by solving the following SDP:
$$\max_{\alpha^{i},\beta^{j}\in\mathbb{R}^{n+m}}0\quad \text{subject to}\quad \left\langle \alpha^{i},\beta^{j} \right\rangle=\sin (c\cdot \left\langle u_{i},v_{j} \right\rangle),\quad \\|\alpha^i\\|^2 = \\|\beta^j\\|^2 = 1.$$
This is essentially a **feasibility** problem (which we just proved is feasible). Any solution is optimal.

> [!tip] Theorem
> There is a randomized algorithm achieving $\rho=\frac{2\ln(1+\sqrt{2})}{\pi}>0.56$ approximation in expectation for the problem of computing $\left\lVert A \right\rVert_{\infty\mapsto 1}$. The runtime of the algorithm is polynomial.

**Proof.** Following the discussion above, we can find the vectors $u_{i}',v_{j}'$ (at least a small representation of them) by solving the SDP written above. Picking uniformly at random a vector $z\in\mathbb{R}^{n+m}$ on the unit sphere, and outputting $x_{i}=\mathrm{sign}(\left\langle u_{i}',z \right\rangle)$ and $y_{j}=\mathrm{sign}(\left\langle v_{j}',z \right\rangle)$, we know by the previous lemma that
$$\frac{\pi}{2}\mathbb{E}\left[\sum_{i,j}a_{ij}x_{i}y_{j}\right]=\sum_{i,j}a_{i,j}\cdot c\cdot \left\langle u_{i},v_{j} \right\rangle,$$
and therefore on average we obtain a $\frac{2c}{\pi}$ approximation of $\left\lVert A \right\rVert_{\infty\mapsto 1}$. $\blacksquare$

## Obtaining an Approximation for the Cut-Norm

We're still to obtain an approximation for the **cut-norm.** Recall that we've shown above that when the rows and columns of the matrix $A$ sum to zero, it holds $\left\lVert A \right\rVert_{\infty\mapsto 1}=4\left\lVert A \right\rVert_{C}$. Given a matrix $A$ of size $n\times m$, we can define $A'$ to be a $(n+1)\times (m+1)$ matrix, by setting
$$a_{i,m+1}=-\sum_{k=1}^{m}a_{i,k}\quad,\quad a_{n+1,j}=-\sum_{k=1}^{n}a_{k,j}\quad, \quad a_{n+1,m+1}=0,$$
for every $i\in [n],j\in [m]$. This way, the rows and columns of $A'$ all sum to zero.

> [!important] Lemma
> The cut norms are equal $\left\lVert A \right\rVert_{C}=\left\lVert A' \right\rVert_{C}$.

**Proof.** We have a trivial upper bound $\left\lVert A \right\rVert_{C}\le \left\lVert A' \right\rVert_{C}$, because $A$ is a sub-matrix of $A'$. In the other direction, suppose $I'\subset [n+1]$, $J'\subset [m+1]$ are maximizers for $A'$. If $n+1\in I'$, define $I=[n+1]\setminus I'$ to be the complement, and otherwise set $I=I'$. Similarly, if $m+1\in J'$ define $J=[m+1]\setminus J'$ and otherwise $J=J'$. Note that $I\subset [n]$ and $J\subset [m]$. If $I\not=I'$, then
$$\sum_{i\in I,j\in J}a_{ij}=\sum_{i\in [n],j\in J}a_{ij}-\sum_{i\in [n]\setminus I,j\in J}a_{ij}=-\sum_{j\in J}a_{n+1,j}-\sum_{i\in [n]\setminus I,j\in J}a_{ij}=-\sum_{i\in I',j\in J}a_{ij},$$
by definition of $I$. Similarly, if $J\not=J'$, then
$$\sum_{i\in I',j\in J}a_{ij}=\sum_{i\in I',j\in [m]}a_{ij}-\sum_{i\in I',j\in [m]\setminus J}a_{ij}=-\sum_{i\in I'}a_{i,m+1}-\sum_{i\in I',j\in [m]\setminus J}a_{ij}=-\sum_{i\in I',j\in J'}a_{ij}.$$
Either way, we obtain that
$$\left|\sum_{i\in I,j\in J}a_{ij}\right|=\left|\sum_{i\in I',j\in J'}a_{ij}\right|,$$
and therefore $\left\lVert A \right\rVert_{C}\ge \left\lVert A' \right\rVert_{C}$. $\blacksquare$

Thus, given a matrix $A$, we construct the matrix $A'$ and compute a $\rho$-approximation for $\left\lVert A' \right\rVert_{\infty\mapsto 1}$, multiply by $4$, to obtain a $\rho$-approximation for $\left\lVert A' \right\rVert_{C}=\left\lVert A \right\rVert_{C}$.

## Proving the Identity

> [!important] Grothendieck's Identity
> For every two unit vectors $u,v\in \mathcal{H}$ in a Hilbert space, if $z$ is chosen randomly and uniformly at random from the unit sphere of $\mathcal{H}$, then
> $$\frac{\pi}{2}\mathbb{E}_{z}[\mathrm{sign}(\left\langle u,z \right\rangle)\cdot \mathrm{sign}(\left\langle v,z \right\rangle)]=\arcsin(\left\langle u,v \right\rangle).$$

**Proof.** Let $z$ denote some unit vector, then $\mathrm{sign}(\left\langle u,z \right\rangle)$ has the following geometric interpretation - it is $1$ when $u$ is inside the upper half of the sphere, assuming we orient it so that $z$ is the *north pole*. Therefore, $\mathrm{sign}(\left\langle u,z \right\rangle)\cdot \mathrm{sign}(\left\langle v,z \right\rangle)$ is $1$ only when $u,v$ lie on the same half of the sphere and $-1$ otherwise.
Thus the expectation becomes
$$\Pr(u,v \text{ lie on the same half})- \Pr(u,v \text{ lie on different halves}).$$
We know that $\left\langle u,v \right\rangle=\cos \theta$ where $\theta\in [0,\pi]$ is the angle between them. The probability that $u,v$ lie on the same half of the sphere boils down to how the equator (the orthogonal hyper-plane to $z$) splits up the **unique** dimensional circle on which $u,v$ both reside. In other words, we randomly choose a line (passing through the origin) that splits up the circle into two halves, and we ask whether $u,v$ are in the same half. Note that $u,v$ are not in the same half if and only if the line intersects the circle at a point which sits on the arc connecting $u,v$. That arc has length $\theta$ (because this is the unit circle), and this intersection point uniquely defines the line, so we choose a random point with uniform probability out of the total arc of length $\pi$. Thus the probability of $u,v$ not being in the same half is exactly $\theta/\pi$.
Hence
$$\mathbb{E}[\mathrm{sign}(\left\langle u,z \right\rangle)\cdot \mathrm{sign}(\left\langle v,z\right\rangle)]=\left(1-\frac{\theta}{\pi}\right)-\frac{\theta}{\pi}=1-\frac{2 \theta}{\pi},$$
and the identity follows from the trigonometric identity:
$$\arcsin(\left\langle u,v \right\rangle)=\arcsin(\cos \theta)=\arcsin\left( \sin\left(\frac{\pi}{2}-\theta\right)\right)=\frac{\pi}{2}-\theta=\frac{\pi}{2}\left(1-\frac{2\theta}{\pi}\right).$$ $\blacksquare$

## Conclusion
We have explored two methods for approximating the cut-norm of a matrix. The **deterministic algorithm** relied on a vector programming relaxation and a careful derandomization using orthogonal arrays, achieving a modest approximation ratio of $0.03$. The **randomized improvement** utilized the power of Grothendieck's Identity to lift the problem into an infinite-dimensional Hilbert space and then project it back, yielding a much stronger ratio of $\approx 0.56$.


## References
1.  **Alon, N., & Naor, A.** (2006). [*Approximating the Cut-Norm via Grothendieck's Inequality*](https://web.math.princeton.edu/~naor/homepage%20files/cutnorm.pdf).
2.  **Grothendieck, A.** (1953). [*Résumé de la théorie métrique des produits tensoriels topologiques*](https://www.numdam.org/item/AIF_1952__4__73_0/).
3.  **Håstad, J.** (2001). [*Some optimal inapproximability results*](https://dl.acm.org/doi/10.1145/502090.502098). Journal of the ACM.