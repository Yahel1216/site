---
title: "The Nystrom Method: Spectral Action"
date: 2026-01-20
slug: kernel-4
draft: false
katex: true
description: "A method for estimating the Gram matrix by solving a small eigen-decomposition problem and using some analysis"
series: "Kernel Methods"
tags: ["optimization", "machine-learning", "linear-algebra", "analysis"]
categories: ["Machine Learning", "Kernels"]
---
> [!abstract] Prerequisites
> *   **Linear Algebra:** Eigen-decompositions, positive definite matrices, rank.
> *   **Functional Analysis:** Hilbert spaces, $L^2$ spaces, operators.
> *   **Kernel Methods:** Previous posts in the series.

Continuing our series on kernel methods, recall that these methods allow us to operate in high-dimensional spaces using the "kernel trick". However, they suffer from a major computational bottleneck: constructing and manipulating the Gram matrix requires $O(N^2)$ memory and $O(N^3)$ time for operations like inversion or eigen decomposition, where $N$ is the dataset size. When $N$ reaches hundreds of thousands, exact computation becomes infeasible. In this post, I will explore the **Nystrom method**, a powerful technique for constructing low-rank approximations of these matrices. Rather than treating it merely as a linear algebra heuristic, I want to derive it from first principles: starting with the spectral properties of integral operators on Hilbert spaces and showing how the discretization of these operators naturally leads to the matrix approximation formulas we use in practice.

Let $K:X\times X\to \mathbb{R}$ denote a symmetric positive definite kernel, with $X=\mathbb{R}^{d}$. Recall that by the Moore-Aronszajn theorem, there exists a feature map $\varphi:X\to \mathcal{H}$, where $\mathcal{H}$ is a Hilbert space, such that:
$$K(x,y)=\left\langle \varphi(x),\varphi(y) \right\rangle_{\mathcal{H}}.$$

## Some Analysis

Recall a remark I've written in a previous post, saying that a **kernel** can be used to define an **integral transform**. Let $L^{2}(X)$ denote the space of functions $f:X\to \mathbb{C}$ which are square integrable, i.e.,
$$\int_{X}^{}{\left|f(x)\right|^{2}}\ \mathrm{d}{x}<\infty.$$
This is a **Hilbert space**, with the inner product
$$\left\langle f,g \right\rangle=\int_{X}^{}{f(x)\cdot \overline{g(x)}}\ \mathrm{d}{x}.$$
Then $K$ can be used to define an operator $T_{K}:L^{2}(X)\to L^{2}(X)$, taking in functions and spitting out new functions (like the Fourier transform), by letting:
$$\[T_{K}(f)\] (x)=\int_{X}^{}{K(x,y)\cdot f(y)}\ \mathrm{d}{y}.$$
By linearity of the integral, it is clear that $T_{K}$ is **linear**.

### Properties of the Integral Transform

> [!important] Lemma
> A linear operator is continuous if and only if the operator norm, defined by
> $$\left\lVert T \right\rVert:=\sup_{f\in L^{2}(X),\left\lVert f \right\rVert=1}\left\lVert T(f) \right\rVert$$
> is finite (**bounded**). Here $\left\lVert f \right\rVert=\left(\int_{X}^{}{\left|f(x)\right|^{2}}\ \mathrm{d}{x}\right)^{1/2}$ is the norm induced from the inner product.

**Proof.**
$f_{n}\to f$ in $L^{2}(X)$ if and only if $\left\lVert f_{n}-f \right\rVert\to 0$. Hence:
$$\left\lVert T(f_{n})-T(f) \right\rVert= \left\lVert T(f_{n}-f) \right\rVert\le \left\lVert T \right\rVert\cdot \left\lVert f_{n}-f \right\rVert\to 0,$$
implying $Tf_{n}\to Tf$ when $T$ is bounded. Conversely, if $T$ is continuous, it is continuous at $0$. Hence for every $\varepsilon>0$ there is $\delta>0$ such that every $\left\lVert f \right\rVert\le \delta$ satisfies $\left\lVert Tf \right\rVert\le \varepsilon$. Consequently, setting $\varepsilon=1$, let $\delta>0$ be the respective value. For every $\left\lVert f \right\rVert= 1$ we have $\left\lVert \delta f \right\rVert=\delta\le \delta$ and so $\delta \left\lVert Tf \right\rVert= \left\lVert T(\delta f) \right\rVert\le 1$, which implies $\left\lVert Tf \right\rVert\le 1/\delta$. Since $\delta$ is constant, $\left\lVert T \right\rVert$ is bounded. $\blacksquare$

> [!important] Lemma
> If
> $$\left\lVert K \right\rVert_{L^{2}(X\times X)}^{2}=\int_{X \times X}^{}{\left|K(x,y)\right|^{2}}\ \mathrm{d}{(x,y)}<\infty$$
> then $T_{K}$ is **continuous**.

**Proof.**
By Cauchy-Schwarz inequality,
$$|T_{K}f(x)|=\left|\left\langle K(x,\cdot ), \overline{f} \right\rangle\right|\le \left\lVert K(x,\cdot) \right\rVert\cdot \left\lVert f \right\rVert.$$
Therefore,
$$\left\lVert T_{K}f \right\rVert^{2}=\int_{X}^{}{\left|T_{K}f(x)\right|^{2}}\ \mathrm{d}{x}\le \int_{X}^{}{\left\lVert K(x,\cdot ) \right\rVert^{2}\cdot \left\lVert f \right\rVert^{2}}\ \mathrm{d}{x}=\left\lVert f \right\rVert^{2}\cdot \int_{X}^{}{\int_{X}^{}{\left|K(x,y)\right|^{2}}\ \mathrm{d}{y}}\ \mathrm{d}{x},$$
where we can take $\left\lVert f \right\rVert^{2}$ out because it is a scalar (independent of $x$). The last integral is just $\int_{X\times X}^{}{\left|K(x,y)\right|^{2}}\ \mathrm{d}{(x,y)}$ by Fubini's theorem, and so
$$\left\lVert T_{K}f \right\rVert^{2}\le \left\lVert f \right\rVert^{2}\cdot \left\lVert K \right\rVert_{L^{2}(X\times X)}^{2}\implies \left\lVert T_{K} \right\rVert\le \left\lVert K \right\rVert_{L^{2}(X\times X)},$$
and in particular it is bounded, thus continuous. $\blacksquare$

> [!caution] Definition
> A linear operator is **self-adjoint** if $T=T^{\*}$, where $T^{\*}$ is the **unique** linear operator satisfying
> $$\left\langle Tf,g \right\rangle=\left\langle f,T^{\*}g \right\rangle$$
> for every $f,g\in L^{2}(X)$.

> [!important] Lemma
> If $K$ is **real-valued** then $T_{K}$ is **self-adjoint**.

**Proof.**
It holds that
$$\begin{aligned}
\left\langle T_{K}f,g \right\rangle & =\int_{X}^{}{T_{K}f(x) \overline{g(x)}}\ \mathrm{d}{x}=\int_{X}^{}{\int_{X}^{}{K(x,y)\cdot f(y)\cdot \overline{g(x)}}\ \mathrm{d}{y}}\ \mathrm{d}{x} 
\\\\ &=\int_{X}^{}{\int_{X}^{}{f(y)\cdot \overline{K(y,x)\cdot g(x)}}\ \mathrm{d}{x}}\ \mathrm{d}{y}=\int_{X}^{}{f(y)\cdot \overline{T_{K}g(y)}}\ \mathrm{d}{y}= \left\langle f,T_{K}g \right\rangle
\end{aligned}$$
using Fubini's theorem and the fact $\overline{r}=r$ for $r\in\mathbb{R}$ (and symmetry $K(x,y)=K(y,x)$). $\blacksquare$

> [!caution] Definition
> A linear operator is called **compact** if for every bounded sequence $(f_{n})\_{n=1}^{\infty}$, the sequence $(Tf_{n})\_{n=1}^{\infty}$ has a **convergent** sub-sequence.
>
> Equivalently, $T$ is compact if and only if it is the **limit** (in operator norm) of a sequence $T_{n}$ of operators for which $\mathrm{Im}(T_{n})$ is a finite dimensional subspace.

> [!important] Lemma
> If $\left\lVert K \right\rVert_{L^{2}(X\times X)}<\infty$ then $T_{K}$ is **compact**.

**Proof.**
Fix some orthonormal basis for $L^{2}(X)$, denoted $e_{1},e_{2},\ldots$. Take $P_{n}$ to be the projection operator onto $\mathrm{Span}\{e_{1},\ldots,e_{n}\}$. Define $T_{n}=T_{K}\circ P_{n}$ (composition). Note that the image of $T_{n}$ is the image of $T_{K}$ on a finite dimensional space, hence it is also finite dimensional. Let $\left\lVert f \right\rVert=1$ and expand it with the orthonormal basis $f=\sum_{i=1}^{\infty}\alpha_{i}e_{i}$ (where $\alpha_{i}\in\mathbb{C}$). Note that $\sum |\alpha_i|^2 \le 1$. Then:
$$\begin{aligned}
\left\lVert (T_{K}-T_{n})f \right\rVert^{2}&=\left\lVert \sum_{i=n+1}^{\infty}\alpha_{i}T_{K}e_{i} \right\rVert^{2}\le \left(\sum_{i=n+1}^{\infty}|\alpha_{i}| \left\lVert T_{K}e_{i} \right\rVert\right)^{2}
\\\\ &\le \left(\sum_{i=n+1}^\infty |\alpha_i|^2\right)\left(\sum_{i=n+1}^\infty \|T_K e_i\|^2\right) \le \sum_{i=n+1}^{\infty}\left\lVert T_{K}e_{i} \right\rVert^{2}
\end{aligned}$$
We are left showing
$$\sum_{i=1}^{\infty}\left\lVert T_{K}e_{i} \right\rVert^{2}<\infty,$$
which would imply the tail $\sum_{i=n+1}^{\infty}\left\lVert T_{K}e_{i} \right\rVert^{2}\to 0$ vanishes, hence $\left\lVert T_{K}-T_{n} \right\rVert\to 0$, and so $T_{K}$ is compact. By Parseval's identity, we know that for every fixed $x$, the function $K(x,\cdot)$ satisfies
$$\int_{X}^{}{\left|K(x,y)\right|^{2}}\ \mathrm{d}{y}=\left\lVert K(x,\cdot) \right\rVert^{2}=\sum_{i=1}^{\infty}|\left\langle K(x,\cdot),e_{i} \right\rangle|^{2}=\sum_{i=1}^{\infty}\left|\int_{X}^{}{K(x,y)e_{i}(y)}\ \mathrm{d}{y}\right|^{2},$$
and therefore
$$\begin{aligned}
\sum_{i=1}^{\infty}\left\lVert T_{K}e_{i} \right\rVert^{2}&=\sum_{i=1}^{\infty}\int_{X}^{}{\left|\int_{X}^{}{K(x,y)e_{i}(y)}\ \mathrm{d}{y}\right|^{2}}\ \mathrm{d}{x}=\int_{X}^{}{\sum_{i=1}^{\infty}\left|\int_{X}^{}{K(x,y)e_{i}}\ \mathrm{d}{y}\right|^{2}}\ \mathrm{d}{x}
\\\\ &=\int_{X}^{}{\int_{X}^{}{\left|K(x,y)\right|^{2}}\ \mathrm{d}{y}}\ \mathrm{d}{x}=\left\lVert K \right\rVert_{L^{2}(X\times X)}^{2}< \infty\end{aligned}$$
$\blacksquare$


### Using the Spectral Theorem

The following theorem is a fundamental result in spectral analysis:

> [!tip] Theorem: The Spectral Theorem
> If $T:\mathcal{H}\to \mathcal{H}$ is a self-adjoint, compact, and continuous linear operator on a (separable) Hilbert space $\mathcal{H}$, there exists an orthonormal basis for $\mathcal{H}$ consisting of eigenfunctions of $T$, denoted by $(\psi_{n})\_{n=1}^{\infty}$, corresponding to eigenvalues $(\lambda_{n})\_{n=1}^{\infty}$, where the sequence of eigenvalues is **monotone decreasing** and vanishes at infinity.

This is a generalization of the spectral theorem for self-adjoint matrices in finite-dimensions: a linear operator $T$ on a finite dimensional space can be represented by a matrix $M$, and the operator is self-adjoint iff the matrix is self-adjoint, $\overline{M^{\top}}=M$. On finite dimensions, linear operators are always compact and continuous, and so we obtain the existence of a sequence $(\lambda_{n})\_{n=1}^{d}$ (where $d$ is the dimension) and orthonormal vectors $(v_{n})\_{n=1}^{d}$ such that $Mv_{n}=\lambda v_{n}$, concluding that $M=V \Lambda \overline{V^{\top}}$ with $V=(v_{1},\ldots,v_{d})$ and $\Lambda =\mathrm{diag}(\lambda_{1},\ldots,\lambda_{d})$.

In particular, for $T_{K}$ we conclude that:

> [!important] Corollary (Mercer's Theorem)
> The operator $T_{K}$ has eigenvalues $(\lambda_{n})$ and eigenfunctions $(\psi_{n})$, which form an orthonormal basis. It holds
> $$K(x,y)=\sum_{n}\lambda_{n}\psi_{n}(x)\psi_{n}(y).$$

**Proof Sketch.**
Using the expansion in the eigenfunction basis, we write $f=\sum_{n}\left\langle f,\psi_{n} \right\rangle \cdot \psi_{n}$. Hence:
$$T_{K}f(x)=\left(T_{K}\left[\sum_{n}\left\langle f,\psi_{n} \right\rangle \cdot \psi_{n}\right]\right)(x)=\sum_{n} \lambda_{n} \left\langle f,\psi_{n} \right\rangle\cdot \psi_{n}(x).$$
Expanding the inner product and switching the sum and integration:
$$\sum_{n}\lambda_{n}\left\langle f,\psi_{n} \right\rangle\cdot \psi_{n}(x)=\sum_{n} \lambda_{n}\psi_{n}(x)\cdot \int_{X}^{}{f(y) \cdot \overline{\psi_{n}(y)}}\ \mathrm{d}{y}=\int_{X}^{}{\left(\sum_{n}\lambda_{n}\psi_{n}(x) \overline{\psi_{n}(y)}\right) f(y)}\ \mathrm{d}{y}.$$
Write $\widetilde{K}(x,y)=\sum_{n}\lambda_{n}\psi_{n}(x)\overline{\psi_{n}(y)}$, and note that $T_{K}=T_{\widetilde{K}}$ by the above identity. This implies $K=\widetilde{K}$, otherwise we could have found a function $f$ for which the integrals would have been different. Since $K$ is real valued, the complex conjugation can be removed. $\blacksquare$

### Dealing With Other Measures

The values in the dataset, $x_{1},\ldots,x_{N}$ come from some probability distribution, representing the data. In the discussion above, we implicitly used the **Lebesgue measure** which is not a probability distribution. However, all the arguments carry over to the case where we switch the definition of integration to be against some probability measure $\mu$. So the $L^{2}(X)$ space has the inner product
$$\left\langle f,g \right\rangle=\int_{X}^{}{f(x)\overline{g(x)}}\ \mathrm{d}{\mu(x)},$$
and $T_{K}$ is given by
$$T_{K}f(x)=\int_{X}^{}{K(x,y)f(y)}\ \mathrm{d}{\mu(x)}.$$
The orthonormal basis is then orthonormal with respect to this measure.

In fact, proving $T_K$ is compact becomes much more easier when the probability measure is supported on a **compact subset** of $\mathbb{R}^d$. All the proofs we gave above can be simplified to conclude that $T_K$ is compact, continuous and self-adjoint. It is crucial to take the measure into account. For example, the Gaussian kernel (radial basis function) we discussed before does not induce a compact operator on $\mathbb{R}^d$ with the standard measure, or even a non-compactly supported measure.

Compactly supported measures are very often in practice, simply because we cannot really sample from an unbounded distribution, due to machine restrictions.

## Approximating the Kernel

Recall that our goal is to compute a fast approximation to $K(x_{i},x_{j})$ over a dataset of points $x_{1},\ldots,x_{N}\sim \mu$. The previous discussion suggests that we should approximate
$$\lambda_{n}\psi_{n}(x_{i})\psi_{n}(x_{j})$$
for the **largest** eigenvalues (recall $\lambda_{1}\ge \lambda_{2}\ge \ldots$), as they are likely to contribute the most the sum. Moreover, computing $\lambda_{n}\psi_{n}(y)$ is the same as computing
$$\lambda_{n}\psi_{n}(y)=T_{K}\psi_{n}(y)=\int_{X}^{}{K(y,x) \psi_{n}(x)}\ \mathrm{d}{\mu(x)}.$$
By randomly sampling $\widehat{x}\_{1},\ldots,\widehat{x}\_{q}$ from $X$ (according to the probability measure $\mu$), we can approximate the integral using numerical integration:
$$\frac{1}{q}\sum_{i=1}^{q}K(y,\widehat{x}\_{i})\cdot \psi_{n}(\widehat{x}\_{i})\approx \int_{X}^{}{K(y,x)\psi_{n}(x)}\ \mathrm{d}{x}=\lambda_{n}\psi_{n}(y).$$
The orthonormality of $\psi_{n}$ and $\psi_{m}$ can be empirically checked by
$$\frac{1}{q}\sum_{i=1}^{q}\psi_{n}(\widehat{x}\_{i})\psi_{m}(\widehat{x}\_{i})\approx \int_{X}^{}{\psi_{n}(x)\psi_{m}(x)}\ \mathrm{d}{x}=\delta_{n,m}.$$
Note that numerical integration does not require explicit dealing with the probability measure $\mu$, because it is implicit in the sampling process.

**Remark.** Numerical integration works in expectation even for $q=1$, and taking $q$ to be large enough we can use concentration inequalities that guarantee a very good approximation with high probability, as we did in the post about random Fourier Features. Although we'll not quantify this here, it can be shown to be a very accurate approximation with high probability, uniformly on compact domains.

### As a Matrix Eigenproblem

Define
$$K^{(q)}\in \mathbb{R}^{q \times q}:\quad (K^{(q)})\_{i,j}=K(\widehat{x}\_{i},\widehat{x}\_{j})$$
and also
$$U^{(q)}\in\mathbb{R}^{q \times q}:\quad (U^{(q)})\_{i,j}=\frac{1}{\sqrt{q}}\psi_{j}(\widehat{x}\_{i}).$$
Furthermore, let $\Lambda^{(q)}$ be the diagonal $q\times q$ matrix with values $\lambda_{1}^{(q)}\ge \ldots \ge \lambda^{(q)}\_{q}\ge 0$, given by $\lambda_{i}^{(q)}=q\cdot \lambda_{i}$.

Note that this definition satisfies the equation
$$K^{(q)}U^{(q)}\approx U^{(q)}\Lambda^{(q)},$$
as the $i,j$-th element on the left is just the numerical integration expression
$$\frac{1}{\sqrt{q}}\sum_{t=1}^{q}K(\widehat{x}\_{i},\widehat{x}\_{t})\cdot \psi_{j}(\widehat{x}\_{t})\approx \sqrt{q} \lambda_{j} \psi_{j}(\widehat{x}\_{i}),$$
while the $i,j$-th element on the right is the approximated quantity
$$\frac{1}{\sqrt{q}}\cdot q \cdot \lambda_{j} \cdot \psi_{j}(\widehat{x}\_{i})=\sqrt{q}\lambda_{j}\psi_{j}(\widehat{x}\_{i}).$$
Moreover, $U^{(q)}$ has orthonormal columns, hence it is invertible, and therefore 
$$K^{(q)}\approx U^{(q)}\Lambda^{(q)}(U^{(q)})^{\top}.$$
Of course, equality would mean that our integration is **exact**, thus we can view the numerical integration problem as the matrix eigen-decomposition problem of $K^{(q)}$.

## Estimating the Gram Matrix

We now arrive at the core of the Nyström method: bridging the small sample $q$ to the full dataset $N$. Define
$$\mathbf{K}\in\mathbb{R}^{N\times N}:\quad \mathbf{K}\_{i,j}=K(x_{i},x_{j}),$$
as the Gram matrix of the input dataset. By re-ordering the order of the points, we may assume the first $q$ points are our sample $\widehat{x}\_{1}, \dots, \widehat{x}\_{q}$, and the remaining $N-q$ points are the rest of the dataset. In other words, $\widehat{x}\_i = x_i$ for $1\le i\le q$. Thus, we can write $\mathbf{K}$ in block form:
$$\mathbf{K}=\begin{bmatrix}K^{(q)} & K_{q,N-q} \\\\ K_{N-q,q} & K_{N-q,N-q}\end{bmatrix}.$$
Since the Gram matrix is symmetric, it must hold $K_{q,N-q}=K_{N-q,q}^{\top}$. Denote $K_{q,q}=K^{(q)}$. Moreover, denote $$K_{N,q}=\begin{bmatrix}K^{(q)} \\\\ K_{N-q,q} \end{bmatrix},\quad K_{q,N}=\begin{bmatrix}K^{(q)} & K_{q,N-q}\end{bmatrix}$$
By symmetry, $K_{N,q}=K_{q,N}^{\top}$. If $N$ is large, then setting $q=N$ and computing the eigen decomposition of $\mathbf{K}$ is very expensive. So the question becomes:
*How can we approximate $\mathbf{K}$ using the eigen decomposition of the small block $K^{(q)}$?*

This is the method's claim to fame -- because we are dealing with kernels, we can use the decomposition of $K^{(q)}$ to estimate the other values of $\mathbf{K}$.

### The Nystrom Extension
The key insight comes from our integral approximation earlier. Recall that for an eigenfunction $\psi_j$ and eigenvalue $\lambda_j$, we have:
$$\lambda_{j}\psi_{j}(y) \approx \frac{1}{q}\sum_{i=1}^{q}K(y,x_i)\cdot \psi_{j}(x_{i}).$$
Rearranging this allows us to **extend** the value of the eigenfunction to *any* new point $y \in X$ given its values on the set $x_1,\ldots ,x_q$ (approximately):
$$ \psi_{j}(y) \approx \frac{1}{q\lambda_n} \sum_{i=1}^{q}K(y,x_{i})\cdot \psi_{j}({x}\_{i}). $$
Using our matrix definitions $\lambda_j^{(q)} = q\lambda_j$ and $(U^{(q)})\_{i,j} = \frac{1}{\sqrt{q}}\psi_j({x}\_i)$, we can rewrite the extension formula as:
$$ \psi_{j}(y) \approx \frac{1}{\lambda_j^{(q)}} \sum_{i=1}^{q}K(y,{x}\_{i}) \cdot \sqrt{q}(U^{(q)})\_{i,j}. $$
In particular, for $i>q$, we have $$\psi_j(x_i)\approx \frac{\sqrt{q}}{\lambda_j^{(q)}}\sum_{t=1}^q K(x_i,x_t)\cdot (U^{(q)})\_{t,j}=\frac{\sqrt{q}}{\lambda_j^{(q)}}(K_{N,q} U^{(q)})\_{i,j}.$$

Define $\tilde{u}^{(j)}\in\mathbb{R}^N$ denote the vector of **approximations** to $v_j=(\psi_j(x_1),\ldots, \psi_j(x_N))$, defined by the formula above. Then $$\tilde{u}^{(j)}=\frac{\sqrt{q}}{\lambda_j^{(q)}} \[K_{N,q}U^{(q)}\]\_{\*,j}=\sqrt{q}\cdot [K_{N,q}U^{(q)}(\Lambda^{(q)})^{-1}]\_{\*,j}$$that is, the $j$-column of the scaled matrix product. Define $$\tilde{U}=\begin{bmatrix}\tilde{u}^{(1)} & \cdots & \tilde{u}^{(q)}\end{bmatrix}\in \mathbb{R}^{N\times q}.$$

### Arriving at a Low-Rank Approximation
Recall that we've shown $$K(x,y)=\sum_n \lambda_n \psi_n(x)\psi_n(y)$$
The best $q$-rank approximation is given by taking the top $q$ eigenvalues. Therefore, $$\mathbf{K}\_{i,j}=K(x_i,x_j)\approx \sum_{n=1}^q \lambda_n \psi_n(x_i)\psi_n(x_j)$$
Plugging in the notation from the previous section: $$\mathbf{K}\_{i,j}\approx \sum_{n=1}^q \lambda_n v_n(i)v_n(j)=\sum_{n=1}^q \lambda_n (v_n \otimes v_n)\_{i,j}$$ where $\otimes$ denotes the Kronecker product of vectors (as we've encountered in the previous post). Therefore the best $q$-rank approximation for $\mathbf{K}$ is $\sum_{n=1}^q \lambda_n (v_n \otimes v_n)$. We approximate this by using $\tilde{u}^{(n)}$ instead of $v_n$: $$\mathbf{K}\approx \sum_{n=1}^q\lambda_n (\tilde{u}^{(n)}\otimes \tilde{u}^{(n)}) = \sum_{n=1}^q \lambda_n \tilde{u}^{(n)}(\tilde{u}^{(n)})^{\top}$$ where we use the fact $x\otimes y= xy^{\top}$ is an equivalent way to define the Kronecker product of vectors. 

> [!tip] Lemma
> Let $A,B$ be matrices of size $n\times m,m\times n$ respectively. Let $a_1,\ldots,a_m\in \mathbb{R}^n$ denote the columns of $A$ and $b_1,\ldots,b_m\in\mathbb{R}^n$ denote the rows of $B$. Then $$AB=\sum_{i=1}^m a_i b_i^{\top}.$$

**Proof.** The $(k,j)$-th element of $AB$ is by definition $\sum_{i=1}^m A_{k,i}B_{i,j}=\sum_{i=1}^m a_i(k)\cdot b_i(j)$. On the other hand, the $(k,j)$-th element of $a_ib_i^{\top}$ is $a_i(k)\cdot b_i(j)$ by definition, and the equality follows. $\blacksquare$

Returning to the approximation of $\mathbf{K}$, note that scaling the $n$-th column of $\tilde{U}$ by $\lambda_n$ is the same as multiplying by a diagonal matrix from the right. Using the lemma, the sum can be re-written as $$\mathbf{K}\approx \frac{1}{q} \tilde{U}\Lambda^{(q)}\tilde{U}^{\top}$$
where the extra $1/q$ factor sets off the definition of $\Lambda^{(q)}$. Substituting the definition of $\tilde{U}$: 
$$
\begin{aligned}
\mathbf{K} &\approx \left( K_{N,q} U^{(q)} (\Lambda^{(q)})^{-1} \right) \Lambda^{(q)} \left( K_{N,q} U^{(q)} (\Lambda^{(q)})^{-1} \right)^\top \\\\
&= K_{N,q} U^{(q)} (\Lambda^{(q)})^{-1} \underbrace{\Lambda^{(q)} (\Lambda^{(q)})^{-1}}\_{I} (U^{(q)})^\top K_{q,N} \\\\
&= K_{N,q} \underbrace{U^{(q)} (\Lambda^{(q)})^{-1} (U^{(q)})^\top}\_{(K^{(q)})^{-1}} K_{q,N}\\\\
& = K_{N,q} (K^{(q)})^{-1} K_{q,N}
\end{aligned}
$$

In the last step, we used the fact that $K^{(q)} = U^{(q)} \Lambda^{(q)} (U^{(q)})^\top$, which implies its inverse (or pseudo-inverse) is $U^{(q)} (\Lambda^{(q)})^{-1} (U^{(q)})^\top$.

> [!tip] Theorem: The Nystrom Approximation
> The Nyström method approximates the full Gram matrix $\mathbf{K}$ by:
> $$ \widetilde{K} = K_{N,q} (K^{(q)})^{-1} K_{q,N}. $$
> In block form, this results in:
> $$ \widetilde{K} = \begin{bmatrix} K^{(q)} & K_{q, N-q} \\\\ K_{N-q, q} & K_{N-q, q} (K^{(q)})^{-1} K_{q, N-q} \end{bmatrix}. $$

Moreover, the term $K_{N-q, q} (K^{(q)})^{-1} K_{q, N-q}$ is precisely the **Schur complement** of $K^{(q)}$ in $\widetilde{K}$ being set to zero (meaning $\widetilde{K}$ has rank $q$). I hope to write a blog on Schur complements soon enough, but for now I won't go into details here.

**Remark**. The only block which is approximated is the bottom right block $K_{N-q,N-q}$. The others are computed exactly.

**Remark 2.** Low-rank approximations are useful because the operations cost proportionally to the rank. For example, inverting the matrix $K^{(q)}$ is done in time $O(q^3)$ instead of $O(N^3)$! Computing the product of $\tilde{K}$ by a vector can be done in time $O(Nq)$ instead of $O(N^2)$.

## Conclusions
The Nystrom method gives an alternative to the direct eigen decomposition of the Gram matrix, resulting in an approximate best low rank approximation to $\mathbf{K}$ (the best low-rank approximation requires the eigen decomposition). The core idea is to use the fact numerical integration on a small number of points yields a good approximation of the eigenfunctions on **new points** too. Thus computing the top eigenfunctions on a small number of points (equivalent to decomposing the small matrix $K^{(q)}$) suffices to obtain good approximations of all the other values.

There have been many innovations and adaptations based on this basic method, that propose different sampling methods, reconstruction methods and numerical stability improvements. However, the core ideas are the same.

## References
1.   **Williams, C. K., & Seeger, M.** (2001). [*Using the Nyström method to speed up kernel machines*](https://papers.nips.cc/paper_files/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf).
2.   **Drineas, P., & Mahoney, M. W.** (2005). [*On the Nyström Method for Approximating a Gram Matrix for Improved Kernel-Based Learning*](https://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf).
3.   **Kumar, S., Mohri, M., & Talwalkar, A.** (2012). [*Sampling methods for the Nyström method*](https://jmlr.csail.mit.edu/papers/volume13/kumar12a/kumar12a.pdf).
4.   **Gittens, A., & Mahoney, M. W.** (2016). [*Revisiting the Nyström Method for Improved Large-Scale Machine Learning*](https://arxiv.org/abs/1303.1849).
5.   **Bach, F.** (2013). [*Sharp analysis of low-rank kernel matrix approximations*](https://arxiv.org/abs/1208.2015).
6.   **Nyström, E. J.** (1930). [*Über die Praktische Auflösung von Integralgleichungen with Anwendungen auf Randwertaufgaben*](https://projecteuclid.org/journals/acta-mathematica/volume-54/issue-none/%C3%9Cber-Die-Praktische-Aufl%C3%B6sung-von-Integralgleichungen-mit-Anwendungen-auf-Randwertaufgaben/10.1007/BF02547521.full).