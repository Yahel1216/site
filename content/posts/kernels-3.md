---
title: "Tensor Sketch: Polynomial Kernels"
date: 2026-01-09
slug: kernel-3
draft: false
katex: true
description: "Using linear sketching techniques for fast approximation of the polynomial kernel"
series: "Kernel Methods"
tags: ["optimization", "machine-learning", "linear-algebra", "analysis"]
categories: ["Machine Learning", "Kernels"]
---
> [!info] Prerequisites
> *   **Linear Algebra:** Inner products, tensor products, and vectorization.
> *   **Probability:** Hash functions, independence, and variance.
> *   **Algorithms:** Fast Fourier Transform (FFT) and basic convolution.
> *   **Kernel Methods:** Familiarity with the Polynomial Kernel.

In previous posts, we discussed the Radial Basis Function (RBF) kernel and how to approximate it using **Random Fourier Features**. Today, we turn our attention to another fundamental kernel—the **Polynomial Kernel**—and a powerful algebraic technique to approximate it called **Tensor Sketching**.

## The Polynomial Kernel

For $x,y\in \mathbb{R}^{d}$ and $c\ge 0$, the polynomial kernel of degree $p$ is defined as:
$$K(x,y)=(\left\langle x,y \right\rangle+c)^{p}=\left(\sum_{i=1}^{d}x_{i}y_{i}+c\right)^{p}.$$
If $c=0$, the kernel is called **Homogeneous**.

By applying the binomial formula, we can expand the kernel as:
$$K(x,y)=\sum_{j=0}^{p}\binom{p}{j}\cdot(\left\langle x,y \right\rangle)^{j}c^{p-j}.$$
Further applying the multinomial formula to the term $\left\langle x,y \right\rangle^{j}$:
$$\left\langle x,y \right\rangle^{j}=\sum_{n_{1}+\ldots+n_{d}=j}\binom{j}{n_{1},\ldots,n_{d}} \cdot \prod_{i=1}^{d}(x_{i}y_{i})^{n_{i}}.$$
Note that for $n_{1}+\ldots+n_{d}=j$, we have the combinatorial identity:
$$\binom{p}{j}\cdot \binom{j}{n_{1},\ldots,n_{d}}=\frac{p!}{(p-j)!n_{1}!\cdots n_{d}!}=\binom{p}{n_{1},\ldots,n_{d},p-j}.$$
Letting $n_{d+1}=p-j$, we have $n_{1}+\ldots+n_{d+1}=p$. Since we sum over all $j$, we can rewrite the kernel as a summation over all non-negative integer partitions of $p$ into $d+1$ parts:
$$K(x,y)=\sum_{n_{1}+\ldots+n_{d+1}=p}\binom{p}{n_{1},\ldots,n_{d+1}}\prod_{i=1}^{d}(x_{i}y_{i})^{n_{i}}\cdot c^{n_{d+1}}.$$
The total number of summands is the number of solutions to $n_{1}+\ldots+n_{d+1}=p$, which is $\binom{p+d}{d}$.

We can explicitly construct a feature map $\varphi:\mathbb{R}^{d}\to \mathbb{R}^{\binom{p+d}{d}}$. Enumerate all solutions to the partition equation, and for the $k$-th solution $(n_{1},\ldots,n_{d+1})$, define the $k$-th feature:
$$\psi_{k}(x)=\sqrt{\binom{p}{n_{1},\ldots,n_{d+1}}}\cdot x_{1}^{n_{1}}\cdots x_{d}^{n_{d}}\cdot c^{n_{d+1}/2}.$$
Defining $\varphi(x)=(\psi_{1}(x),\ldots,\psi_{\binom{p+d}{d}}(x))$, we satisfy:
$$\left\langle \varphi(x),\varphi(y) \right\rangle = K(x,y).$$
While $\varphi$ maps to a finite-dimensional space, the dimension $\binom{p+d}{d} \approx p^d$ grows exponentially with $d$. For high-dimensional data, computing explicit features is intractable.

**Note:** The polynomial kernel is **not** stationary. For example, if $p=2$ and $c=0$, $K(x,x) = \|x\|^4$. If we take two vectors with different norms, $K(x,x) \neq K(y,y)$, even if $x-x=y-y=0$. Therefore, the Random Fourier Features method (which relies on Bochner's theorem for shift-invariant kernels) is **not applicable**.

**Note:** we can always reduce the general case to the homogeneous case by appending a coordinate with value $\sqrt{c}$ to each vector. Thus, we will focus on the homogeneous case where $K(x,y) = \langle x, y \rangle^p$.

## As Tensor Products

> [!caution] Definition
> The **tensor product** (or Kronecker product) of two vectors $a\in\mathbb{R}^{n},b\in\mathbb{R}^{m}$ is the vector $a \otimes b\in\mathbb{R}^{n\cdot m}$, whose $(i,j)$-th coordinate is $a_{i}b_{j}$ (using double indexing).
>
> For a vector $a$, let $a^{(p)}$ denote the $p$-th tensor power:
> $$a^{(p)} = \underbrace{a \otimes \ldots \otimes a}\_{p \text{ times}} \in \mathbb{R}^{d^p}.$$

> [!important] Lemma
> It holds that $\left\langle x^{(p)},y^{(p)} \right\rangle=\left\langle x,y \right\rangle^{p}$.

**Proof.**
We proceed by induction on $p$.
*   **Base case:** $p=1$ is trivial ($\langle x, y \rangle = \langle x, y \rangle$).
*   **Step:** Suppose the claim holds for $p$. Let $a=x^{(p)}$ and $b=y^{(p)}$. Then:
    $$\left\langle x\otimes a,y\otimes b \right\rangle=\sum_{i\in [d],j\in [d^p]}(x\otimes a)\_{(i,j)}\cdot (y\otimes b)\_{(i,j)}=\sum_{i,j}x_{i}y_{i}a_{j}b_{j}=\left(\sum_{i}x_{i}y_{i}\right)\cdot \left(\sum_{j}a_{j}b_{j}\right).$$
    This equals $\langle x,y \rangle \cdot \langle a,b \rangle$. By the induction hypothesis, $\langle a,b \rangle = \langle x,y \rangle^p$, so the result is $\langle x,y \rangle^{p+1}$. $\blacksquare$

This suggests that if we can approximate $x^{(p)}$ and $y^{(p)}$ with low-dimensional vectors, we can approximate the polynomial kernel efficiently.

## Sketching

Linear sketches are random linear **dimension-reducing** maps that preserve the norm of vectors with high probability. The most famous examples include the **Johnson-Lindenstrauss Transform** and the **Count-Sketch**. We will focus on the Count-Sketch because of its unique interaction with tensor products.

### Hash Functions
Recall the definition of $k$-wise independent hash families.

> [!caution] Definition
> A set $\mathcal{H}$ of functions $[d]\to [R]$ is a **$k$-wise independent family** if for every distinct set of indices $I\subset [d]$ of size $|I| \le k$, and any vector of values $\mathbf{v}\in [R]^{|I|}$, it holds:
> $$\Pr_{f\in \mathcal{H}}\left((f(i))\_{i\in I}=\mathbf{v}\right)=\frac{1}{R^{|I|}}.$$
> In simple terms, any $k$ inputs are mapped to independent uniform random outputs.

Note that if $\mathcal{H}$ is $k$-wise independent it is also $k'$-wise independent for $k'<k$.

### Count Sketch
> [!caution] Definition: Count Sketch
> Given a hash function $h:[d]\to [R]$ drawn from a $2$-wise independent family, and a **sign function** $s:[d]\to \set{\pm 1}$ drawn from a $4$-wise independent family, define the **Count-Sketch** with respect to $h,s$ to be the matrix in $\mathbb{R}^{R\times d}$ denoted by $C$, defined by:
> $$C_{k,i}:=\begin{cases}s(i) & h(i)=k, \\\\ 0 & \text{else},\end{cases}$$
> for every $i\in [d]$ and $k\in [R]$.
> Note that every column $i$ has a **single** non-zero element, at row $h(i)$. Explicitly, given a vector $x\in\mathbb{R}^{d}$, its Count-Sketch is:
> $$(Cx)\_{k}=\sum_{i:h(i)=k}s(i)\cdot x_{i},$$
> for every $k\in [R]$.

One way to think of count-sketch is as a random *binning* operator. We send each element in the vector to a different bin, with a total of $R$ bins, while multiplying it by a random sign. Note that since $C$ has at $d$ non-zero values, computing $Cx$ can be done in $O(d)$ operations.

An important property of Count-Sketch is that it preserves inner products in expectation, and the variance is controllable:

> [!important] Lemma
> Let $C$ be a random Count-Sketch matrix. Then for any $x,y\in\mathbb{R}^{d}$:
> $$\mathbb{E}[\left\langle Cx,Cy \right\rangle]=\left\langle x,y \right\rangle, \quad \text{and} \quad \mathbf{Var}(\left\langle Cx,Cy \right\rangle)\le \frac{2}{R}\left\lVert x \right\rVert^{2} \left\lVert y \right\rVert^{2}.$$

**Proof.**
The inner product of the sketches is:
$$\left\langle Cx,Cy \right\rangle=\sum_{k=1}^{R}(Cx)\_{k}(Cy)\_{k}=\sum_{k=1}^{R}\sum_{i,j:h(i)=h(j)=k} s(i)s(j)x_{i}y_{j}.$$
Taking the expectation over the sign functions $s$:
*   If $i \neq j$, $\mathbb{E}[s(i)s(j)] = \mathbb{E}[s(i)]\mathbb{E}[s(j)] = 0$.
*   If $i = j$, $s(i)^2 = 1$, implying $\mathbb{E}[s(i)^2]=1$.

Thus, the only surviving terms are where $i=j$:
$$\mathbb{E}[\left\langle Cx,Cy \right\rangle] = \sum_{k=1}^R \sum_{i:h(i)=k} x_i y_i = \sum_{i=1}^d x_i y_i = \langle x, y \rangle.$$
Note that since $\sum_k \mathbb{1}\_{h(i)=k} = 1$ always, the randomness of $h$ does not affect the mean.

For the variance, let $\delta_{i,j}$ be the indicator that $h(i)=h(j)$. Note $$\begin{aligned}
\left\langle Cx,Cy \right\rangle&=\sum_{i,j}\delta_{i,j}s(i)s(j)\cdot x_{i}y_{j}
\\\\ &=\sum_{i}\delta_{i,i}s(i)^{2} x_{i}y_{i}+\sum_{i\not=j}\delta_{i,j}s(i)s(j)x_{i}y_{j}
\\\\ &=\left\langle x,y \right\rangle+\sum_{i\not=j}\delta_{i,j}s(i)s(j)x_{i}y_{j}
\end{aligned}$$
Thus:
$$\left\langle Cx,Cy \right\rangle^2 = \left( \langle x,y \rangle + \sum_{i \neq j} \delta_{i,j} s(i)s(j) x_i y_j \right)^2.$$
The cross-terms ($\langle x,y\rangle\cdot \sum_{i\not=j}\delta_{i,j}s(i)s(j)x_i y_j$) vanish in expectation due to $\mathbb{E}[s(i)]=0$. The expectation of the square of the sum involves terms like $\delta_{i,j}\delta_{i',j'}s(i)s(j)s(i')s(j')$.
Due to 4-wise independence of $s$, the expectation is non-zero only if the indices $\{i,j,i',j'\}$ match in pairs. Specifically, either $(i,j)=(i',j')$ or $(i,j)=(j',i')$.
Using $\mathbb{E}[\delta_{i,j}] = 1/R$ (for $i \neq j$), we obtain $$\mathbb{E}\left[\left(\sum_{i\not=j}\delta_{i,j}s(i)s(j)x_{i}y_{j}\right)^{2}\right]= \frac{1}{R}\sum_{i\not=j}(x_{i}^{2}y_{j}^{2}+x_{i}y_{i}x_{j}y_{j})$$
Note that $$\sum_{i\not=j}x_{i}^{2}y_{j}^{2}\le \sum_{i,j}x_{i}^{2}y_{j}^{2}=\left(\sum_{i}x_{i}^{2}\right)\left(\sum_{j}y_{j}^{2}\right)= \left\lVert x \right\rVert^{2}\cdot \left\lVert y \right\rVert^{2},$$and by Cauchy-Schwarz $$\sum_{i\not=j}x_{i}y_{i}x_{j}y_{j}\le \sum_{i,j}x_{i}y_{i}x_{j}y_{j}=\left(\sum_{i}x_{i}y_{i}\right)^{2}=\left\langle x,y \right\rangle^{2}\le \left\lVert x \right\rVert^{2}\cdot \left\lVert y \right\rVert^{2},$$
and so we conclude that $$\mathbf{Var}(\left\langle Cx,Cy \right\rangle) = \mathbb{E}[\langle Cx, Cy\rangle^2]- \mathbb{E}[\langle Cx,Cy\rangle]^2= \frac{1}{R}\sum_{i\not=j}(x_i ^2 y_j^2 + x_iy_i x_j y_j)\le \frac{2}{R}\left\lVert x \right\rVert^{2}\left\lVert y \right\rVert^{2}$$
$\blacksquare$

## Tensor Sketch
Suppose we have a $2$-wise independent family $[d]\to [R]$. How can we construct a $2$-wise family $[d^{2}]\to [R]$? 

One way is to take $h_{1},h_{2}:[d]\to [R]$ (sampled randomly) and define
$$H(i_{1},i_{2})=h_{1}(i_{1})+ h_{2}(i_{2})\pmod{R}.$$
The function $H$ sampled this way is $2$-wise independent too. For sign functions, we can take
$$S(i_{1},i_{2})=s_{1}(i_{1})\cdot s_{2}(i_{2}),$$
giving a new $2$-wise independent sign function on $[d^{2}]$. Of course we can take $h_{1},h_{2}$ to be from different families and universe sizes, say $h_1:[d_1]\to [R],h_2:[d_2]\to [R]$ and obtain $H:[d_1 d_2]\to [R]$.

The main observation is the following lemma.

> [!important] Lemma
> Let $h_{1},h_{2}:[d]\to [R]$ be drawn from a $2$-wise independent family and $s_{1},s_{2}$ also drawn from a $2$-wise independent family of sign functions. Define $H=h_{1}+h_{2}\pmod{R}$ and $S=s_{1}\cdot s_{2}$ and let $C$ denote the Count-Sketch with respect to $H,S$, sketching vectors of size $d^{2}$. Moreover, let $C^{1},C^{2}$ denote the Count-Sketch with respect to $h_{1},s_{1}$ and $h_{2},s_{2}$. Then for all $x,y\in\mathbb{R}^{d}$, it holds
> $$C(x \otimes y)=C^{1}x\* C^{2}y,$$
> where $\*$ denotes $R$-cyclic convolution.

Let us define cyclic convolution:
> [!caution] Definition
> Given two vectors $u,v\in\mathbb{R}^m$, the $m$-cyclic convolution denoted $u\*v\in\mathbb{R}^{m}$ is given by
> $$(u\*v)\_{k}=\sum_{i=1}^{m}u_{i}\cdot v_{(k-i)\mod m},$$
> where we interpret $v_{0}=v_{m}$. We can also write it as
> $$(u\*v)\_{k}=\sum_{i,j:i+j\equiv k\pmod{m}} u_{i}\cdot v_{j}.$$

**Proof.**
Let $\equiv$ denote congruence modulo $R$. For $k\in [R]$ we have
$$\begin{aligned}(C(x \otimes y))\_{k} &= \sum_{i_{1},i_{2}:H(i_{1},i_{2})=k}S(i_{1},i_{2})\cdot (x \otimes y)\_{i_{1}i_{2}}
\\\\ &=\sum_{i_{1},i_{2}:h_{1}(i_{1})+h_{2}(i_{2})\equiv k}s_{1}(i_{1})s_{2}(i_{2})x_{i_{1}}y_{i_{2}}
\\\\ &= \sum_{j_{1},j_{2}\in [R]: j_{1}+j_{2}\equiv k}\sum_{i_{1},i_{2}:h_{1}(i_{1})=j_{1},h_{2}(i_{2})=j_{2}}(s_{1}(i_{1})x_{i_{1}})\cdot (s_{2}(i_{2})y_{i_{2}})
\\\\ &= \sum_{j_{1},j_{2}\in [R]: j_{1}+j_{2}\equiv k}\left(\sum_{i:h_{1}(i)=j_{1}} s_{1}(i)x_{i} \right)\cdot \left(\sum_{i: h_{2}(i)=j_{2}}s_{2}(i)y_{i}\right)
\\\\ &= \sum_{j_{1},j_{2}\in[R]:j_{1}+j_{2}\equiv k}(C^{1}x)\_{j_{1}}\cdot (C^{2}y)\_{j_{2}}
=(C^{1}x * C^{2}y)\_{k}.\end{aligned}$$
$\blacksquare$

We thus obtain the corollary:

> [!important] Corollary
> Suppose $C$ is the count-sketch for dimension $d^{p}$ with respect to
> $$H(i_{1},\ldots,i_{p})=h_{1}(i_{1})+\ldots + h_{p}(i_{p})\pmod{R},$$
> and $S(i_{1},\ldots,i_{p})=s_{1}(i_{1})\cdots s_{p}(i_{p})$, where all the hash functions are drawn independently from respective $2$-wise independent families. Let $C^{j}$ be the count-sketch for dimension $d$ with respect to $h_{j},s_{j}$. Then
> $$C(x_{1}\otimes \ldots \otimes x_{p})=C^{1}x_{1}* \ldots  * C^{p}x_{p}.$$
> In particular $C(x^{(p)})$ is the convolution of $p$-different count sketches of $x$.

### Computing Convolutions

The nice thing about cyclic convolution is that it can be computed fast via the fast Fourier transform.

> [!tip] Theorem: The Convolution Theorem
> The $m$-cyclic convolution of two vectors $u,v$ is given by
> $$u\*v=\mathrm{DFT}^{-1}(\mathrm{DFT}(u)\odot \mathrm{DFT}(v)),$$
> where $\odot$ denotes element-wise multiplication of vectors.

By induction, we see that (note convolution is associative)
$$u\*v\*w=(u\*v)\*w=\mathrm{DFT}^{-1}(\mathrm{DFT}(u\*v)\odot \mathrm{DFT}(w))=\mathrm{DFT}^{-1}(\mathrm{DFT}(u)\odot \mathrm{DFT}(v)\odot \mathrm{DFT}(w))$$
and in general to compute the convolution of $p$ vectors, we need to compute the DFT of each vector, take the element-wise products, and compute a single inverse DFT.

> [!tip] Theorem: The FFT Theorem
> The DFT of size $m$ can be computed in time $O(m\log m)$, and so can the inverse DFT.

Therefore computing the convolution of $p$ vectors requires time $O(pm\log m)$.

Thus, to compute $Cx^{(p)}$, one needs to:
- draw $p$ different hash functions $h_{1},\ldots,h_{p}$ and sign functions $s_{1},\ldots,s_{p}$,
- compute $p$ different count sketches $C^{1}x,\ldots,C^{p}x$,
- compute the DFT (of size $R$) for each sketch, denoted $\widehat{C^{1}x},\ldots,\widehat{C^{p}x}$,
- compute the element-wise product $u=\widehat{C^{1}x}\odot \ldots\odot \widehat{C^{p}x}$,
- return the inverse DFT of $u$.

## Recap
To conclude, the tensor sketch produces a **random feature vector** for an input $x\in\mathbb{R}^{d}$, in $\mathbb{R}^{R}$, which is given as the count-sketch of $x^{(p)}$. Computing this random feature vector requires
$$O(p(d+R\log R)),$$
since we have to perform $p$ sketches and compute the convolution of $p$ vectors of size $R$. Using the statistical properties of count-sketch, we have
$$\mathbb{E}\left[\left\langle Cx^{(p)},Cy^{(p)} \right\rangle\right]=\left\langle x^{(p)},y^{(p)} \right\rangle=\left\langle x,y \right\rangle^{p}=K(x,y),$$
with considerably good variance (which by Chebyshev's inequality implies decent concentration around the mean).

This is much better than computing the inner product of $x^{(p)}$ and $y^{(p)}$ in time $O(d^{p})$. The ideas behind tensor sketch and count-sketch are relevant to many other fields, especially numerical linear algebra, dimensionality reduction and even neural network acceleration.
## References
1.  **Pagh, R.** (2013), [*Compressed Matrix Multiplication*](https://dl.acm.org/doi/10.1145/2493252.2493254).
2.  **Pham, N., & Pagh, R.** (2013), [*Fast and Scalable Polynomial Kernels via Explicit Feature Maps*](https://dl.acm.org/doi/10.1145/2487575.2487591).
3.  **Ahle, T. D., et al.** (2020), [*Oblivious Sketching of High-Degree Polynomial Kernels*](https://epubs.siam.org/doi/abs/10.1137/1.9781611975994.9).
4.  **Charikar, M., et al.** (2002), [*Finding Frequent Items in Data Streams*](https://www.khoury.northeastern.edu/home/pandey/courses/cs7280/spring25/papers/frequent.pdf).
5.  **Woodruff, D. P.** (2014), [*Sketching as a Tool for Numerical Linear Algebra*](https://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/wNow3.pdf).