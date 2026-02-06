---
title: "Fast Matrix Multiplication - Part 2"
date: 2026-01-14
slug: fmm-2
draft: false
katex: true
description: "Bilinear Algorithms, Tensors and the Tensor rank"
series: "Fast Matrix Multiplication"
tags: ["theory", "matrix-multiplication", "tensors"]
categories: ["Theory", "Matrix Multiplication"]
---
> [!note] Prerequisites
> *   **[Part 1 of this series](/posts/fmm-1):** Familiarity with standard Matrix Multiplication (MM) and $\omega$.
> *   **Linear Algebra:** Vector spaces, Bases, Dual spaces, and Tensor Products (basic definition).
> *   **Abstract Algebra:** Fields and Polynomial rings (helpful for the tensor intuition).

***

In the previous post, we discussed Strassen's algorithm and the definition of the exponent $\omega$. We demonstrated that the complexity of matrix multiplication is dominated by the number of multiplications in the base algorithm, assuming a recursive approach.

In this post, we will introduce **bilinear algorithms**, their tensor representations, and how these concepts fundamentally relate to fast matrix multiplication.

## Quick Recap on Tensor Products

Given two vector spaces $V,U$, a standard operation is to consider their **direct sum**, $V\oplus U$, which is a vector space structure over the Cartesian product $V\times U$, where addition and scalar multiplication is done separately in each coordinate (so $(v,u)+(v',u')=(v+v', u+u')$ and similarly for scalar multiplication). This is the simplest way to combine vector spaces. Note that $\dim (V\oplus U)=\dim V+\dim U$.

Another way to combine spaces is via the **tensor product**, denoted $V\otimes U$. We will first need the following definition of a bilinear function.
> [!caution] Definition
> Let $U,V,W$ be vector spaces over a field $\mathbb{F}$. A function $f:U\times V\to W$ is called **bilinear** if it is linear in each argument separately.
> That is, for all $u,u'\in U$, $v,v'\in V$, and $\alpha,\beta\in \mathbb{F}$:
> $$ \begin{aligned} f(\alpha u+\beta u',v) &= \alpha f(u,v) + \beta f(u',v) \\\\  f(u,\alpha v+\beta v') &= \alpha f(u,v) + \beta f(u,v') \end{aligned} $$

An example of a bilinear function is the matrix multiplication function over $U=\mathbb{F}^{n\times m},V=\mathbb{F}^{m\times p}$ and $W=\mathbb{F}^{n\times p}$, given by $f(A,B)=AB$.

To define $V\otimes U$, we consider a new vector space $W$ which is the linear span of elements in $V\times U$, treating each couple $(v,u)$ as being linearly independent from other couples. We then define the **bilinear relations** on $W$, which is an **equivalence** relation on the elements of $W$, defined by: $$(\alpha v+\beta v', \gamma u+ \delta u')\sim \alpha\cdot \gamma (v,u)+ \beta\cdot \gamma (v',u)+ \alpha\cdot \delta(v,u') + \beta\cdot \delta(v',u')$$
Now, we define $V\otimes U$ to be the **quotient space** of $W$ by the equivalence. This defines a new vector space, where equivalent vectors are treated as identical vectors.

We define $\otimes: V\times U\to V\otimes U$ by setting $(v,u)\mapsto [(v,u)]\_{\sim}$ and we denote the equivalence class by $v\otimes u$. Note that by construction, $\otimes$ is a bilinear map, since the equivalence relation ensures for example $$[(\alpha v,u)]\_{\sim}=\alpha \cdot [(v,u)]_\sim \implies (\alpha v)\otimes u= \alpha (v\otimes u)$$

In the case of vector spaces, if $v_1,\ldots,v_n$ is a basis for $V$ and $u_1,\ldots,u_m$ for $U$, then we can think of $V\otimes U$ to be the linear span of the vectors $\lbrace v_{i}\otimes u_j\rbrace_{i\in [n],j\in [m]}$. Indeed, for $v\in V,u\in U$ we can write $v=\sum_{i=1}^n \alpha_i v_i$ and $u=\sum_{j=1}^m \beta_j u_j$ to obtain that $$v\otimes u=\sum_{i,j}\alpha_i \beta_j (v_i\otimes u_j)$$
It is also easy to see that the vectors $v_i \otimes u_j$ are linearly independent from each other. Thus $\dim (V\otimes U)=\dim V\cdot \dim U$.

**The Universal Property of the Tensor Product** is the following observation: Every **bilinear** map $f:V\times U\to X$ extends in a **unique** manner to a **linear** map $\tilde{f}:V\otimes U\to X$, which satisfies $f(v,u)=\tilde{f}(v\otimes u)$.

This property gives a recipe to construct linear maps from the tensor product space. Start with a bilinear map on $V\times U$, and extend linearly to $V\otimes U$. In particular, given a bilinear function $f:V\times U\to X$, the extension must be $\tilde{f}\left( \sum_{i,j} \alpha_{i,j} v_i\otimes u_j\right)=\sum_{i,j}\alpha_{i,j} f(v_i,u_j)$.

### Special Case of Matrix Spaces
If $U,V$ are both matrix spaces, i.e. $U=\mathcal{M}\_n(\mathbb{F})$ and $V=\mathcal{M}\_m(\mathbb{F})$, then $V\otimes U$ has a special interpretation. Taking the standard basis matrices $E_{i,j}$ of size $n\times n$ and $F_{i,j}$ of size $m\times m$, the basis for $V\otimes U$ is $F_{k,\ell}\otimes E_{i,j}$. We can think of this basis vector as the matrix of size $nm\times nm$ with $E_{i,j}$ in the $(k,\ell)$-th block of size $n\times n$. In other words, it has $1$ in row $k\cdot n + i$ and column $\ell\cdot n +j$, and zeros everywhere else. For general matrices, expanding them in the respective bases, we recover the **Kronecker** product of matrices, defined by $$\mathbf{A}\in V,\mathbf{B}\in U:\quad A\otimes B = \begin{pmatrix} A_{0,0} \mathbf{B} & \cdots & A_{0,m-1} \mathbf{B} \\\\ \vdots &\ddots
\\\\ A_{m-1,0} \mathbf{B} & \cdots & A_{m-1,m-1}\mathbf{B}\end{pmatrix}$$
It is easy to see that $V\otimes U$ is therefore just $\mathcal{M}_{mn}(\mathbb{F})$.

## Bilinear Algorithms
Strassen proved that *any* bilinear function can be computed using a specific structural recipe. This is often called the **Strassen Normal Form** (or Rank Decomposition).

> [!tip] Theorem (Strassen Normal Form)
> Fix bases for $U,V,W$. A function $f:U\times V\to W$ is bilinear **if and only if** there exists a vector space $X$ (of dimension $r$) and linear maps (matrices) $\mathbf{U}: U \to X$, $\mathbf{V}: V \to X$, and $\mathbf{W}: W \to X$ such that:
> $$f(u,v)=\mathbf{W}^{\top}(\mathbf{U}u\odot \mathbf{V}v)$$
> where $\odot$ denotes the element-wise product of vectors in $X$ (Hadamard product) relative to a fixed basis.

**Proof.**
First, we assume $f$ has the form $f(u,v)=\mathbf{W}^{\top}(\mathbf{U}u\odot \mathbf{V}v)$ and show it is bilinear. Indeed,
$$
\begin{aligned}
f(u+u',v) &= \mathbf{W}^{\top}(\mathbf{U}(u+u') \odot \mathbf{V}v) \\\\
&= \mathbf{W}^{\top}((\mathbf{U}u + \mathbf{U}u') \odot \mathbf{V}v) \\\\
&= \mathbf{W}^{\top}(\mathbf{U}u\odot \mathbf{V}v) + \mathbf{W}^{\top}(\mathbf{U}u'\odot \mathbf{V}v) \\\\
&= f(u,v)+f(u',v)
\end{aligned}
$$
Here we used the distributivity $(x+x')\odot y=x\odot y+ x'\odot y$. Linearity in the second argument and scalar multiplication follows identically.

Conversely, suppose $f$ is bilinear. Let $\lbrace u_{i}\rbrace \_{i=1}^n$, $\lbrace v_{j}\rbrace \_{j=1}^m$, and $\lbrace w_{\ell}\rbrace \_{\ell=1}^k$ be bases for $U,V,W$.
For every $i,j$ we can expand $f(u_i, v_j)$ in terms of the $w$-basis, thus there are constants $\alpha_{i,j,\ell}$ such that:
$$ f(u_{i},v_{j})=\sum_{\ell=1}^{k}\alpha_{i,j,\ell}w_{\ell}$$
Given arbitrary vectors $u=\sum_{i}\beta_{i}u_{i}$ and $v=\sum_{j}\gamma_{j}v_{j}$, bilinearity implies:
$$
f(u,v)=\sum_{i,j} \beta_{i}\gamma_{j} f(u_{i},v_{j}) = \sum_{\ell=1}^k \left(\sum_{i,j}\alpha_{i,j,\ell}\beta_{i}\gamma_{j}\right)w_{\ell}
$$
We define the intermediate space as the **tensor product space** $X = U \otimes V$. The dimension is $r = nm$, and the basis elements are $u_i\otimes v_j$ for every pair $(i,j)$.
We define the maps $\mathbf{U}$ and $\mathbf{V}$ to "select" coordinates:
$$
(\mathbf{U}u)\_{(i,j)} = \beta_i \quad \text{and} \quad (\mathbf{V}v)\_{(i,j)} = \gamma_j
$$
Formally, $\mathbf{U}\_{(i,j), i'} = \delta_{i,i'}$ (where the rows are indexed by tuples $(i,j)$, and the $\delta$ is the Kroncker delta function).
Then, the element-wise product corresponds to the cross-terms:
$$ (\mathbf{U}u \odot \mathbf{V}v)\_{(i,j)} = \beta_i \gamma_j $$
Finally, we define $\mathbf{W}$ to encode the $\alpha$ constants. Set $\mathbf{W}\_{(i,j), \ell} = \alpha_{i,j,\ell}$. Then:
$$
[\mathbf{W}^{\top}(\mathbf{U}u\odot \mathbf{V}v)]\_{\ell} = \sum_{i,j} \mathbf{W}\_{(i,j),\ell} (\beta_i \gamma_j) = \sum_{i,j} \alpha_{i,j,\ell} \beta_i \gamma_j
$$
This matches the $\ell$-th coordinate of $f(u,v)$. If we choose the bases to be the standard canonical bases we obtain the equality. Otherwise, all that is left to do is compose with the coordinate transformation for each basis. $\blacksquare$

### Interpreting the Normal Form
1.  **Algorithms vs. Functions:** The theorem provides a recipe. Given the matrices $\mathbf{U}, \mathbf{V}, \mathbf{W}$, we can *compute* $f$ by calculating three matrix-vector products and one element-wise product. We call the triple $(\mathbf{U}, \mathbf{V}, \mathbf{W})$ a **bilinear algorithm** for $f$.
2.  **Inner Dimension:** In the proof, we constructed $X$ with dimension $n \cdot m$. However, this is rarely optimal. The dimension of $X$, denoted by $r$, is called the **Inner Dimension** (or rank). Finding the smallest possible $r$ is the key to fast matrix multiplication, as we will soon see.

## The Inner Dimension & Tensor Powers

Suppose we have a bilinear algorithm $(\mathbf{U},\mathbf{V},\mathbf{W})$ for $f$ with inner dimension $r$. We can define the **tensor square** of the function, $f^{\otimes 2}: U^{\otimes 2} \times V^{\otimes 2} \to W^{\otimes 2}$, via:
$$ f^{\otimes 2}\left(\sum_i \alpha_i u_i \otimes u_i^{\prime}\ ,\  \sum_j \beta_j v_j \otimes v^{\prime}_j\right) =\sum\_{i,j}\alpha_i\beta_j f(u,v) \otimes f(u',v') $$

**Example:** If $f$ is $2\times 2$ matrix multiplication ($U=V=W=\mathbb{R}^{2\times 2}$), then $\otimes$ corresponds to the Kronecker (outer) product of matrices. By the mixed-product property of Kronecker products, we know that $(A \otimes B)(C \otimes D) = AC \otimes BD$. Thus, $f^{\otimes 2}$ represents $4 \times 4$ matrix multiplication, because every $4\times 4$ matrix can be written as the sum of Kronecker products of $2\times 2$ matrices.

Generalizing this, we define $f^{\otimes k}$ recursively. If $(\mathbf{U}, \mathbf{V}, \mathbf{W})$ computes $f$, does it help us compute $f^{\otimes k}$?
Using the property $(\mathbf{A} \otimes \mathbf{B})(x \otimes y) = \mathbf{A}x \otimes \mathbf{B}y$, one can show that the algorithm for $f^{\otimes 2}$ is given by the Kronecker products of the maps:
$$ (\mathbf{U}^{\otimes 2}, \mathbf{V}^{\otimes 2}, \mathbf{W}^{\otimes 2}) $$
This algorithm operates with inner dimension $r^2$. For the $k$-th power, the inner dimension is $r^k$.

> [!tip] Theorem
> Let $(\mathbf{U}, \mathbf{V}, \mathbf{W})$ be a bilinear algorithm for $f$ with inner dimension $r$.
> Assuming $$r > \max(\dim U, \dim V, \dim W)$$
> the runtime to compute $f^{\otimes k}$ using the algorithm $(\mathbf{U}^{\otimes k}, \mathbf{V}^{\otimes k}, \mathbf{W}^{\otimes k})$ is $O(r^k)$.

**Proof.**
The computation of $f^{\otimes k}(u,v)$ reduces to computing:
1.  $x = \mathbf{U}^{\otimes k} u$
2.  $y = \mathbf{V}^{\otimes k} v$
3.  $z = x \odot y$
4.  $\text{result} = (\mathbf{W}^{\otimes k})^{\top} z$

We must show that the matrix-vector product $\mathbf{U}^{\otimes k} u$ can be computed efficiently. Let $\mathbf{U}$ be an $r \times n$ matrix. The vector $u$ has size $n^k$. We decompose $u$ as a concatenation of $n$ vectors $u^1, \dots, u^n$ of size $n^{k-1}$.
$$
\mathbf{U}^{\otimes k} u = (\mathbf{U} \otimes \mathbf{U}^{\otimes k-1}) u =
\begin{pmatrix}
U_{1,1} \mathbf{U}^{\otimes k-1} & \cdots & U_{1,n} \mathbf{U}^{\otimes k-1} \\\\
\vdots & \ddots & \vdots \\\\
U_{r,1} \mathbf{U}^{\otimes k-1} & \cdots & U_{r,n} \mathbf{U}^{\otimes k-1}
\end{pmatrix}
\begin{pmatrix} u^1 \\\\ \vdots \\\\ u^n \end{pmatrix}
$$
The $i$-th block of the output (for $i \in [r]$) is $\sum_{j=1}^n U_{i,j} (\mathbf{U}^{\otimes k-1} u^j)$.
To compute this:
1.  Recursively compute $z^j = \mathbf{U}^{\otimes k-1} u^j$ for all $j \in [n]$.
2.  Compute the linear combinations $\sum_{j} U_{i,j} z^j$.

Let $T(N)$ be the time to compute $\mathbf{U}^{\otimes k} u$ where $N=n^k$.
The recursive step involves $n$ calls on size $N/n$. The combination step sums $n$ vectors of size $r^{k-1}$ for each of the $r$ output blocks.
$$ T(n^k) = n \cdot T(n^{k-1}) + O(n \cdot r \cdot r^{k-1}) = n \cdot T(n^{k-1}) + O(n r^k) $$
Solving this recurrence (where $r > n$ by assumption) yields $T(n^k) = O(r^k)$.
The same logic applies to $\mathbf{V}$ and $\mathbf{W}$. The element-wise product takes $O(r^k)$. Thus, total time is $O(r^k)$. $\blacksquare$

### Strassen's Algorithm Revisited
We can now explicitly write down the matrices $(\mathbf{U}, \mathbf{V}, \mathbf{W})$ for Strassen's algorithm. Here, the inner dimension is $r=7$, and the vector spaces have dimension 4 (identified with $\mathbb{R}^{2\times 2}$).

$$
\mathbf{U}=\begin{bmatrix}
1 & 0 & 0 & 1 \\\\ 0 & 0 & 1 & 1 \\\\ 1 & 0 & 0 & 0 \\\\
0 & 0 & 0 & 1 \\\\ 1 & 1 & 0 & 0 \\\\ -1 & 0 & 1 & 0 \\\\ 0 & 1 & 0 & -1
\end{bmatrix},
\quad
\mathbf{V}=\begin{bmatrix}
1 & 0 & 0 & 1 \\\\ 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & -1 \\\\
-1 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 1 \\\\ 1 & 1 & 0 & 0 \\\\ 0 & 0 & 1 & 1
\end{bmatrix}
$$
$$
\mathbf{W}=\begin{bmatrix}
1 & 0 & 0 & 1 \\\\ 0 & 0 & 1 & -1 \\\\ 0 & 1 & 0 & 1 \\\\
1 & 0 & 1 & 0 \\\\ -1 & 1 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \\\\ 1 & 0 & 0 & 0
\end{bmatrix}
$$

If you vectorize $2 \times 2$ matrices row-major, meaning $$\left(\begin{smallmatrix} a & b \\\\ c & d  \end{smallmatrix}\right)\mapsto \left(\begin{smallmatrix} a \\\\ b \\\\ c \\\\ d \end{smallmatrix}\right),$$
computing $\mathbf{W}^\top (\mathbf{U} \mathrm{vec}(A) \odot \mathbf{V} \mathrm{vec}(B))$ exactly reconstructs Strassen's logic. This proves (again) that Strassen runs in $O(7^k)$. In particular, taking tensor powers corresponds to a recursive application of the algorithm.

## Another View: Tensors

> [!caution] Definition
> A **3-dimensional tensor** $\mathcal{T}$ is an element of the tensor product space $U \otimes V \otimes W$.
> Fixing bases $\lbrace u_i\rbrace , \lbrace v_j\rbrace , \lbrace w_{\ell}\rbrace $, there is a unique decomposition:
> $$ \mathcal{T} = \sum_{i,j,\ell} T_{i,j,\ell} \cdot (u_i \otimes v_j \otimes w_{\ell}) $$

This looks exactly like the constants $\alpha_{i,j,\ell}$ in our proof of the Strassen Normal Form. To make this intuitive, we can use the **Polynomial View**.
Let $\lbrace X_i\rbrace , \lbrace Y_j\rbrace , \lbrace Z_{\ell}\rbrace $ be sets of formal variables. We can represent a bilinear function $f$ as a polynomial:
$$ P_f(X,Y,Z) = \sum_{i,j,\ell} T_{i,j,\ell} X_i Y_j Z_{\ell} $$
Here, $X_i Y_j Z_{\ell}$ is shorthand for $u_i \otimes v_j \otimes w_{\ell}$.
Now, we can think of $P_f$ as a bilinear function by -
1. Given vectors $u\in U,v\in V$,
2. Expand the vectors in the bases - $u=\sum_i \alpha_i u_i$ and $v=\sum_j \beta_j v_j$,
3. Evaluate $P_f$ at $X_i=\alpha_i$ and $Y_j=\beta_j$, which produces a polynomial in the $Z$ variables,
4. Let $\gamma_{\ell}$ denote the coefficient of $Z_{\ell}$ in the obtained polynomial,
5. Output $\sum_{\ell}\gamma_{\ell}w_{\ell}$.

> [!important] Key Takeaway
> Every bilinear function corresponds uniquely to a 3-tensor (once bases are fixed).

### The Matrix Multiplication Tensor
Consider $n \times n$ matrix multiplication. The spaces correspond to matrices, so we index variables by pairs.
Variables: $\lbrace X_{i,j}\rbrace , \lbrace Y_{i,j}\rbrace , \lbrace Z_{i,j}\rbrace $ for $i,j \in [n]$.
The condition for the product is: $C_{p,q} = \sum_k A_{p,k} B_{k,q}$.
The corresponding tensor, denoted $\langle n,n,n \rangle$, is:
$$ \langle n,n,n \rangle = \sum_{i,j,k} X_{i,k} Y_{k,j} Z_{i,j} $$
In particular, we define $$T_{(i,k),(k',j),(p,q)}=\begin{cases}1 & i=p,j=q,k=k', \\\\
0 & \text{else}.\end{cases}$$

## Tensor Ranks and $\omega$
Note that a vector (called a **tensor**) in $V\otimes U$ might have a simple presentation as a tensor product $v\otimes u$ for $v\in V,u\in U$, but most of the vectors do not have such a presentation. For example $v_1 \otimes u_1 - v_2 \otimes u_2$ cannot be written as a single tensor product of two vectors. A natural question is - what is the minimal number of *simple* terms that we need to present some tensor in $V\otimes U$?
> [!caution] Definition
> The **Rank** of a tensor $\mathcal{T}\in U\otimes V\otimes W$, denoted $R(\mathcal{T})$, is the minimum integer $r$ such that $\mathcal{T}$ can be written as a sum of $r$ simple tensors:
> $$ \mathcal{T} = \sum_{s=1}^r u_s \otimes v_s \otimes w_s $$

Computing the rank of a specific tensor is NP-hard. However, finding upper bounds on the rank of $\langle n,n,n \rangle$ gives us upper bounds on $\omega$.

In particular, suppose we fix some bases $\set{u_i},\set{v_j},\set{w_{\ell}}$ for $U,V,W$, then any simple tensor can be written as $$\left(\sum_{i}\alpha_i u_i\right)\otimes \left(\sum_j \beta_j v_j \right)\otimes \left(\sum_{\ell}\gamma_{\ell}w_{\ell}\right)$$
Hence a tensor $\mathcal{T}$ can be decomposed as $$\sum_{s=1}^r \left(\sum_{i}\alpha_{s,i} u_i\right)\otimes \left(\sum_{s,j} \beta_j v_j \right)\otimes \left(\sum_{\ell}\gamma_{s,\ell}w_{\ell}\right)$$
and writing $$\mathbf{U}=(\alpha_{s,i}),\quad \mathbf{V}=(\beta_{s,j}),\quad \mathbf{W}=(\gamma_{s,\ell})$$
gives us an algorithm for $\mathcal{T}$ with inner dimension $r$. To sum this discussion up:

> [!important] Proposition
> A decomposition of $\mathcal{T}$ into $r$ simple tensors is equivalent to a bilinear algorithm $(\mathbf{U}, \mathbf{V}, \mathbf{W})$ with inner dimension $r$.
> Specifically, there is a correspondence between tensor decompositions and bilinear algorithms for that tensor.

This leads to the fundamental connection between tensors and complexity:

> [!tip] Corollary
> $$ \omega \le \log_n R(\langle n,n,n \rangle) $$

**Proof.**
If $R(\langle n,n,n \rangle) = r$, we have an algorithm for $n \times n$ matrix multiplication with $r$ multiplications. By the recursive theorem from Part 1 (and the tensor power theorem from this post), it follows that $\omega\le \log_n r$. $\blacksquare$

## References
1. **Strassen, V.** (1969). [*Gaussian Elimination is not Optimal*](https://doi.org/10.1007/BF02165411)..
2. **Håstad, J.** (1990). [*Tensor rank is NP-complete*](https://doi.org/10.1016/0196-6774(90)90014-6).
3. **Bürgisser, P., Clausen, M., & Shokrollahi, M. A.** (1997). [*Algebraic Complexity Theory*](https://doi.org/10.1007/978-3-662-03338-8).
4. **Bläser, M.** (2013). [*Fast Matrix Multiplication*](https://doi.org/10.4086/toc.gs.2013.005).
