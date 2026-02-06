---
title: "Fast Matrix Multiplication - Part 1"
date: 2026-01-13
slug: fmm-1
draft: false
katex: true
description: "Matrix Multiplication is sub-cubic (Strassen's algorithm)"
series: "Fast Matrix Multiplication"
tags: ["theory", "matrix-multiplication"]
categories: ["Theory", "Matrix Multiplication"]
---

> [!note] Prerequisites
> *   **Linear Algebra:** Basic definitions (matrix multiplication, inner products, block matrices).
> *   **Asymptotic Notation:** Big-O notation and the Master Theorem.
> *   **No prior knowledge** of Group Theory or Representation Theory is required for this post.
***

In this series of posts, we will discuss a fascinating line of work that aims to use group theory and representation theory to design fast algorithms for matrix multiplication. We will start with the basics of matrix multiplication algorithms and build up to the group theoretic approach.
This post is intended for readers without any prior knowledge of groups or representation theory, or of matrix multiplication algorithms. My goal is to build an intuitive understanding of the problem space before we dive deeper in future posts.

## Matrix Multiplication Algorithms
Recall the standard definition of matrix multiplication. Given a matrix $A$ of size $n\times m$ and a matrix $B$ of size $m\times p$, their product $AB$ is an $n\times p$ matrix given by:
$$(AB)\_{i,j}=\sum_{k=1}^{m}A_{i,k}B_{k,j}$$
where $i\in[n]$ and $j\in [p]$. It is often helpful to view this geometrically. Let $A_{i}$ denote the $i$-th row of $A$ (transposed to a column vector) and $B^{j}$ denote the $j$-th column of $B$. Then:

$$(AB)\_{i,j} = \langle A_{i}, B^j \rangle = A_{i}^{\top}B^j$$

This definition immediately yields the "naive" algorithm to compute $AB$:

> [!caution] Algorithm: Naive Matrix Multiplication
> For each entry $(i, j)$ in the output matrix ($n \cdot p$ entries total), compute the inner product between row $i$ of $A$ and column $j$ of $B$.

Each inner product requires $m$ multiplications and $m-1$ additions. Thus, the overall runtime is approximately $2nmp = O(nmp)$. In the square case where $n=m=p$, we obtain the familiar cubic runtime $O(n^{3})$. Specifically, the number of arithmetic operations is:
*   **Multiplications:** $n^{3}$
*   **Additions:** $n^{2}(n-1)$

## Strassen's Algorithm

In 1969, Volker Strassen made a startling discovery: the product of $2\times 2$ matrices can be computed using **7 multiplications** instead of the usual 8, albeit at the cost of performing more additions.

> [!caution] Algorithm: Strassen's $2 \times 2$
> Given matrices:
> $$A=\begin{pmatrix} a & b \\\\ c & d \end{pmatrix},\quad B=\begin{pmatrix} e & f \\\\ g & h \end{pmatrix}$$
> First, compute the following 7 intermediate products:
> $$
> \begin{aligned}
> M_{1}&= (a+d)(e+h) & M_{2}&= (c+d)e \\\\
> M_{3}&= a(f-h)      & M_{4}&= d(g-e) \\\\
> M_{5}&= (a+b)h      & M_{6}&= (c-a)(e+f)\\\\
> M_{7}&= (b-d)(g+h)  & &
> \end{aligned}
> $$
> Then, construct the result matrix $C = AB$ via linear combinations of the $M_i$:
> $$
> C=\begin{bmatrix}
> M_{1}+M_{4}-M_{5}+M_{7} & M_{3}+M_{5} \\\\
> M_{2}+M_{4} & M_{1}-M_{2}+M_{3}+M_{6}
> \end{bmatrix}
> $$

Why is this useful? Why trade one multiplication for several additions? The answer lies in recursion.

### Recursive Application of Matrix Multiplication Algorithms

Suppose we have an algorithm for multiplying $n \times n$ matrices (the "base" size). Given much larger matrices of size $n^2 \times n^2$, we can view them as $n \times n$ block matrices, where each element is itself a smaller sub-matrix of size $n \times n$.

Let $\mathbf{A}, \mathbf{B}$ be $n^2 \times n^2$ matrices partitioned into $n \times n$ blocks. 
Define the $n\times n$ blocks by  $A^{I,J}=[\mathbf{A}\_{I\cdot n+i,J\cdot n+j}]\_{i,j\in [n]}$ for $I,J\in [n]$. Similarly define $B^{I,J}$. Then $$\mathbf{A}=\left(\begin{smallmatrix} A^{0,0} & \cdots & A^{0,n-1} \\\\ \vdots  & \ddots  \\\\ A^{n-1,0} & \cdots & A^{n-1,n-1} \end{smallmatrix}\right),\quad \mathbf{B}=\left(\begin{smallmatrix} B^{0,0} & \cdots & B^{0,n-1} \\\\ \vdots  & \ddots \\\\ B^{n-1,0} & \cdots & B^{n-1,n-1} \end{smallmatrix}\right).$$
It is an easy observation that the product block satisfies:
 $$(\mathbf{AB})^{I,J}=\sum_{K=0}^{n-1}A^{I,K}B^{K,J}.$$
 Crucially, the product $A^{I,K}B^{K,J}$ represents a matrix multiplication of the smaller blocks, and the summation represents element-wise matrix addition.
 
 > [!note] Remark
 > For any **ring** $R$, one can define the matrix ring $\mathcal{M}_n(R)$ of square matrices over $R$. In particular, the observations above show that $$\mathcal{M}\_n (\mathcal{M}\_n (\mathbb{F}))\cong \mathcal{M}\_{n^2}(\mathbb{F}).$$
> On the left hand side, matrices over the ring $R=\mathcal{M}_n(\mathbb{F})$ have the same multiplication law, with scalar multiplication replaced by multiplication in $R$, that is $n\times n$ block matrix multiplication.

If our base algorithm for size $n$ uses $M$ scalar multiplications and $L$ scalar additions, we can lift this to the block setting. The algorithm works **if we replace scalar by block matrix operations**. Therefore to compute a $n^2\times n^2$ product, we need to:
1.  Perform $M$ **matrix multiplications** of size $n\times n$. Each product can be computed **recursively** using the same algorithm, costing $M$ scalar computations and $L$ additions.
2.  Perform $L$ **matrix additions** of size $n\times n$. Each addition requires computing exactly $n^2$ scalar additions.

Therefore the total operation count is $$M\cdot (M+L) + n^2\cdot L=M^2 + (M+n^2)\cdot L.$$
This discussion can be generalized to any power of $n$. To multiply $n^k\times n^k$ matrices, we apply the algorithm recursively $k$ times (the depth of the recursion). Let $T(k)$ denote the time to multiply $n^k\times n^k$ matrices using the algorithm. Then the recurrence relation is:
$$
T(k) = M \cdot T(k-1) + L \cdot (n^{k-1})^2.
$$

Solving this recurrence with the Master Theorem yields a runtime of:
$$
T(k)=O(M^{k})
$$
provided $M > n^2$. In other words, the runtime is **dominated by the number of multiplications**, while the additions only affect the constant factors.

**Example:** Lets apply this to Strassen's algorithm, where the base size is $n=2$ and the number of multiplications is $M=7$. We can compute matrices of size $2^{k}\times 2^{k}$ in time $O(7^{k})$. This is significantly faster than the naive $O(2^{3k})=O(8^{k})$.

Writing $N=2^{k}$, we have $k = \log_2 N$, and the complexity to multiply $N\times N$ matrices becomes:
$$
O(7^{\log_2 N}) = O(N^{\log_{2}7}) \approx O(N^{2.807})
$$
This result proved that matrix multiplication is **sub-cubic**.

## The Main Question in FMM

Given Strassen's breakthrough, we know that the naive cubic algorithm is not optimal. The central pursuit in the field of Fast Matrix Multiplication (FMM) is determining just how fast we can go.

Formally, we define the exponent of matrix multiplication, $\omega$:

> [!caution] Definition
> $$
> \omega = \inf \lbrace{ \tau \le 3 \mid \forall \varepsilon > 0, \text{ there is an algorithm for } n \times n \text{ MM with runtime } O(n^{\tau+\varepsilon}) \rbrace}
> $$

The main open question is: **What is the precise value of $\omega$?** (It is conjectured that $\omega = 2$).

The recursive argument we utilized for Strassen's algorithm can be generalized into the following fundamental theorem regarding $\omega$:

> [!tip] Theorem
> If there exists an algorithm for computing an $n_{0}\times n_{0}$ matrix multiplication using $M$ scalar multiplications, then:
> $$ \omega \le \log_{n_{0}}M $$

**Proof.**
Let $n \in \mathbb{N}$ be arbitrary, and let $A, B$ be $n \times n$ matrices. We must show we can multiply them in time $O(n^{\log_{n_0} M+\varepsilon})$ for every $\varepsilon>0$.
Define $k$ to be the minimal integer such that $n_{0}^{k} \ge n$, and let $N=n_{0}^{k}$. We construct $N \times N$ matrices $\widetilde{A}, \widetilde{B}$ by embedding $A$ and $B$ into the upper-left $n \times n$ block and padding with zeros:
$$\widetilde{A}=\begin{pmatrix}A & 0 \\\\ 0 & 0\end{pmatrix}.$$

We can recursively apply the base algorithm to multiply $\widetilde{A}$ and $\widetilde{B}$. Based on our previous analysis, the runtime is:
$$ O(M^{k}) = O(N^{\log_{n_{0}}M}) $$

The $n \times n$ upper-left block of the result $\widetilde{A}\widetilde{B}$ is exactly $AB$.

Since we chose the minimal $k$, we have $n_{0}^{k-1} < n \le n_{0}^{k}$. Treating $n_0$ as a constant, this implies $\log_n N =\frac{k}{\log_{n_0} n}$ which by definition is between $1$ and $1+\frac{1}{k-1}$. Taking $k_0$ to be large enough so that $\frac{1}{k-1}\cdot \log_{n_0}M<\varepsilon$, we see that for $n=\Omega(n_0^{k_0})$ it holds $O(N^{\log_{n_0}M})=O(n^{\log_n N\cdot \log_{n_0}M})=O(n^{\log_{n_0}M+\varepsilon})$, thus concluding the proof. $\blacksquare$

***

## References
1.  **Strassen, V.** (1969). [*Gaussian Elimination is not Optimal*](https://link.springer.com/article/10.1007/BF02165411).
2.  **Cohn, H., & Umans, C.** (2003). [*A Group-theoretic Approach to Fast Matrix Multiplication*](https://arxiv.org/abs/math/0307321).