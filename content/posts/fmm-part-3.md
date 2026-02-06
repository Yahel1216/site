---
title: "Fast Matrix Multiplication - Part 3"
date: 2026-01-15
slug: fmm-3
draft: false
katex: true
description: "Ways to manipulate tensors - the tensor product, direct sum, symmetrization and restriction"
series: "Fast Matrix Multiplication"
tags: ["theory", "matrix-multiplication", "tensors"]
categories: ["Theory", "Matrix Multiplication"]
---
> [!note] Prerequisites
> *   **[Part 2 of this series](/posts/fmm-2):** Familiarity with the tensor product of vector spaces and bilinear algorithms.
> *   **Linear Algebra:** Vector spaces, Bases, Dual spaces, and Tensor Products (basic definition).

In this post, we continue building the foundation for fast matrix multiplication algorithms. We will discuss essential tensor operations—product, sum, and restriction—and establish the Triple Product Condition, setting the stage for the group-theoretic approach.

## Tensor Operations - Tensor Product

Recall the definition of a 3-d tensor as a weighted sum of simple tensors in a space $U\otimes V \otimes W$:
$$
\mathcal{T}=\sum\_{i}\sum\_{j}\sum \_{\ell}T_{i,j,\ell}(u_{i}\otimes v_{j} \otimes w_{\ell})
$$
where $\{u_{i}\},\{v_{j}\},\{w_{\ell}\}$ are bases for $U,V,W$ respectively.

> [!caution] Definition
> The **tensor product** of two 3-d tensors $\mathcal{T}\_1$ (over $U_{1}\otimes V_{1}\otimes W_{1}$) and $\mathcal{T}\_2$ (over $U_{2}\otimes V_{2}\otimes W_{2}$) is a tensor $\mathcal{T}\_{1}\otimes \mathcal{T}\_{2}$ over the space:
> $$ \underbrace{(U_{1}\otimes U_{2})}\_{U}\otimes\underbrace{(V_{1}\otimes V_{2})}\_{V }\otimes\underbrace{(W_{1}\otimes W_{2})}\_{W} $$
> It is defined by the formula:
> $$ \mathcal{T}\_{1}\otimes \mathcal{T}\_{2}=\sum\_{\substack{i,i',j,j'\ell,\ell'}} (T_{i,j,\ell}^{1} \cdot T^{2}\_{i',j',\ell'}) \cdot [(u_{i}\otimes u_{i'})\otimes (v_{j}\otimes v_{j'})\otimes (w_{\ell}\otimes w_{\ell'})] $$
> In other words, it is a **simple** tensor over $6$ vector spaces which we fold to a $3$-d tensor by treating each pair $U_1\otimes U_2$, $V_1\otimes V_2$ and $W_1\otimes W_2$ as the underlying vector spaces (forgetting the fact they too are tensor spaces).

We can apply this to the specific tensors representing matrix multiplication.

> [!important] Lemma
> Let $\langle n,m,p \rangle$ denote the tensor corresponding to the multiplication of $n \times m$ and $m \times p$ matrices. Then:
> $$ \langle n_{1},m_{1},p_{1} \rangle \otimes \langle n_{2},m_{2},p_{2} \rangle \cong \langle n_{1}n_{2},m_{1}m_{2},p_{1}p_{2} \rangle $$

**Proof.**
Our vector spaces are $U_{k}=\mathbb{F}^{n_{k}\times m_{k}}$, $V_{k}= \mathbb{F}^{m_{k} \times p_{k}}$, and $W_{k}=\mathbb{F}^{n_{k}\times p_{k}}$ for $k=1,2$.
First, observe the isomorphism of the tensor product of matrix spaces:
$$ U_{1}\otimes U_{2}=\mathbb{F}^{n_{1}\times m_{1}}\otimes \mathbb{F}^{n_{2}\times m_{2}}\cong \mathbb{F}^{n_{1}n_{2}\times m_{1}m_{2}} $$
This isomorphism is given explicitly by the Kronecker product map:
$$ E_{i,k} \otimes E_{i',k'} \mapsto E_{i n_2 + i', k m_2 + k'} $$
Using double indices for the bases, and recalling the definition of the matrix multiplication tensor, we see that non-zero products of coefficients of both tensors are:
$$
T^{1}\_{(i,k),(k,j),(i,j)}\cdot T^{2}\_{(i',k'),(k',j'),(i',j')}=1\cdot 1$$
Hence the non-zero coefficients of $\langle n_1,m_1,p_1\rangle \otimes \langle n_2,m_2,p_2\rangle$ are the coefficients of $$(E_{i,k}\otimes E_{k,j}\otimes E_{i,j})\otimes (E_{i',k'}\otimes E_{k',j'}\otimes E_{i',j'})\in (U_1\otimes V_1\otimes W_1)\otimes (U_2 \otimes V_2\otimes W_2)$$
which is mapped to $$E_{in_2 +i', km_2 +k'}\otimes E_{km_2 + k', j p_2 + j'} \otimes E_{in_2+i',j p_2 +j'} \in (U_1\otimes U_2)\otimes (V_1\otimes V_2)\otimes (W_1 \otimes W_2)$$
This is exactly the matrix multiplication tensor of size $\langle n_1n_2 ,m_1m_2,p_1p_2\rangle$. $\blacksquare$

The most important property of the tensor product regarding complexity is:

> [!important] Lemma
> Tensor rank is **sub-multiplicative** under tensor products.
> $$ R(\mathcal{T}\_{1}\otimes \mathcal{T}\_{2})\le R(\mathcal{T}\_{1})\cdot R(\mathcal{T}\_{2}) $$

**Proof.**
This follows directly from the definition. If $\mathcal{T}\_1 = \sum\_{r=1}^{R_1} \mathbf{t}^1_r$ and $\mathcal{T}\_2 = \sum\_{s=1}^{R_2} \mathbf{t}^2_s$ are optimal decompositions into simple tensors, then $\mathcal{T}\_1 \otimes \mathcal{T}\_2 = \sum\_{r,s} \mathbf{t}^1_r \otimes \mathbf{t}^2_s$ is a decomposition into $R_1 R_2$ simple tensors. $\blacksquare$

## Permutations and Symmetrization
We wish to discuss certain helpful properties of the matrix multiplication tensor. We start with the following lemma:
> [!important] Lemma (Trace Formula)
> Consider the polynomial $P$ for the matrix multiplication tensor $\langle n,m,p\rangle$. Let $X$ denote the $X_{i,k}$ variables ordered as a matrix, and similarly let $Y,Z$ denote the variable matrices. Then $$\langle n,m,p\rangle =\mathrm{Tr}(XYZ^{\top})$$

**Proof.** By definition $$\mathrm{Tr}(XYZ^{\top})=\sum_{i} [XYZ^{\top}]\_{i,i}=\sum_{i}\sum_{j,k} X_{i,k}Y_{k,j} Z^{\top}\_{j,i}=\sum_{i,j,k}X_{i,k}Y_{k,j}Z_{i,j}=\langle n,m,p\rangle$$


> [!important] Lemma (Permutation Invariance)
> For every permutation $\sigma:[3]\to [3]$ of the dimensions $(n_0, n_1, n_2)$, it holds that:
> $$ R(\langle n_{0},n_{1},n_{2} \rangle)=R(\langle n_{\sigma(0)},n_{\sigma(1)},n_{\sigma(2)} \rangle) $$

**Proof.**
It suffices to prove this for the generators of the permutation group $S_3$: the cyclic shift $(012)$ and the transposition $(02)$. Let $(\mathbf{U},\mathbf{V},\mathbf{W})$ be an algorithm (decomposition) for $\langle n_{0},n_{1},n_{2} \rangle$.

1.  **Cyclic Shift $\sigma=(012)$:** We want to show $R(\langle n_{1},n_{2},n_{0} \rangle) = R(\langle n_{0},n_{1},n_{2} \rangle)$.
Recall that cyclic property of the trace implies $\text{Tr}(ABC) = \text{Tr}(BCA)$. Thus, letting $X$ denote the $n_0\times n_1$ variable matrix, $Y$ denote the $n_1\times n_2$ variable matrix and $Z$ denote the $n_0\times n_2$ variable matrix, we have by the trace formula: $$\langle n_0,n_1,n_2\rangle = \mathrm{Tr}(XYZ^{\top})=\mathrm{Tr}(YZ^{\top}X)$$
Note that $Y$ has shape $n_1\times n_2$, $Z^{\top}$ has shape $n_2\times n_0$ and $X$ has shape $n_0\times n_1$. Therefore by **filling in** the values of an input $n_1\times n_2$ matrix $A$ into $Y$ variables, the values of the second input $n_2\times n_0$ matrix into $Z^{\top}$, we can read the value of $(AB)\_{(i,j)}$ from the coefficient of the variable $X_{j,i}$. More specifically, the formula above computes $(AB)^{\top}$, and moreover provides a bilinear algorithm with the same inner dimension. Note that by permuting the rows of a bilinear algorithm we can deal with a transposed input or output, which is just a change of basis. Thus we have obtained an algorithm for $\langle n_1,n_2,n_0\rangle$ with the same rank.

2.  **Transposition $\sigma=(02)$:** We want to show $R(\langle n_{2},n_{1},n_{0} \rangle) = R(\langle n_{0},n_{1},n_{2} \rangle)$.
    This corresponds to the fact that $(AB)^\top = B^\top A^\top$. Indeed, $\langle n_2, n_1, n_0 \rangle$ represents the multiplication of an $n_2 \times n_1$ matrix by an $n_1 \times n_0$ matrix. By identifying the spaces via the transpose map (swapping row/column indices in the basis), the tensor remains structurally identical.
    Explicitly, define $\widetilde{\mathbf{U}}$ by taking $\mathbf{W}$ and re-ordering rows according to the transpose order, and $\widetilde{\mathbf{W}}$ by taking $\mathbf{U}$ in transpose order. Then $(\widetilde{\mathbf{W}}, \widetilde{\mathbf{V}}, \widetilde{\mathbf{U}})$ is an algorithm for the permuted tensor. $\blacksquare$

This leads to a powerful reduction technique:

> [!tip] Proposition (Symmetrization)
> If $R(\langle n,m,p \rangle) \le r$, then:
> $$ \omega \le \log_{nmp}(r^{3}) = \log_{\sqrt[3]{nmp}}(r) $$
> In the square case $n=m=p$, this recovers the result $\omega \le \log_n r$.

**Proof.**
Suppose $R(\langle n,m,p \rangle) \le r$. By the Lemma above, the rank is invariant under permutations, so $R(\langle m,p,n \rangle) \le r$ and $R(\langle p,n,m \rangle) \le r$.
Using the sub-multiplicative property:
$$
R( \langle n,m,p \rangle \otimes \langle m,p,n \rangle \otimes \langle p,n,m \rangle ) \le r \cdot r \cdot r = r^3
$$
However, the tensor product of these three is isomorphic to $\langle nmp, nmp, nmp \rangle$. Thus:
$$ R(\langle nmp, nmp, nmp \rangle) \le r^3 $$
By a corollary from the previous post, we obtain the upper bound $$ \omega \le \log_{nmp} (R(\langle nmp, nmp, nmp \rangle)) \le \log_{nmp}(r^3) $$
$\blacksquare$

## Tensor Operations - Direct Sum

> [!caution] Definition
> Let $\mathcal{T}\_{1}$ (over $U_{1}\otimes V_{1}\otimes W_{1}$) and $\mathcal{T}\_{2}$ (over $U_{2}\otimes V_{2} \otimes W_{2}$) be tensors. Their **direct sum** $\mathcal{T}\_{1}\oplus \mathcal{T}\_{2}$ is a tensor over $(U_{1}\oplus U_{2})\otimes (V_{1}\oplus V_{2}) \otimes (W_{1} \oplus W_{2})$, given by:
> $$
> \begin{aligned}
> \mathcal{T}\_{1}\oplus \mathcal{T}\_{2} &= \sum_{i,j,\ell} T_{i,j,\ell}^{1}[(u^{1}\_{i} \oplus 0)\otimes (v_{j}^{1} \oplus 0) \otimes (w_{\ell}^{1} \oplus 0)] \\\\
> &+ \sum_{i',j',\ell'} T_{i',j',\ell'}^{2}[(0\oplus u_{i'}^{2})\otimes (0\oplus v_{j'}^{2}) \otimes(0\oplus w_{\ell'}^{2})]
> \end{aligned}
> $$

In words, we embed the variables of $\mathcal{T}_{1}$ and $\mathcal{T}_2$ into disjoint subspaces. The direct sum acts like $\mathcal{T}_1$ on the first subspace and $\mathcal{T}_2$ on the second, with zero interaction between them. The following lemma follows easily from definition:

> [!important] Lemma
> Tensor rank is **sub-additive** under direct sums:
> $$ R(\mathcal{T}\_{1}\oplus \mathcal{T}\_{2})\le R(\mathcal{T\}_{1})+ R(\mathcal{T}\_{2}) $$

**Remark.** Note that $\langle n_{1},m_{1},p_{1} \rangle \oplus \langle n_{2},m_{2},p_{2} \rangle \neq \langle n_{1}+n_{2},m_{1}+m_{2},p_{1}+p_{2} \rangle$. The direct sum corresponds to performing two independent matrix multiplications side-by-side, not multiplying two larger block matrices.

## Tensor Restriction

> [!caution] Definition
> We say that a tensor $\mathcal{T}$ **restricts** to $\mathcal{T}'$, denoted by $\mathcal{T}'\le \mathcal{T}$, if there exist **linear maps** $f_{U}:U\to U'$, $f_{V}:V\to V'$, $f_{W}:W\to W'$ such that:
> $$ (f_{U}\otimes f_{V} \otimes f_{W})(\mathcal{T}) = \mathcal{T}' $$

Recall that given linear functions $f:U\to X,g:V\to Y$ we can define $f\otimes g:U\otimes V\to X\otimes Y$ by defining $$(f\otimes g)(u,v)=(f(u))\otimes (g(v))$$
and noting this definition is bilinear and thus extends to $U\otimes V$ (by the universal property).

> [!important] Lemma
> If $\mathcal{T}'\le \mathcal{T}$, then $R(\mathcal{T}')\le R(\mathcal{T})$.

**Proof.**
If $\mathcal{T} = \sum_{s=1}^r u_s \otimes v_s \otimes w_s$, applying the linear maps element-wise yields:
$$ \mathcal{T}' = \sum_{s=1}^r f_U(u_s) \otimes f_V(v_s) \otimes f_W(w_s) $$
This is a valid decomposition for $\mathcal{T}'$ of size $r$ (though a better one might exist). $\blacksquare$

**Example:**
Consider the tensor over $\mathbb{R}^{3}\otimes \mathbb{R}^{2}\otimes \mathbb{R}^{2}$ given by:
$$ \mathcal{T}=\sum_{i=0}^{2}x_{i}y_{0}z_{0}+\sum_{j=0}^{1}x_{0}y_{i}z_{i} $$
Let $f_{U}$ be the projection that zeros out $x_0$: $f(\alpha_{0} x_{0}+\alpha_{1}x_{1}+\alpha_{2} x_{2})=\alpha_{1}x_{1}+\alpha_{2}x_{2}$. Let $f_{V}, f_W$ be identity maps.
Then $(f_{U}\otimes f_{V} \otimes f_{W})(\mathcal{T}) = x_{1}y_{0}z_{0}+x_{2}y_0z_{0}$.
This result is isomorphic to a tensor in a smaller space, showing we can "restrict" tensors to simpler forms by projecting out variables.

## Triple Product Condition

We finish this post with a precise characterization of the matrix multiplication tensor using the columns of the algorithm matrices.

> [!caution] Definition
> Let $u,v,w \in \mathbb{F}^r$. We define their **triple product** by:
> $$ \langle u,v,w \rangle = \sum_{s=1}^{r} u_{s}v_{s}w_{s} $$

> [!tip] Theorem (Triple Product Condition)
> Let $(\mathbf{U},\mathbf{V},\mathbf{W})$ be a bilinear algorithm with inner dimension $r$. Let the columns of these matrices be indexed by the basis elements of the matrix spaces (e.g., $\mathbf{U}\_{\*, (i,k)}$ is the column corresponding to the matrix entry $A_{i,k}$).
> The algorithm computes Matrix Multiplication $\langle n,m,p \rangle$ if and only if:
> $$ \langle \mathbf{U}\_{\*,(i,k)},\mathbf{V}\_{\*,(k',j)}, \mathbf{W}\_{\*,(i',j')} \rangle = \begin{cases} 1 & i=i' \land k=k' \land j=j' \\\\ 0 & \text{else} \end{cases} $$

**Proof.**
Recall that the tensor for the algorithm $(\mathbf{U},\mathbf{V},\mathbf{W})$ is given by:
$$ \mathcal{T}\_{alg} = \sum_{s=1}^r (\mathbf{u}\_s \otimes \mathbf{v}\_s \otimes \mathbf{w}\_s) $$
where $\mathbf{u}\_s$ are the rows of $\mathbf{U}$. The coefficient of this tensor at the index tuple $((i,k), (k',j), (i',j'))$ is exactly:
$$ \sum_{s=1}^r (\mathbf{U})\_{s, (i,k)} (\mathbf{V})\_{s, (k',j)} (\mathbf{W})\_{s, (i',j')} = \langle \mathbf{U}\_{\*,(i,k)},\mathbf{V}\_{\*,(k',j)}, \mathbf{W}\_{\*,(i',j')} \rangle $$
On the other hand, the definition of the Matrix Multiplication tensor $\langle n,m,p \rangle$ is that the coefficient is $1$ if the indices match $A_{ik} B_{kj} = C_{ij}$ (i.e., $k=k'$, $i=i'$, $j=j'$) and $0$ otherwise.
Equating the coefficients of $\mathcal{T}_{alg}$ and $\langle n,m,p \rangle$ yields the condition. $\blacksquare$


## References
1. **Bürgisser, P., Clausen, M., & Shokrollahi, M. A.** (1997). [*Algebraic Complexity Theory*](https://doi.org/10.1007/978-3-662-03338-8).
2. **Bläser, M.** (2013). [*Fast Matrix Multiplication*](https://doi.org/10.4086/toc.gs.2013.005).