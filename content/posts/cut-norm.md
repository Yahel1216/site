---
title: "Approximating the Cut-Norm - Part 1"
date: 2026-01-12
slug: cut-norm
draft: false
katex: true
series: "Cut-Norm"
description: "Using Grothendick's identity to obtain an approximate algorithm for computing the Cut-Norm of a matrix"
tags: ["optimization", "semi-definite-programming"]
categories: ["Optimization", "Complexity"]
---
> [!note] Prerequisites
> *   **Linear Algebra:** Eigenvalues, PSD matrices, Tensor products.
> *   **Convex Optimization:** Basic familiarity with Semidefinite Programming (SDP).
> *   **Probability:** Expectations, Markov's inequality.
> *   **Graph Theory:** Basic definitions, Cuts, Regularity.

Consider the following problem: Given an undirected graph $G=(V,E)$, let $A,B\subset V$ denote non-empty disjoint sets. Let $E(A,B)$ denote the set of edges in $E$ that cross from $A$ to $B$. Denote $D(A,B)=\frac{\left|E(A,B)\right|}{\left|A\right|\left|B\right|}$ to be the **density** of edges crossing from $A$ to $B$.

Regularity, in the context of graphs, can describe some notion of pseudo-random structure. The idea of *regularity decomposition* of graphs is to decompose a graph into sub-graphs which are regular in some sense. This allows us to prove results regarding general graphs by reducing the problem to random-like graphs, which are often easier to analyze and reason about. There is, generally, a tradeoff between the strength of the regularity notion and the efficiency of the decomposition under that notion (efficiency being the number of sub-graphs).

One notion of regularity is the following $\varepsilon$-regularity of sets $(A,B)$ as above: we call $(A,B)$ $\varepsilon$-regular if for every $X\subset A,Y\subset B$ with $\left|X\right|\ge \varepsilon \left|A\right|$ and $\left|Y\right|\ge \varepsilon \left|B\right|$, the density $D(X,Y)$ is $\varepsilon$-close to the density $D(A,B)$, meaning $$\forall X\subset A,Y\subset B, |X|\ge \varepsilon\cdot |A|,| Y\ge \varepsilon\cdot |B|:\quad \left|D(X,Y)-D(A,B)\right|\le \varepsilon$$
In other words, the edges crossing between $A,B$ are not concentrated on a small subset of vertices.

Suppose we wish to determine if a pair $(A,B)$ is $\varepsilon$-regular, assuming $\left|A\right|=\left|B\right|=n$. We could denote $d=D(A,B)$ and define a matrix $F=(f_{ab})\_{a\in A,b\in B}$ by setting
$$f_{ab}=\begin{cases}
1-d & \lbrace a,b\rbrace \in E \\\\
-d & \text{else.}
\end{cases}$$
Note that
$$\left|\sum_{a\in X,b\in Y}f_{ab}\right|=\left|\sum_{(a,b)\in E(X,Y)}1-d\cdot \left|X\right|\cdot \left|Y\right|\right|=\left|\left|E(X,Y)\right|-d\cdot |X||Y|\right|=\left|X\right|\cdot \left|Y\right|\cdot \left|D(X,Y)-d\right|.$$
Hence, if the pair is **not** $\varepsilon$-regular, then there are $X\subset A,Y\subset B$ with $\left|X\right|, \left|Y\right|\ge \varepsilon n$ for which
$$\left|D(X,Y)-D(A,B)\right|=\left|D(X,Y)-d\right|> \varepsilon\iff \left|\sum_{a\in X,b\in Y}f_{ab}\right|> \varepsilon \left|X\right|\cdot \left|Y\right|\ge \varepsilon^{3}n^{2}.$$
This motivates the following definition:

> [!caution] Definition: Cut Norm
> The cut norm of a matrix $A=(a_{ij})\_{i\in R,j\in S}$ indexed by $R,S$ is
> $$\left\lVert A \right\rVert_{C}:=\max_{I\subset R,J\subset S}\left|\sum_{i\in I,j\in J}a_{ij}\right|.$$

So a pair $(A,B)$ is not $\varepsilon$-regular if and only if $\left\lVert F \right\rVert_{C}\ge \varepsilon^{3}n^{2}$. Therefore, irregularity in graphs can be detected using the cut-norm. The question then becomes - is there an algorithm that efficiently computes the cut norm?

#### Application: Cut-Decomposition
The cut-decomposition of a graph is another motivating example for the need of an efficient algorithm to compute the cut-norm. For many graph problems, it is useful to find a **cut-decomposition** for the graph, which involves finding matrices $D^{(1)},\ldots, D^{(k)}$ of a special structure which approximate a matrix $A$ (which is often based on the adjacency matrix of the graph) in the **cut-norm**. The matrices $D^{(i)}$ are restricted to be **cut** matrices, which are of the form $d_{i}\cdot 1_{I}1_{J}^{\top}$, where $I\subset[n],J\subset [m]$ assuming $A$ has size $n\times m$ and $d_{i}$ is some scalar. Such a computation can be performed rather fast (polynomial time, also in the accuracy) if one has access to a good approximation of the cut norm.

## Hardness of Approximation of the Cut-Norm
> [!caution] Definition
> An algorithm is called a $C$-approximation algorithm if it delivers, on **expectation**, a solution whose objective value is $C$ times the optimum. In particular, this is relevant only for optimization problems.
>
> A problem is called MAX-SNP hard if there is no polynomial time algorithm that approximates the problem with arbitrary precision. In other words, there exists some constant $\rho$, so the existence of a $\rho$-approximation algorithm implies P equals NP.

One famous MAX-SNP problem is MAX-CUT, in which we are given a graph $G=(V,E)$ and are asked to find a cut $(A,B)$ (so $A\cap B=\emptyset,A\cup B=V$) for which $\left|E(A,B)\right|$ is maximal. The hardness of computing the cut-norm will be proved by a reduction from MAX-CUT.

### Moving to an Equivalent Norm
> [!caution] Definition: $\infty \mapsto 1$ Norm
> Let $A$ be a matrix $(a_{ij})\_{i\in [n],j\in [m]}$, then we define the $\infty\mapsto 1$ norm by
> $$\left\lVert A \right\rVert_{\infty\mapsto 1}:=\max_{x\in \lbrace \pm1\rbrace ^{n},y\in \lbrace \pm1\rbrace ^{m}}\sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}\cdot x_{i}y_{j}.$$

> [!important] Lemma
> For every real matrix it holds $\left\lVert A \right\rVert_{C}\le \left\lVert A \right\rVert_{\infty\mapsto 1}\le 4 \left\lVert A \right\rVert_{C}$.

**Proof.** Fix $x,y$, then
$$\sum_{i}\sum_{j}a_{ij}x_{i}y_{j}=\sum_{i,j:x_{i}=y_{j}=1}a_{ij}-\sum_{i,j:x_{i}=-y_{j}=1}a_{ij}-\sum_{i,j:x_{i}=-y_{j}=-1}a_{ij}+\sum_{i,j:x_{i}=y_{j}=-1}a_{ij}.$$
Each of the sums corresponds to taking $I=\lbrace i:x_{i}=\alpha\rbrace $ and $J=\lbrace j:y_{j}=\beta\rbrace $ for $(\alpha,\beta)\in \lbrace \pm1\rbrace ^{2}$, and so each of the sums is in absolute value less than or equal to $\left\lVert A \right\rVert_{C}$, thus giving the upper bound.

For the lower bound, let $I,J$ denote the sets achieving the maximum. Suppose $\left\lVert A \right\rVert_{C}=\sum_{i\in I,j\in J}a_{ij}$. Define $x_{i}=y_{j}=1$ for $i\in I,j\in J$ and $-1$ otherwise. Then $\frac{x_i+1}{2}\cdot \frac{y_j+1}{2}=1$ if $i\in I,j\in J$ and $0$ otherwise. Therefore:
$$\left\lVert A \right\rVert_{C}=\sum_{i\in I,j\in J}a_{ij}=\sum_{i,j}a_{ij}\cdot \frac{x_{i}+1}{2} \cdot \frac{y_{j}+1}{2}=\frac{1}{4}\sum_{i,j}a_{ij}+\frac{1}{4}\sum_{i,j}a_{ij}x_{i}+\frac{1}{4}\sum_{i,j}a_{ij}y_{j}+\frac{1}{4}\sum_{i,j}a_{ij}x_{i}y_{j}.$$
Hence $\left\lVert A \right\rVert_{C}$ is the average of four different assignments of $x,y$, which are all smaller than $\left\lVert A \right\rVert_{\infty\mapsto 1}$, thus their average is too, concluding that $\left\lVert A \right\rVert_{C}\le \left\lVert A \right\rVert_{\infty\mapsto 1}$. $\blacksquare$

> [!note] Remark
> If $A$'s rows and columns all sum to zero, then the first three sums above are zero, thus showing that $\left\lVert A \right\rVert_{C}\le \frac{1}{4}\left\lVert A \right\rVert_{\infty\mapsto 1}\le \frac{1}{4}\cdot 4\left\lVert A \right\rVert_{C}=\left\lVert A \right\rVert_{C}$. Hence $\left\lVert A \right\rVert_{\infty\mapsto 1}=4 \left\lVert A \right\rVert_{C}$.

### Reducing from MAX-CUT
> [!important] Proposition
> Let $G=(V,E)$. Then there exists an efficient construction of a matrix $A$ of size $2\left|E\right|$ by $\left|V\right|$, such that $\mathrm{MAXCUT}(G)=\left\lVert A \right\rVert_{C}=\frac{1}{4}\left\lVert A \right\rVert_{\infty\mapsto 1}$. This proves that MAX-CUT reduces to computing the cut-norm.

**Proof.** Define $A$ as follows: for every edge $\lbrace u,v\rbrace $, we define two rows in $A$, one for $(u,v)$ and one for $(v,u)$. The first one is defined by $e_{u}-e_{v}$ and the second $e_{v}-e_{u}$, where $e_{i}$ is the $i$-th standard basis vector for $\mathbb{R}^{\left|V\right|}$. Note that rows and columns of $A$ sum to zero. Therefore $\left\lVert A \right\rVert_{C}= \frac{1}{4}\left\lVert A \right\rVert_{\infty\mapsto 1}$.

Consider
$$\left\lVert A \right\rVert_{\infty\mapsto 1}=\max_{x\in \lbrace\pm 1\rbrace^{2|E|} ,y\in \lbrace \pm 1\rbrace^{|V|}}\sum_{(u,v),w} a_{(u,v),w}x_{(u,v)}y_{w}=\max_{x,y}\sum_{(u,v)}(x_{(u,v)}y_{u}-x_{(u,v)}y_{v}).$$
Note that if $y_{u}=y_{v}$ then $x_{(u,v)}y_{u}-x_{(u,v)}y_{v}=0$ and the same goes for $x_{(v,u)}y_{v}-x_{(v,u)}y_{u}=0$. Alternatively, if $y_{u}\ne y_{v}$, then
$$\left|x_{(u,v)}(y_{u}-y_{v})\right|=\left|x_{(v,u)}(y_{v}-y_{u})\right|=2,$$
and since $x_{(u,v)},x_{(v,u)}$ don't participate in any other elements of the sum, their contribution to the sum is maximized by choosing the signs of the expression $y_u-y_v$ and $y_v-y_u$ respectively. In other words, the sum is maximized by
$$\left\lVert A \right\rVert_{\infty\mapsto 1}=\max_{y}4\sum_{\lbrace u,v\rbrace \in E}\mathbf{1}\_{y_{u}\not=y_{v}},$$
and thus we can treat $y$ as the partition rule setting $A=\lbrace u:y_u=1\rbrace $ and $B=\lbrace u:y_u=-1\rbrace $, showing that $\left\lVert A \right\rVert_{\infty\mapsto 1}=4\cdot \left|E(A,B)\right|=4\cdot\mathrm{MAXCUT}(G)$ by maximality (any valid cut induces a valid assignment for $y$). $\blacksquare$

For MAX-CUT it is known that if P is not equal to NP, there is no polynomial time approximation with ratio exceeding $16/17$, so this is also an upper bound for the ratio of approximation for the Cut Norm problem.

## A Deterministic Approximation Algorithm

### Casting the Problem as an SDP

> [!note] Reminder
> A semi-definite program (SDP) is a constrained optimization problem of the following form:
> $$\min_{x\in\mathbb{R}^{n}}c^{\top}x\quad \text{subject to}\quad x_{1}A_{1}+\ldots+x_{n}A_{n}\preceq B,\quad Ax=b,$$
> where $c\in\mathbb{R}^{n}$ and $B,A_{1},\ldots,A_{n}$ are **symmetric** matrices of size $k\times k$ for some $k$, and $b\in \mathbb{R}^m,A\in \mathbb{R}^{m\times n}$ are the linear constraints. We denote $A\preceq B$ when $B-A$ is a **positive semi-definite** matrix.

Note that the value $\left\lVert A \right\rVert_{\infty\mapsto 1}$ can be written as the solution to the following **integer** quadratic program:
$$\max \ \sum_{i,j}a_{ij}x_{i}y_{j}\quad \text{subject to}\quad x_i,y_{j}\in \lbrace \pm1\rbrace .$$
As always when dealing with integer programs, we can try to **relax** the integer constraints to obtain easier problems. Hopefully, if the problem is nice enough, we can then **round** the solutions of the relaxed problems to integral solutions, without too much error.

In this case, one such way to relax the problem is by replacing the scalars $x_i,y_j$ with **vectors** $u_i,v_j$ (of potentially higher dimension), and so the constraint $x_i,y_j\in \lbrace \pm1 \rbrace$ becomes $\\| u_i \\|=\\|v_j\\| =1$:
$$\max \sum_{i,j}a_{ij}\cdot \left\langle u_{i},v_{j} \right\rangle \quad \text{subject to}\quad \left\lVert u_{i} \right\rVert^2=\left\lVert v_{j} \right\rVert^2=1.$$

This type of problem is called a **Quadratically Constrained Quadratic Program**. In general, it is as hard as integer programming, and cannot be solved efficiently. However, in the special case above, we are not really using the vectors, but only use their inner products (because $\\|u\\|^2= \langle u,u\rangle$).

> [!important] Lemma
> The vector programming relaxation above can be solved using an SDP.

**Proof.** Let $p=n+m$ and index the vectors as $w_1,\ldots,w_p$ where the first $n$ are $u_i$ and the rest are $v_j$. The Gram matrix is defined by $$G=(\langle w_i, w_j\rangle)\_{i,j\in [p]}$$
It is a symmetric positive-definite matrix, by properties of the real inner product. Therefore, it is uniquely determined by $p(p+1)/2$ values (the main diagonal + the upper triangle). Note that $$\sum_{i\in [n],j\in [m]} a_{ij}\cdot \langle u_i ,v_j\rangle = \sum_{i\in [n],j\in [m]}a_{ij}\cdot G_{i,n+j}$$ and the constraints can be written as $\\| u_i\\|^2=G_{i,i}=1$ (similarly for $v_j$). This motivates the following scalar program: for $x\in \mathbb{R}^{p(p+1)/2}$ we can index the elements of $x$ by pairs $(i,j)$ where $1\le i\le j\le p$. Let $G(x)$ denote the symmetric matrix determined by $x$. Write: $$\max_{x\in \mathbb{R}^{p(p+1)/2}} \sum_{i\in [n], j\in [m]}a_{ij}\cdot x_{i,n+j} \quad \text{subject to}\quad x_{i,i}=1,\quad G(x)\succeq 0.$$

Given a solution to the vector problem, we see that setting $x_{i,j}=G_{i,j}$ gives the same objective value in the scalar program (and is feasible, because $G_{i,i}=1$ and $G\succeq 0$), hence the vector maximum value (denoted $V^\*$) is at most the scalar maximum value (denoted $S^\*$), i.e., $V^\*\le S^\*$. 

Conversely, given a solution $x$ for the scalar problem, the matrix $G(x)$ is positive semi-definite and symmetric, thus by the spectral theorem (from linear algebra) it can be written as $U\Lambda U^{\top}$ for an orthogonal matrix $U$ and diagonal matrix $\Lambda$. Take $w_i=(\sqrt{\lambda_j} U_{i,j})\_{j\in [p]}$ where $\lambda_1,\ldots,\lambda_p\ge 0$ are the eigenvalues on the diagonal of $\Lambda$. Then $(G(x))\_{i,j}=w_i^{\top}w_j=\langle w_i,w_j\rangle$, hence a solution to scalar problem translates to a solution to the vector problem, so $S^{\*}\le V^\*$. This concludes the equality $S^\*=V^\*$.

To see the scalar problem is an SDP is easy. The objective is clearly linear in $x$ and the equality constraints are also linear in $x$. We just need to write $G(x)$ as a sum of matrices: $$G(x)=\sum_{1\le i< j\le p} x_{i,j}\cdot (E_{i,j} +E_{j,i}) +\sum_{1\le i\le p} x_{i,i} E_{i,i}$$
where $E_{i,j}$ is the unit matrix with $1$ in the $(i,j)$-th position and zero everywhere else. $\blacksquare$

**What have we achieved so far:** Assuming we solved the relaxed vector problem using the scalar version SDP, we arrive at vectors $u_i,v_j\in \mathbb{R}^p$ (see the previous proof), that give a $\delta$-approximation (for some $\delta>0$) of the relaxed objective, which is at least $\left\lVert A \right\rVert_{\infty\mapsto 1}$. We can choose $\delta$ to be negligible compared to the entries of $A$. Now we need to round the result to obtain an approximation to the integer program. Note that $\langle u_i ,v_j\rangle$ is any real number (by Cauchy-Schwarz it is in $[-1,1]$), not necessarily $\pm1$ as required.

### Orthogonal Arrays
An orthogonal array is just a sample space with limited independence between values.

> [!caution] Definition
> A set of vectors $V$ is called an orthogonal array for $k$ of strength $4$, every vector $\varepsilon\in V$ is a sign vector $\varepsilon\in \lbrace \pm1\rbrace ^{k}$ in which the values of the vector are $4$-wise independent, and chosen randomly from $\lbrace \pm1\rbrace $. More precisely, for every quadruple of coordinates $1\le i_{1}<i_{2}<i_{3}<i_{4}$ and every choice of $\alpha\in \lbrace \pm1\rbrace ^4$, exactly $1/16$ of the vectors in $V$ have $\alpha_{j}$ in coordinate $i_{j}$ for $j\in \lbrace 1,2,3,4\rbrace $.

We can treat $V$ as a probability space with the uniform probability. For $\\| q\\|=1$ define $h(q)$ as the random variable given by $\langle \varepsilon,q\rangle$ where $\varepsilon$ is randomly drawn from $V$. Take $V$ to have size $t=O(p^2)$ and $k=p$.

> [!important] Lemma
> It holds $\mathbb{E}[h(q)\cdot h(q')]=\left\langle q,q' \right\rangle$. In particular, $\mathbb{E}[h(q)^{2}]=1$ since $\left\lVert q \right\rVert=1$. Moreover, $\mathbb{E}[h(q)^{4}]\le 3$.

**Proof.** We have
$$\mathbb{E}[h(q)\cdot h(q')]=\sum_{i,j}\mathbb{E}\_{\varepsilon\in V}[q_{i}\varepsilon_{i}q_{j}'\varepsilon_{j}]=\sum_{i,j}q_{i}q_{j}'\cdot \mathbb{E}\_{\varepsilon\in V}[\varepsilon_{i} \varepsilon_{j}]=\sum_{i}q_{i}q_{i}'=\left\langle q,q' \right\rangle,$$
using the fact $\varepsilon_{i},\varepsilon_{j}$ are independent if $i\not=j$ and their expectation is $0$. Moreover, 4-wise independence also implies
$$\mathbb{E}[h(q)^{4}]=\sum_{i,j,k,l}q_{i}q_{j}q_{k}q_{l}\cdot \mathbb{E}[\varepsilon_{i} \varepsilon_{j} \varepsilon_{k} \varepsilon_{l}]=\sum_{i}q_{i}^{4}+\binom{4}{2}\sum_{j<k}q_{j}^{2}q_{k}^{2}\le 3\cdot\left(\sum_{i}q_{i}^{2}\right)^{2}=3.$$ $\blacksquare$

Define the $M$-truncation $h^{M}(q)$ to be
$$h^{M}(q)=\begin{cases}
h(q) & \left|h(q) \right|\le M, \\\\
M & h(q)>M, \\\\
-M & h(q)<-M.
\end{cases}$$
For any $m\in\mathbb{R}$, Markov's inequality says
$$\Pr(\left|h(q)\right|\ge m)=\Pr(\left|h(q)\right|^{4}\ge m^{4})\le \frac{\mathbb{E}[\left|h(q)\right|^{4}]}{m^{4}}\le \frac{3}{m^{4}}.$$
Note that $$\mathbb{E}[\left|h(q)-h^{M}(q)\right|^{2}]=0\cdot \Pr(\left|h(q)\right|\le m)+\mathbb{E}\left[\left|h(q)-h^{M}(q)\right|^{2}\mid \left|h(q)\right|> M\right]\cdot \Pr(\left|h(q)\right|>M).$$
Conditioned on $\left|h(q)\right|>M$ we have
$$\mathbb{E}[(h(q)-h^{M}(q))^{2}]=\mathbb{E}[h(q)^{2}]-2M \cdot \mathbb{E}[h(q)]+M^{2}=1+M^{2},$$
and so
$$ \mathbb{E}[\left|h(q)-h^{M}(q)\right|^{2}]\le (1+M^{2})\cdot \frac{3}{M^{4}}\le 1/M.$$
Define $H(q)\in\mathbb{R}^{t}$ to be the vector given by $H(q)\_{\varepsilon}=\frac{1}{\sqrt{t}}h(q)(\varepsilon)=\frac{1}{\sqrt{t}}\left\langle q,\varepsilon \right\rangle$, for every $\varepsilon\in V$. Similarly, define the truncation $H^{M}(q)$ to be the scaled truncation of $h^{M}(q)$.

> [!important] Lemma
> For every unit vector $q\in\mathbb{R}^p$, the vector $H(q)\in\mathbb{R}^{t}$ is also a unit vector. The norm of $H^{M}(q)$ is at most $1$, that of $H(q)-H^{M}(q)$ is at most $1/M$. If $q'\in\mathbb{R}^{p}$ is another vector then $\left\langle H(q),H(q') \right\rangle=\left\langle q,q' \right\rangle$.

**Proof.** We have $\left\lVert H(q) \right\rVert^{2}=\frac{1}{t}\sum_{\varepsilon\in V}\left\langle q,\varepsilon \right\rangle^{2}=\mathbb{E}[h(q)^{2}]=\langle q,q\rangle =1$. Moreover, $\left|H^{M}(q)\_{i}\right|\le \left|H(q)\_{i}\right|$ for every $i$ by definition, so $\left\lVert H^{M}(q) \right\rVert\le \left\lVert H(q) \right\rVert=1$. Similarly,
$$\left\lVert H(q)-H^{M}(q) \right\rVert^{2}=\frac{1}{t}\sum_{\varepsilon\in V}\left|\left\langle q,\varepsilon \right\rangle-T_{M}(\left\langle q,\varepsilon \right\rangle)\right|^{2}=\mathbb{E}[\left|h(q)-h^{M}(q)\right|^{2}]\le 1/M.$$
Finally,
$$\left\langle H(q),H(q') \right\rangle=\frac{1}{t} \sum_{\varepsilon\in V}(\left\langle q,\varepsilon \right\rangle \cdot \left\langle q',\varepsilon \right\rangle)=\mathbb{E}[h(q)\cdot h(q')]=\left\langle q,q' \right\rangle.$$ $\blacksquare$

**What have we achieved so far:** A tool based on $\pm1$-valued vectors, that allows us to compute inner products of unit vectors using inner products with sign vectors, to a good degree of accuracy (in the truncation functions).

### The Rounding Procedure
Given solutions $u_{i},v_{j}\in\mathbb{R}^{p}$ that achieve a $\delta$ approximation of the SDP, whose optimal value we denote by $B$, it follows that
$$B-\delta\le \sum_{ij}a_{ij} \cdot \left\langle u_{i},v_{j} \right\rangle=\sum_{ij}a_{ij}\left\langle H(u_{i}),H(v_{j}) \right\rangle.$$
Note that $$\left\langle H(u_{i}),H(v_{j}) \right\rangle=\left\langle H^{M}(u_{i}),H^{M}(v_{j}) \right\rangle+\left\langle (H(u_{i})-H^{M}(u_{i})),H^{M}(v_{j}) \right\rangle+\left\langle H(u_{i}),(H(v_{j})-H^{M}(v_{j})) \right\rangle.$$
By Cauchy-Schwartz, using that fact that the norms of $H^{M}(v_{j}),H(u_{i})$ are at most $1$ and the norm of $H(u_{i})-H^{M}(u_{i}),H(v_{j})-H^{M}(v_{j})$ are at most $1/M$, we obtain
$$\left\langle H(u_{i}),H(v_{j}) \right\rangle\le \left\langle H^{M}(u_{i}),H^{M}(v_{j}) \right\rangle+\frac{2}{M}.$$
Thus
$$B-\delta\le \sum_{ij}a_{ij}\left\langle H^{M}(u_{i}),H^{M}(v_{j}) \right\rangle+\sum_{ij}a_{ij}\cdot \frac{2}{M}\le \sum_{ij}a_{ij}\left\langle H^{M}(u_{i}),H^{M}(v_{j}) \right\rangle+\frac{2B}{M},$$
where the second inequality follows from the fact $\sum_{ij}a_{ij}\le B$ due to the maximality of $B$. This proves
$$B\left(1-\frac{2}{M}\right)-\delta\le \sum_{ij}a_{ij}\left\langle H^{M} (u_{i}),H^{M}(v_{j}) \right\rangle.$$

> [!important] Lemma
> There exists a vector $\varepsilon\in V$ for which
> $$\sum_{ij}a_{ij}H^{M}(u_{i})\_{\varepsilon}\cdot H^{M}(v_{j})\_\varepsilon \ge \frac{1}{t}\cdot \left[B\left(1-\frac{2}{M}\right)-\delta\right].$$

**Proof.** Suppose otherwise, then
$$\begin{aligned}
\sum_{ij}a_{ij}\left\langle H^{M}(u_{i}),H^{M}(v_{j}) \right\rangle &= \sum_{ij}a_{ij}\sum_{\varepsilon\in V}H^{M}(u_{i})\_\varepsilon\cdot H^{M}(v_{j})\_\varepsilon\\\\
&= \sum_{\varepsilon\in V}\sum_{ij}a_{ij}H^{M}(u_{i})\_\varepsilon \cdot H^{M}(v_{j})\_\varepsilon \\\\
&< \sum_{\varepsilon\in V}\frac{1}{t}[B(1-2/M)-\delta]=B(1-2/M)-\delta,\end{aligned}$$
contradicting the previous inequality. $\blacksquare$

For such $\varepsilon$ we have by definition of $H^{M}$ and $h^{M}$ that
$$\sum_{ij}a_{ij}h^{M}(u_{i})(\varepsilon)\cdot h^{M}(v_{j})(\varepsilon)\ge B\left(1-\frac{2}{M}\right)-\delta.$$
Thus choosing $M=3$ and letting $x_{i}=\frac{h^{M}(u_{i})(\varepsilon)}{M}$ and $y_{j}=\frac{h^{M}(v_{j})(\varepsilon)}{M}$, we obtain that $\left|x_{i}\right|,\left|y_{j}\right|\le 1$ and
$$\sum_{ij}a_{ij}x_{i}y_{j}\ge \frac{B(1-2/M)-\delta}{M^{2}}=\frac{B/3-\delta}{9}=\frac{B}{27}-\frac{\delta}{9}.$$
By keeping the same signs for $x_{i},y_{j}$ but shifting them to $\pm1$, the objective cannot decrease. 

This gives a rounded solution in polynomial time, achieving an approximation factor of at least $0.03 < 1/27$.

## Conclusion
In the next post, we'll see another algorithm, which is Randomized and based on Grothendieck's inequality, that achieves much better approximation of the Cut-Norm. Essentially, this post was about reducing the problem to an SDP, and providing a very smart way to round the solution. These are, of course, not my ideas, and they were proposed by Alon and Naor in their influential paper cited below.

## References
1.  **Alon, N., & Naor, A.** (2006). [*Approximating the Cut-Norm via Grothendieck's Inequality*](https://doi.org/10.1137/050628320). SIAM Journal on Computing.
2.  **Grothendieck, A.** (1953). [*Résumé de la théorie métrique des produits tensoriels topologiques*](https://eudml.org/doc/144766). Boletim da Sociedade de Matemática de São Paulo.
3.  **Håstad, J.** (2001). [*Some optimal inapproximability results*](https://dl.acm.org/doi/10.1145/502090.502098). Journal of the ACM.