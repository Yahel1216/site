---
title: "Algebraic Techniques for Fast Integer Multiplication"
date: 2026-01-18
slug: fast-integer-mult
draft: false
katex: true
description: "Different algebraic techniques and tricks used to derive fast integer multiplication algorithms"
tags: ["algebraic-algorithms", "algebra", "fft", "algorithms"]
categories: ["Algebraic Algorithms", "Theory"]
---
> [!note] Prerequisites
> *   **Ring Theory:** Homomorphisms, Ideals, Quotient Rings, Chinese Remainder Theorem (see [Appendix](#appendix-algebraic-definitions) for definitions).
> *   **Polynomials:** Polynomial multiplication, division, convolutions.
> *   **Algorithm Analysis:** Basic recursive complexity and $O$-notation.

Given two integers $a,b\in \mathbb{N}$, how fast can we multiply them?

Let's recall the elementary school algorithm for integer multiplication:
1. Given two **decimal** numbers $a,b$ of length $n$ and $m$ (assume $n \ge m$).
2. For every $i \in [n]$ and $j \in [m]$:
   1. Compute the product of the $i$-th digit of $a$ and the $j$-th digit of $b$.
   2. Shift the result by $i+j$ positions.
   3. Add to the accumulator.
3. Return the accumulated number.

Overall we perform $O(n\cdot m)$ multiplications of one-digit numbers, which can be done in constant time just by writing down the multiplication table. We also perform $O(nm)$ additions. By carefully ordering the addition operations, we can ensure there are $O(n)$ additions of $O(m)$-long numbers. Since addition requires linear time, we obtain a total runtime of $O(nm)$.

Note that by moving to binary instead of decimal, the runtime remains asymptotically equivalent, because the number of digits needed to represent a number $a$ in base $B$ is roughly $\log_B (a)$ and $\log_{B'}(a)=\frac{\log_B(a)}{\log_{B}(B')}$ showing $\log_B(a)=\Theta(\log_{B'}(a))$ for any other base $B'$.

For many years, up to Karatsuba's discovery in the 60's, it was the best known method. In this post, we discuss the deep ideas behind fast algorithms for integer multiplication, building up the fastest algorithms known up to a few years ago.

While often treated as bit-manipulation tricks, these algorithms are fundamentally **algebraic**. What do we mean by an **Algebraic Algorithm**?

One way to define algebraic algorithms, also called *symbolic computation*, is that we limit the algorithm to algebraic manipulations of symbolic expressions (like polynomials). In particular, we do not allow numerical approximation, and the algorithm must output an exact result. We often treat *ring* operations as constant time operations, and they are the basic building block for the algorithms.

The algorithms we'll cover in this post are all algebraic, but it should be noted that the recent state-of-the-art algorithms for integer multiplication rely analytic bounds and numerical approximations.

## The Algebraic Framework

Before diving into specific algorithms, we distill the common techniques they all share. These tools allow us to transport a multiplication problem from one ring to another where it might be easier to solve.

### Tool 1: Mapping
> [!caution] Definition: Mapping
> Let $R, A$ be rings and $f: R \to A$ be a ring homomorphism. Given $r, s \in R$, we can compute $r \cdot s$ by computing $f(r) \cdot f(s)$ in $A$. If $f$ is injective (or we have a way to invert specific elements), we can recover $r \cdot s$.

**Example: Modular Arithmetic**. If $a,b\in\mathbb{N}$ are integers which are both (strictly) smaller than $m$, then we can compute $a\cdot b$ by first reducing $a,b$ modulo $m^2-1$, computing the product of the reduced elements, and return that product as the result. This works because $\mathbb{Z}\to \mathbb{Z}\_{m^2-1}$ given by $x\mapsto x\mod (m^2-1)$ is a homomorphism, and $\mathbb{Z}\_{m^2-1}\to \mathbb{Z}$ given by $x\mapsto x$ inverts the homomorphism for $a,b$ which are small enough.

**Example: Modular Arithmetic for Polynomials**.
Fix a monic polynomial $p \in R[x]$ with $\deg(p) > 2n$. If $r, s \in R[x]$ have degree at most $n$, their product has degree at most $2n$. Thus, we can recover $r \cdot s$ perfectly from its image in the quotient ring $R[x]/(p)$.
Formally, we use the homomorphism $f: R[x] \to R[x]/(p)$ given by $q(x) \mapsto q(x) \pmod{p(x)}$.

If $\deg(p) = 2n$, we might lose the highest degree coefficient ($x^{2n}$). We can **fix** this missing data by performing a single scalar multiplication of the leading coefficients $r_n s_n$ and adding $r_n s_n \cdot p(x)$ back to result. This is often called **evaluating at $\infty$**.

### Tool 2: Lifting
> [!caution] Definition: Lifting
> Let $I \lhd R$ be an ideal. The quotient ring $R/I$ can be "lifted" back to $R$. If we have an equation $ab = c$ in $R/I$, we can choose representatives $\hat{a}, \hat{b} \in R$ and compute $\hat{a}\hat{b}$. The result will be congruent to $c$ modulo $I$.

A crucial application of lifting is **Clumping** (or Substitution).

**1. Base-$B$ Clumping (Integers $\to$ Polynomials)**
To multiply integers $a, b$, we can view them as polynomials evaluated at a base $B$.
1.  Write $a = \sum a_i B^i$ and $b = \sum b_i B^i$ where $0 \le a_i, b_i < B$.
2.  Lift $a \to a(y) = \sum a_i y^i$ and $b\to b(y)=\sum b_i y^i$ in $\mathbb{Z}[y]$.
3.  Compute $p(y) = a(y) \cdot b(y)$ and recover the result $a\cdot b$ from $p$.

Formally, we have **mapped** $\mathbb{Z}\to \mathbb{Z}[y]/(y-B)$ (view the numbers as polynomials evaluated at $B$), and then **lifted** $\mathbb{Z}[y]/(y-B)\to \mathbb{Z}[y]$, forgetting the evaluation at $B$. We know that $p(y)\equiv a(y)b(y)\mod (y-B)$, which means $p(B)=ab$. Note that computing $a(y)\cdot b(y)$ can be now done using integer multiplications of size $B$, while $a,b$ are potentially much larger than $B$.

**2. Polynomial Clumping** This is a useful way to control the degrees of polynomials. Formally, we **map** $R[x]\to R[x][y]/(x^n-y)$, replacing powers of $x^n$ by powers of $y$. Then, we **lift** $R[x][y]/(x^n-y)\to R[x][y]$, forgetting the relationship between $x,y$. Doing this to a polynomial $p(x)$ with degree $kn$, we obtain a new polynomial with $x$-degree $n$ and $y$-degree $k$. Another way to view this is -- doing this process we obtain a new polynomial $P(y)$ whose coefficients are **polynomials** in $x$. The degree of $P(y)$ is $k$.

### Tool 3: Chinese Remainder Theorem (CRT)
> [!tip] Theorem (Chinese Remainder Theorem)
> Let $I, J$ be co-prime ideals in $R$ (meaning $I+J=R$). Then:
> $$ R/(IJ) \cong (R/I) \times (R/J) $$
> The isomorphism is given by $z \mapsto (z \bmod I, z \bmod J)$.

This allows us to perform arithmetic in the product ring $(R/I) \times (R/J)$—which effectively parallelizes the computation—and reconstruct the result in $R/(IJ)$.

**Remark.** Note that $I,J$ are co-prime iff $\exists u\in I,v\in J$ such that $u+v=1$. The inverse CRT map $(R/I)\times (R/J)\to R/IJ$ is given by $(x,y)\mapsto vx+uy$. Note the order of $v,u$ in the map is crucial! 
We mostly work over **principal ideal domains**, which are rings in which there are no-zero divisors and that every ideal $I$ is generated by a single element, $I=(a)=Ra$ for some $a\in R$. Thus assuming $I=(a),J=(b)$, we have that  $I+J=R$ iff and there are $r,s\in R$ such that $ra+sb=1$. The elements $r,s$ are called **Bezout's coefficients**.

**Example: Remainder integer arithmetic**. To multiply in $\mathbb{Z}_{(p-1)(p+1)}$ we can map to $\mathbb{Z}\_{p-1}\times \mathbb{Z}\_{p+1}$, compute the product in the product ring and invert by the map $$(x,y)\mapsto a(p+1)x+b(p-1)y,$$where $a,b$ are the unique numbers in $\mathbb{Z}\_{(p-1)(p+1)}$ satisfying $$a(p+1)+b(p-1)=1.$$

**Example: Polynomial remainder arithmetic**. Consider the ring $R[x]/(x^{2}-x)$. We can factor $(x^{2}-x)=(x)(x-1)$, and these are co-prime ideals because taking $u=x\in (x)$ and $v=-(x-1)=-x+1\in (x-1)$ we see that $$u+v=x-x+1=1\in R[x]/(x^{2}-x).$$Hence the inverse map for the isomorphism $z\mapsto (z\pmod{x},z\pmod{(x-1)})$ is given by $$(a,b)\mapsto va+ub=-ax+a+b=a+(b-a)x.$$
The formula suggests that **inverting** the CRT in this setup requires $2$ multiplications and $1$ addition.

## Algorithms

We now present the major algorithms using the framework above.

### 1. Karatsuba's Algorithm
Consider multiplying two linear polynomials $a(x) = a_0 + a_1x$ and $b(x) = b_0 + b_1x$ in $R[x]$. The standard product requires **4** multiplications ($a_0b_0, a_0b_1, a_1b_0, a_1b_1$).

Karatsuba's trick reduces this to **3** using the CRT with ideals $(x)$ and $(x-1)$:

1.  **Map:** Reduce modulo $x^2-x$, $R[x]\to R[x]/(x^2-x)$. For $a(x),b(x)$ this doesn't do anything.
2.  **CRT Projection:** Map to $R[x]/(x^2-x)\to (R[x]/x) \times (R[x]/(x-1))$.
    *   $a(x) \mapsto (a_0, a_0+a_1)$
    *   $b(x) \mapsto (b_0, b_0+b_1)$
3.  **Multiply:** Compute pair-wise products:
    *   $P_0 = a_0 b_0$
    *   $P_1 = (a_0+a_1)(b_0+b_1)$
4.  **Invert CRT:** Reconstruct the polynomial in $R[x]/(x^2-x)$.
    *   The linear term coefficient is $P_1 -P_0$.
5.  **Fix:** Evaluate at $\infty$, by computing $P_2=a_1b_1$ directly and returning $$P_0 +(P_1-P_0) x+ P_2 \cdot (x^2-x)=P_0+(P_1-P_0-P_2)x+ P_2 x.$$

This requires only $3$ multiplications ($P_0, P_1, P_2$). It also requires $4$ addition and subtraction operations.

> [!note] Variation: Knuth's Trick
> Instead of $x(x-1)$, we can use $x(x+1)$, mapping to evaluations at $0$ and $-1$.
> $$ (a_0, a_0-a_1) \cdot (b_0, b_0-b_1) $$
> The reconstruction logic is symmetric.

#### Application: Complex Multiplication
We can identify $\mathbb{C}$ with $\mathbb{R}[i]/(i^2+1)$. Standard multiplication $(a+bi)(c+di)$ takes 4 real multiplications. Using a variation of the trick above (evaluating at $\infty$ and $1$, essentially), we can do it in 3: compute $P_0=ac,P_1=(a+b)(c+d)$ and $P_2=bd$ and return
$$ (ac-bd) + i[(a+b)(c+d) - ac - bd] =(P_0 -P_2)+ i(P_1-P_0 -P_2)$$

#### Application: Integer Multiplication
Given two integers $a,b\in\mathbb{Z}$, which have $n$-digit long $B$-basis representation, we can:
1. Perform base $B^{n/2}$-**clumping** to obtain two polynomials $$p(x)=a_{0}+a_{1}x\quad,\quad q(x)=b_{0}+b_{1}x$$such that $a=a_{0}+a_{1}B^{n/2}$ and $b=b_{0}+b_{1}B^{n/2}$.  The numbers $a_0,a_1,b_0,b_1$ are all $n/2$-digit long (in basis $B$).
2. Computing $p(x)\cdot q(x)$ using Karatsuba's trick, requires $3$ multiplications and $4$ additions of $n/2$-long numbers.
3. Note that $$ab=a_{0}b_{0}+a_{1}b_{0}B^{n/2}+a_{0}b_{1}B^{n/2}+a_{1}b_{1}B^{n},$$so we can recover the result $ab$ by shifting the coefficients of the product polynomial by a correct number of digits, and sum up the results. This requires $3$ additions of numbers $2n$-long numbers (at most).

By recursively applying this method, we see that the time to compute $n$-long multiplication is $$T(n)=3T(n/2)+O(n)$$
Using the master theorem, this leads to $T(n)=O(n^{\log_2 3})$.

### 2. Toom-Cook (Toom-$k$)
Toom-Cook generalizes Karatsuba. Instead of splitting a number into 2 parts (degree 1 polynomial), we split it into $k$ parts (degree $k-1$ polynomial).
Note that in $R[x]$, the ideals generated by $(x-\alpha)$ and $(x-\beta)$ (linear polynomials) are co-prime whenever $\alpha\not=\beta$.

**Procedure:**
1. Setup: Choose $2k-1$ distinct points $\alpha_1,\ldots,\alpha_{2k-1}\in R$.
2.  **CRT:** Map $R[x] \to \prod_{i=1}^{2k-1} R[x]/(x - \alpha_i)$. This is equivalent to evaluating the polynomial at these points.
3.  **Multiply:** Perform $2k-1$ recursive multiplications of the values.
4.  **Invert CRT:** Recover the product polynomial (degree $2k-2$) from the $2k-1$ point-value pairs. This step is often called **interpolation**.

**Runtime Analysis:**
We perform $2k-1$ recursive calls on inputs of size $n/k$.
$$ T(n) = (2k-1)T(n/k) + O(n) $$
The solution is $T(n) = O(n^{\log_k(2k-1)})$.
*   For $k=2$, $\log_2 3 \approx 1.58$. (Karatsuba)
*   For $k=3$, $\log_3 5 \approx 1.46$. (Toom-3)
*   As $k \to \infty$, the exponent approaches 1.

### 3. Fast Fourier Transform (FFT)
The overhead of CRT and interpolation (inverting the CRT) in Toom-Cook grows rapidly with $k$. The FFT is essentially Toom-Cook where the evaluation points are **roots of unity**. The special structure and algebraic properties of roots of unity give a much faster algorithm for computing both the CRT and the inverse CRT.

> [!caution] Definition
> A primitive $n$-th root of unity is an element $\zeta\in R$ such that $\zeta^n=1$ and for every $1\le k<n$ it holds $\zeta^k\not=1$.

For example, in $\mathbb{C}$, we have the complex exponent $\exp(2\pi i / n)$, which is a primitive $n$-th root of unity. Note that $\zeta^k\not=\zeta^j$ for every $j\not=k$ in $[1,n]$, otherwise that would mean (by exponent rules) $\zeta^{k-j}=1$, contradicting the definition.

Therefore, by CRT we have $$R[x]/(x^{n}-1)\cong \prod_{k=1}^n R[x]/(x-\zeta^k).$$

**The Procedure**: Given two polynomials $p(x),q(x)\in R[x]$ with degree at most $n-1$,
1. **Map:** $R[x]\to R[x]/(x^{2n}-1)$. This does nothing to $p,q$.
2. **CRT:** $R[x]/(x^{2n}-1)\to \prod_{k=1}^{2n} R[x]/(x-\zeta^k)$ assuming $\zeta$ is a primitive $2n$-th root of unity.
3. **Multiply:** in each factor separately, i.e., compute $p(\zeta^k)\cdot q(\zeta^k)$ for every $k$.
4. **Inverse CRT:** Interpolate using the inverse CRT map. Recover $p(x)\cdot q(x)\in R[x]/(x^{2n}-1)$.

**The Trick:** The CRT map in this case is called the **Discrete Fourier Transform** and it can be computed fast, using the **Cooley-Tukey** algorithm, in time $O(n\log n)$. Let us sketch the algorithm: Let $\alpha\in R$ denote an **invertible** element, and note $$R[x]/(x^{2n}-\alpha^2)\cong R[x]/(x^n-\alpha)\times R[x]/(x^n +\alpha).$$
Note that $\frac{1}{2\alpha}(x^n+\alpha)-\frac{1}{2\alpha}(x^n-\alpha)=1$ and so the inverse CRT map is given by $$(p(x),q(x))\mapsto \frac{1}{2\alpha}\cdot [(x^n+\alpha)\cdot p(x)+(x^n-\alpha)\cdot q(x)]$$
The product $x^n\cdot p(x)$ is just shifting a degree $\le n-1$ polynomial by $n$ powers. So overall to compute the inverse CRT map we need $n$ multiplications in $R$ (namely multiplying $1/2\alpha$ by the coefficients of $p,q$) and $2n$ additions of elements in $R$ (multiplied by $1/2$). If $\alpha=\zeta^{n}$ and $n$ is a power of two, we can apply this recursively to obtain a total operation count of $O(n\log n)$ to invert the CRT. A similar approach gives the same runtime for the CRT map. This type of algorithm for computing the DFT in time $O(n\log n)$ is called a Fast Fourier transform (FFT).

The case where $n$ is not a power of $2$ can be dealt with using **Bluestein's trick**, which encodes the DFT as a polynomial multiplication, which can be computed using slightly larger power of $2$ FFTs. So the result of $O(n\log n)$ carries over to the general case of $n$ not being a power of $2$.

To use the FFT for computing integer multiplication, we need a root of unity of the relevant order. Without it, the runtime is the same as in the Toom-Cook method. The next algorithm deals with this issue.

### 4. Schönhage-Strassen Algorithm
Standard FFT requires roots of unity. The ring of integers $\mathbb{Z}$ does not have them. We must "manufacture" a ring that does.

**Procedure:**
1.  **Base $2^m$-Clumping:** Break $N$-bit numbers into blocks of size $m$. Treat them as polynomials of degree $n-1$ where $N = mn$. Formally, $\mathbb{Z}\to \mathbb{Z}[x]/(x-2^m)\to \mathbb{Z}[x]$. By construction, the coefficients of the polynomial are $m$-bit numbers, at most $2^m-1$.
2.  **Map:** Reduce modulo $(x^n +1)$, i.e., $\mathbb{Z}[x]\to \mathbb{Z}[x]/(x^n+1)$.
3. **Map:** Reduce modulo $(2^{nk}+1)$ where $k$ is some positive integer. In other words, identify $2^{nk}$ with $-1$. Formally, we reduce the coefficients of the polynomials $$\mathbb{Z}[x]/(x^n +1)\to (\mathbb{Z}/(2^{nk}+1))[x]/(x^n+1).$$
4. **Apply the FFT trick:** Let $R=\mathbb{Z}/(2^{nk}+1)$, then we want to compute a product in the ring $R[x]/(x^n+1)$. Note that $\zeta=2^k$ is a primitive root of unity of order $2n$ in $R$, therefore we can apply the FFT trick to compute the product in $O(n\log n)$ operations in $R$.
5. **Recursion**: Apply the procedure recursively to compute products in $R$, which are $nk$-bit integers.

When can we recover the result? In other words, when is this invertible?
- Note that polynomial multiplication modulo $x^n +1$ is just the **nega-cyclic convolution** of the coefficient vectors. In other words, if $p(x)=\sum_{i=0}^{n-1}a_i x^i$ and $q(x)=\sum_{i=0}^{n-1}b_i y^i$, then $$
\begin{aligned} p(x)\cdot q(x)\mod (x^n+1)& =\sum_{j=0}^{2n-2}\left(\sum_{i=0}^{j} a_i b_{j-i} \right)x^j\mod (x^n+1)
\\\\ &=\sum_{j=0}^{n-1}\left(\sum_{i=0}^{j} a_i b_{j-i} - \sum_{i=j+1}^{n-1} a_{i} b_{(j-i)\mod n}\right) x^j
\end{aligned}$$
where we interpret indices out of range as $0$. In particular, the coefficient of $x^j$ is the **sum of $n$ integer products** where each product involves two numbers smaller than $2^m-1$. 
- Therefore, the coefficients of this product polynomials are strictly smaller than $2^{2m}n$. Hence to ensure we can recover the correct result we must have $2^{nk}\ge 2^{2m}n$.
- This implies $k\ge \frac{\log_2(n)+2m}{n}$, and when the ratio of $m/n$ is fixed, we can pick $k$ to be a small constant.

Note that the second and third steps don't really change anything about the polynomials, but only move them between rings with different multiplication rules. To actually implement this, we need to choose a balance between $n,m$ and $k$.

**Runtime Analysis:** Let $T(N)$ be the bit-complexity of multiplying two $N$-bit numbers. Choosing $n,m= \sqrt{N}$ (without loss of generality assume $N$ has a square root), we reduce the problem to $\sqrt{N}$ sub-problems of size $\sqrt{N}$. In this setup, we can choose $k=3$ (assuming $n$ is large enough, this satisfies the condition above). 

Since multiplication by the root of unity ($2^k$) is just a shift of the binary representation, it can be done in linear time. Therefore the FFT requires $O(n\log n)$ operations which are all **linear** (recall that FFT only uses multiplications by the root of unity and division by $2$, which is also a shift), i.e., cost $O(nk)$. Thus, the FFT runs in $$O(nk\cdot n\log n)=O(N\log N)$$
 Hence the runtime complexity satisfies the recurrence:
$$ T(N) = \sqrt{N} \cdot T(\sqrt{N}) + O(N \log N) $$
This resolves to $T(N)=O(N\log N\log\log N)$.

### 5. Nussbaumer's Trick

When dealing with multi-variate polynomials, we can define the **Multi-dimensional DFT**, which is the CRT map:
$$ R[x_{1},\ldots,x_{d}]/(x_{1}^{t_{1}}-1,\ldots,x_{d}^{t_{d}}-1)\to \prod_{(i_1, \dots, i_d)} R[x_{1},\ldots,x_{d}]/(x_{1}-\xi_{1}^{i_{1}},\ldots,x_{d}-\xi_{d}^{i_{d}}) $$
where $\xi_{j}$ is a primitive root of unity of order $t_{j}$. This is equivalent to computing the FFT along one axis (variable), and then by a second, and so on. If we choose $\xi_{j}$ to be **ring elements** in $R$, the total number of **ring multiplications** is $\Theta(t^{d}\log (t^{d}))$. As we observed before, the multiplications done in the FFT are just multiplications by a power of $\xi$ (and division by $2$, but this can be done at the end of the recursion for cheaper).

However, suppose $t_{1}=\ldots=t_{d}=t$. Note that $x_{1}$ is a suitable choice for $\xi_{2},\ldots,\xi_{d}$, since it satisfies $x_1^{t}=1$. Suppose we start by applying the FFT along the last variable $x_{d}$, viewing the polynomial as having coefficients which are themselves **polynomials in $x_1,\ldots ,x_{d-1}$**. Multiplying a polynomial in $x_1,\ldots,x_{d-1}$ by by $x_{1}=\xi_{d}$ is essentially a **shift of the coefficients**. Therefore, there are no **ring multiplications** when applying the FFT trick, on the last $d-1$ axes. The only ring multiplications happen at the base level of the recursion, which is $R[x_{1}]/(x_{1}^{t}-1)$. 

At the base level of the recursion, there are $O(t\log t)$ ring multiplications. Since every application of the FFT produces $t$ products in the ring, there are $t^{d-1}$ leaves in the recursion tree. Hence the total number of **ring multiplications** is:
$$ O(t^{d}\log t)=O\left(\frac{t^{d}\log (t^{d})}{d}\right) $$
This works even when $t_{1},\ldots,t_{d}$ are not equal but close enough to each other. The implication is that we saved a factor of $d$ in the number of ring multiplications. If we are working on an integer multiplication algorithm, this saves a factor $d$ in the number of **recursive** calls to the algorithm.

Since additions and **shifts** are much cheaper than ring multiplications (at least in the case of integers), this $d$-factor may be instrumental for the runtime. In the newest state-of-the-art algorithm, this trick is exploited.


## Conclusion & Outlook

We have journeyed from the $O(n^2)$ grade-school method to the $O(n \log n \log \log n)$ of Schönhage-Strassen using purely algebraic insights: mapping, lifting, and the Chinese Remainder Theorem.

For decades, Schönhage-Strassen was the champion. In 2007, Martin Fürer introduced an algorithm running in $O(n \log n \cdot 2^{O(\log^* n)})$. Finally, in 2019, Harvey and van der Hoeven achieved the holy grail: **$O(n \log n)$**. While these modern algorithms rely on heavy analytic machinery (and valid over $\mathbb{Z}$ specifically), the algebraic pillars we discussed here remain the foundation of computer algebra, and are just beautiful ideas!

## Appendix: Algebraic Definitions

> [!caution] Definition: Ring
> A **ring** $R$ is a set equipped with two binary operations, addition ($+$) and multiplication ($\cdot$), satisfying the following axioms:
> 1.  $(R, +)$ is an abelian group (associative, commutative, has an identity $0$, and every element has an additive inverse $-r$).
> 2.  Multiplication is associative: $(a\cdot b)\cdot c = a \cdot (b \cdot c)$.
> 3.  Distributivity holds: $a(b+c) = ab + ac$ and $(a+b)c = ac + bc$.
>
> If multiplication is commutative ($ab=ba$), $R$ is a **commutative ring**. If there exists an element $1 \in R$ such that $1 \cdot r = r \cdot 1 = r$ for all $r$, $R$ is a **ring with identity**. In this post, we assume all rings are commutative with identity.

> [!caution] Definition: Field
> A **field** $\mathbb{F}$ is a commutative ring with identity where every non-zero element $r \in \mathbb{F}$ has a multiplicative inverse $r^{-1}$ such that $r \cdot r^{-1} = 1$. Examples include $\mathbb{Q}, \mathbb{R}, \mathbb{C}$, and finite fields $\mathbb{Z}_p$ for prime $p$.

> [!caution] Definition: Polynomial Ring
> Given a ring $R$, the **polynomial ring** $R[x]$ consists of formal sums $\sum_{i=0}^n a_i x^i$ with coefficients $a_i \in R$.
> *   The **degree** of a polynomial, denoted $\deg(p)$, is the largest $k$ such that $a_k \neq 0$.
> *   A polynomial is **monic** if its leading coefficient $a_n$ is $1$.

> [!caution] Definition: Homomorphism & Isomorphism
> A map $\phi: R \to S$ between rings is a **ring homomorphism** if it preserves the structure:
> $$ \phi(a+b) = \phi(a) + \phi(b) \quad \text{and} \quad \phi(ab) = \phi(a)\phi(b) $$
> If $\phi$ is a bijection (one-to-one and onto), it is an **isomorphism**, denoted $R \cong S$.

> [!caution] Definition: Ideals
> An **ideal** $I \subseteq R$ is a subset that forms a subgroup under addition and is "sticky" under multiplication: for any $r \in R$ and $x \in I$, the product $rx \in I$.
> *   **Principal Ideal:** An ideal generated by a single element $a$, denoted $(a) = \{ra : r \in R\}$.
> *   **Coprime Ideals:** Two ideals $I, J$ are **coprime** if their sum is the whole ring, i.e., $I + J = R$. This implies there exist $u \in I, v \in J$ such that $u+v=1$.

> [!caution] Definition: Quotient Ring
> Given an ideal $I \subseteq R$, the **quotient ring** $R/I$ is the set of equivalence classes modulo $I$. The elements are of the form $a + I$, with operations:
> $$ (a+I) + (b+I) = (a+b) + I $$
> $$ (a+I) \cdot (b+I) = (ab) + I $$

> [!caution] Definition: Roots of Unity
> An element $\omega \in R$ is an **$n$-th root of unity** if $\omega^n = 1$.
> It is a **primitive** $n$-th root of unity if it generates the cyclic subgroup of order $n$ under multiplication (often defined requiring $\omega^k \neq 1$ for all $1 \le k < n$ and that $n$ is invertible in $R$).

## References

1.  **Karatsuba, A. and Ofman, Y.** (1962). [*Multiplication of Many-Digital Numbers by Automatic Computers*](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=26729&option_lang=eng).
2.  **Toom, A. L.** (1963). [*The Complexity of a Scheme of Functional Elements Realizing the Multiplication of Integers*](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=27978&option_lang=eng).
3.  **Schönhage, A. and Strassen, V.** (1971). [*Schnelle Multiplikation großer Zahlen*](https://doi.org/10.1007/BF02242355).
4.  **Harvey, D. and van der Hoeven, J.** (2021). [*Integer multiplication in time $O(n \log n)$*](https://annals.math.princeton.edu/2021/193-2/p04).