---
title: "Fast Fourier Transform over Finite Fields"
date: 2026-01-18
slug: fft-finite-field
draft: false
katex: true
description: "A generalized algebraic approach to the FFT, extending beyond the complex numbers to finite fields and arbitrary rings"
tags: ["algebraic-algorithms", "algebra", "fft", "algorithms"]
categories: ["Algebraic Algorithms", "Theory"]
---
> [!note] Prerequisites
> *   **Ring Theory:** Basic definitions (Rings, Fields, Polynomials).
> *   **Group Theory:** Cyclic groups and generators.
> *   **Basic FFT:** Familiarity with the standard divide-and-conquer strategy on $\mathbb{C}$ is helpful but not required.

The Fast Fourier Transform (FFT) is one of the most important algorithms in history. Usually, it is introduced over the field of complex numbers $\mathbb{C}$, relying on geometric intuition about roots of unity lying on the unit circle.

However, in Computer Science—particularly in Cryptography and Coding Theory—we often care about **Finite Fields** $\mathbb{F}_q$. The geometric intuition fades, but the algebraic structure remains. In this post, we will explore how to perform efficient Fourier Transforms over any ring, leading us to advanced techniques like Bluestein's trick and the Schönhage-Strassen algorithm for polynomial 

Throughout the post $R$ will denote an arbitrary ring with a unit, $\mathbb{F}_p$ a field of **prime** size $p$, and $\mathbb{F}_{q}$ a finite size (we know $q=p^k$ for some prime $p$).

## Foundations: Roots of Unity

There are many ways to derive the Discrete Fourier Transform (DFT). We will use a formulation that keeps a distance from representation-theoretic roots and works for any ring.

> [!caution] Definition: Principal Root of Unity
> Let $R$ be any ring and $n \ge 1$ an integer. We say $\alpha \in R$ is a **principal** $n$-th root of unity if:
> 1.  $\alpha^n = 1$.
> 2.  $\sum_{j=0}^{n-1} \alpha^{jk} = 0$ for every $1 \le k < n$.
>
> Another definition is that of a **primitive root of unity** -- if $\alpha^n=1$ and $\alpha^k\not=1$ for every $1\le k<n$.

In fields, a primitive root of unity is always a principal root of unity (so it's a **stronger** definition).

> [!caution] Definition: The DFT
> Let $\alpha\in R$ be a principal $n$-th root of unity. Given a tuple $(v_0, \dots, v_{n-1})$ of elements in $R$, the **Discrete Fourier Transform** is the tuple $(f_0, \dots, f_{n-1})$ where:
> $$ f_k = \sum_{j=0}^{n-1} v_j \alpha^{jk} $$
> Equivalently, if $P(x) = \sum v_j x^j$, the DFT is the evaluation tuple $(P(\alpha^0), \dots, P(\alpha^{n-1}))$.

In fields, a primitive root of unity is always a principal root of unity. To work over finite fields $\mathbb{F}_q$, we rely on the following structural fact:

> [!tip] Fact: Structure of $\mathbb{F}_q^\times$
> Let $\mathbb{F}_q^{\times}$ denote the set $\mathbb{F}_q\setminus \set{0}$. Since this is a field, every non-zero element is invertible, so this is a group under the multiplication operation. The crucial fact is that $\mathbb{F}_q^{\times}$ is a **cyclic group** of order $q-1$.
> This means there exists a generator $\gamma \in \mathbb{F}_q^\times$ such that $\gamma^{q-1} = 1$, but $\gamma^j \neq 1$ for any $1 \le j < q-1$.

Thus $\mathbb{F}_q$ contains a primitive $n$-th root of unity if and only if $n$ divides $q-1$.

## The "Easy" Case: Radix-2 FFT

If the ring $R$ has a primitive root of unity of order $2^k$, then we can compute the DFT fast, using a divide-and-conquer approach which is almost identical to the complex case. In fact, it is the same algorithm, but we abstract away details about $\mathbb{C}$. Note that $\mathbb{C}$ has primitive roots of unity of any order. For a finite field $\mathbb{F}_q$ to support this algorithm we need $2^k\mid q-1$ ($2^k$ divides $q-1$), using the observations above.

> [!tip] Theorem (Radix-2 FFT)
> Let $R$ be a ring with a $2^k$-th principal root of unity $\alpha \in R$. Then the DFT of size $n=2^k$ for a polynomial $P \in R[x]$ (degree $< n$) can be computed in time $O(n \log n)$.

**Proof Sketch:**
The idea is the standard Cooley-Tukey algorithm. We decompose the evaluation of $P(x)=\sum a_i x^i$ into evaluations of its even and odd parts:
$$ P(x) = P_{\text{even}}(x^2) + x \cdot P_{\text{odd}}(x^2) $$
where $P_{\text{even}}(y) = \sum a_{2i} y^i$ and $P_{\text{odd}}(y) = \sum a_{2i+1} y^i$.
Since evaluating at $\{1, \alpha, \dots, \alpha^{n-1}\}$ involves squaring the points, the set of evaluation points for the sub-problems reduces to $\{1, \alpha^2, \dots, \alpha^{2(n/2)-1}\}$, which is exactly the set of powers of $\alpha^2$, a principal root of unity of order $n/2=2^{k-1}$. This follows from the fact $$(\alpha^j)^2=\alpha^{2j}=\begin{cases} \alpha^{2j} & j< n/2, \\\\ \alpha^{2j-n}\alpha^n=\alpha^{2j -n} & j\ge n/2,\end{cases}$$
using $\alpha^n=1$. Hence we can recover the DFT by computing two $n/2$ DFTs for $P_{\text{even}},P_{\text{odd}}$ and combining the results in **linear time**.
The runtime recurrence is $T(n) = 2T(n/2) + O(n)$, yielding $T(n) = O(n \log n)$. $\blacksquare$

This algorithm is called an FFT (fast Fourier transform). A similar algorithm works in the inverse direction, yielding the inverse FFT algorithm. Using the famous **Convolution Theorem**, this allows us to multiply polynomials efficiently.

> [!important] Corollary
> Let $R$ be a ring with a $2^k$-th primitive root of unity. Given polynomials $f, g \in R[x]$ with degree $< n = 2^k$, we can compute their product $h \equiv f \cdot g \pmod{x^n - 1}$ in time $O(n \log n)$ using two forward FFTs and one Inverse FFT.

**Proof Sketch:** Every polynomial of degree $<n$ is uniquely determined by its values on $n$ **distinct** points. Moreover, the value of the product polynomial $f\cdot g$ at a point $\xi\in R$ is just $f(\xi)\cdot g(\xi)$. Let $h=f\cdot g\mod (x^n-1)$, and note it can be written as $h(x)=(f(x)\cdot g(x)) - (x^n -1)\cdot q(x)$ where $q$ is the quotient polynomial (and $h$ is the remainder). Note that $h$ has degree $<n$. The value of $h$ on a $n$-th root of unity $\alpha$ (not a principal necessarily) is $$h(\alpha)=(f(\alpha)\cdot g(\alpha))- (\alpha^n -1)\cdot q(\alpha)=f(\alpha)\cdot g(\alpha)$$
since $\alpha^n-1=0$ as $\alpha$ is a root of unity. Therefore $h$ is uniquely determined by the values $$(f(1)\cdot g(1), f(\alpha)\cdot g(\alpha),\ldots , f(\alpha^{n-1})\cdot g(\alpha^{n-1}))$$
where $\alpha$ is a principal $n$-th root of unity. Computing the values requires computing the DFT of $f$ and $g$, taking the element-wise products (in linear time), and then applying the inverse DFT to obtain $h$ (this is interpolation). Overall, this takes $O(n\log n)$ time. $\blacksquare$


## FFT of General Size
What if we want to compute the DFT of size $n$ where $n$ is not a power of $2$? Then the Radix-2 FFT cannot be used directly. **Bluestein's trick** is a way to solve this problem, by encoding the DFT of size $n$ as polynomial multiplication, which can be computed fast by DFTs of slightly larger size, which is a power of $2$.

> [!tip] Theorem: Bluestein's Trick
> Let $\alpha\in R$ denote a principal $2n$-th root of unity. Given a polynomial $P(x)\in R[x]$ with degree $<n$, write $P(x)=\sum_{i=0}^{n-1}a_i x^i$. Then the DFT of $P$, denoted $f=(P(1),P(\alpha^2),\ldots,P(\alpha^{2(n-1)}))$, can be computed via the multiplication of $$p(y)=\sum_{i=0}^{n-1}a_i \alpha^{i^2}\cdot y^i\quad \text{by} \quad q(y)=\sum_{i=1}^{2n-1}\alpha^{-(n-i)^2}y^i$$
In particular, $f_i=P(\alpha^{2i})=\alpha^{i^2}\cdot b_i$ where $b_i$ is the coefficient of $y^i$ in $p(y)\cdot q(y)$.

**Proof.** By definition $$p(y)\cdot q(y)=\sum_{k=0}^{3n-2} \left(\sum_{i=0}^{n-1}a_i \alpha^{i^2}\cdot \alpha^{-(n-(k-i))^2}\right)y^k$$Noting that $i^2 - (n-(k-i))^2=i^2 -n^2 +2n(k-i)-k^2+2ki-i^2 = n(2(k-i)-n) -k^2 +2ki$ and since $\alpha^n=1$, the sum simplifies to $$p(y)\cdot q(y)=\sum_{k=0}^{3n-2} \alpha^{-k^2} \cdot \left(\sum_{i=0}^{n-1}a_i (\alpha^{2k})^{i}\right)y^k=\sum_{k=0}^{n-1} \alpha^{-k^2}\cdot P(\alpha^{2k})\cdot y^k +\sum_{k=n}^{3n-2}(\ldots)y^k,$$and so the coefficient of $y^k$ for $k<n$ is just $\alpha^{-k^2}\cdot f_k$. $\blacksquare$

> [!important] Corollary
> If $R$ has a $2n$-th principal root of unity and also a $2^k$-th principal root of unity, where $2^k\ge 3n-1$, the DFT of size $n$ can be computed in time $O(2^k k)=O(n\log n)$.

**Proof.** It reduces to computing a polynomial multiplication of two polynomials whose product degree is at most $3n-2$. It can be computed using two FFTs of size $2^k$ and one inverse FFT. We can choose $k$ to be minimal, because if $R$ has a $2^k$-th root of unity $\zeta$, then $\zeta^2$ is a $2^{k-1}$-th root of unity. Hence $2^k\le 2(3n-1)=O(n)$. $\blacksquare$

## Rings without Roots of Unity
What if the ring $R$ doesn't have the required root of unity to perform a DFT? To solve this we use the Schönhage-Strassen trick.

To multiply polynomials efficiently without a native root of unity, we construct a ring extension that has one. In the polynomial ring $R[x]$, we can "manufacture" a root of unity of order $2n$ by working in the quotient ring $R[x] / (x^{2n}-1)$. In this ring, $x^{2n} \equiv 1$.

> [!tip] Theorem (Schönhage-Strassen)
> The product of two polynomials $f,g\in R[x]$, of degree $<n$, where $n=2^k$ can be computed in time $O(n \log n \log \log n)$, even if $R$ doesn't contain a root of unity of order $2n$.

**Proof Sketch:**
We use a recursive technique involving **Kronecker Substitution** (also called segmentation). Note that $\deg(f\cdot g)\le 2n-2$, so computing $f \cdot g \pmod{x^{2n}-1}$ yields the exact product.
Let $m = 2^{\lfloor k/2 \rfloor}$ and $t = n/m$. Note that $n = m \cdot t$. (Also, $t=2m$ if $k$ is odd, while $t=m$ if $k$ is even).

1.  **Substitution:** We introduce a **new variable** $y$ and **identify it with** $x^m$.
    Explicitly, we map indices $i \to (u, v)$ such that $i = u + v \cdot m$ (where $0\le u<m$).
    This maps a univariate polynomial $f(x)$ to a bivariate polynomial $f'(x, y)$ such that $f(x) = f'(x, x^m)$.
    The resulting polynomials have $x$-degree $< m$ and $y$-degree $< t$.

2.  **The Recursive Ring:** Define the ring $D = R[x] / (x^{2m} + 1)$.
    This ring contains a $4m$-th root of unity (the element $x$, since $x^{2m} \equiv -1$).
    We view $f', g'$ as polynomials in $y$ with coefficients in $D$.
    *   *Note:* The coefficients of $f', g'$ are **polynomials** in $x$ with degree $<m$. Their product has degree $< 2m$. Thus, multiplication in $D$ (modulo $x^{2m}+1$) yields the **exact** product of coefficients, as no wrap-around occurs.

3.  **Choosing a Root:** The ring $D$ contains a $2t$-th primitive root of unity, which we denote $\eta$:
    *   If $t = 2m$, take $\eta = x$ (order $4m = 2t$).
    *   If $t = m$, take $\eta = x^2$ (order $2m = 2t$).

4.  **FFT in the Extension:** Compute the convolution of $f'$ and $g'$ modulo $(y^{2t} - 1)$.
    Since we have a root $\eta$ of order $2t$, we can use the Radix-2 FFT (recall $t$ is a power of $2$).
    This gives a polynomial $h^\*(x, y)$ with $\deg_y h^\* < 2t$ such that:
    $$ f' \cdot g' \equiv h^\* \pmod{(y^{2t} - 1)} $$

5.  **Reconstruction:** Lift $h^\*$ to a univariate polynomial $h \in R[x]/(x^{2n}-1)$ by substituting $y=x^m$.

The correctness relies on the "spacing" provided by the substitution:
1.  **No Aliasing in $x$:** We viewed the coefficients as elements of $D = R[x]/(x^{2m}+1)$. Since the input coefficients had degree $< m$, their point-wise product has degree $< 2m$. The modulus $x^{2m}+1$ is large enough to prevent these products from wrapping around and mixing with each other incorrectly.
2.  **No Aliasing in $y$:** Since the $y$-degrees are at most $t-1$, their products are $2t-2<2t$ and so by computing the product modulo $y^{2t}-1$ we have essentially computed the exact product (similar to the previous point).
3.  Thus, the evaluation $h^\*(x, x^m)=f'(x,x^m)\cdot g'(x,x^m)=f(x)\cdot g(x)$, and the reconstruction gives $h$ for which $h(x)=h^{\*}(x,x^m)$.

**Complexity:**
The runtime $T(k)$ satisfies the recurrence:
$$ T(k) \le 2^{\lceil k/2 \rceil} T(\lfloor k/2 \rfloor + 1) + O(2^k \cdot k) $$
The first term accounts for the recursive multiplications, and the second for the FFT additions/shifts at the current level. This solves to $T(n) = O(n \log n \log \log n)$. $\blacksquare$

> [!important] Corollary
> The DFT of size $n$ for a polynomial $f\in R[x]$ can be computed even in time $O(n\log n\log\log n)$ even if $R$ doesn't have a root of unity of order $n$ (or $2n$).

**Proof.** Use Bluestein's trick to reduce the problem to a polynomial multiplication, which can be computed fast using the Schonhage-Strassen algorithm. $\blacksquare$

## Application: Multipoint Evaluation

We can combine these tools to solve a classic problem: evaluating a polynomial at *every* point in a finite field $\mathbb{F}_p$. Recall that **Fermat's little theorem** says that $$a^{p-1}\equiv 1\pmod{p}$$for every $a\not=0$ in $\mathbb{F}\_p$. Therefore, the polynomial $x^p-x$ is equivalent to $0$ (as a function) over $\mathbb{F}\_p$.

### Univariate Evaluation
> [!important] Corollary
> Given a polynomial $P \in \mathbb{F}_p[x]$ of degree $d$, we can evaluate $P$ on every $\alpha \in \mathbb{F}_p$ in time $O(d + p \cdot \text{poly}(\log p))$.

**Proof:**
1.  **Reduction:** In $\mathbb{F}\_p$, $x^p - x \equiv 0$ for all elements (Fermat's Little Theorem). We first reduce $P$ to $\widetilde{P} \equiv P \pmod{x^{p-1}-1}$. This takes $O(d)$ time by folding coefficients: $a_i$ contributes to $a_{i \pmod{p-1}}$.
2.  **Evaluation via DFT:** The values $\widetilde{P}(1), \dots, \widetilde{P}(p-1)$ correspond exactly to the DFT of the coefficient vector of $\widetilde{P}$ over the cyclic group $\mathbb{F}_p^{\times}$ (we take the **generator** of the group which is a $p-1$-th primitive root of unity). Even if $p-1$ is not a power of $2$, we can use **Bluestein's Trick** combined with **Schönhage-Strassen** to compute this DFT in $O(p \log p \log \log p)$ time.
3.  **Zero Case:** $\widetilde{P}(0)$ is simply the constant term $a_0$, computed in $O(1)$. $\blacksquare$

### Multivariate Evaluation
This result extends powerfully to multivariate polynomials.

> [!important] Corollary
> Given $P \in \mathbb{F}\_p[x_1, \dots, x_m]$ with individual degrees $< d$, we can evaluate $P$ on every $\alpha \in \mathbb{F}\_p^m$ in time $O(d^m + m \cdot p^m \cdot \text{poly}(\log p))$.

**Proof:**
The first step is reduce the polynomial modulo $x_i^p -x_i$ for every $i$. This gives a new polynomial $\tilde{P}$ with individual degrees $<p$. This reduction takes $O(d^m)$ because it requires folding the $O(d^m)$ coefficients onto $p^m$ coefficients. Therefore, we assume $P=\tilde{P}$ has individual degrees $<p$.

We proceed by induction on $m$. **Base Case ($m=1$):** the previous corollary.

**Inductive Step:** Let $R = \mathbb{F}\_p[x_1, \dots, x_{m-1}]$. We can write $P$ as a polynomial in $x_m$ with coefficients in $R$:
$$ P(x_1, \dots, x_m) = \sum_{i=0}^{p-1} Q_i(x_1, \dots, x_{m-1}) \cdot x_m^i $$

1.  **Evaluate Coefficients:** By the induction hypothesis, evaluate each $Q_i$ on all points in $\mathbb{F}_p^{m-1}$.
    *   There are $p$ such polynomials ($Q_0, \dots, Q_{p-1}$).
    *   Cost: $p \times O((m-1) p^{m-1} \text{poly}(\log p)) = O((m-1) p^m \text{poly}(\log p))$.

2.  **Fix the Prefix:** For every fixed tuple $\vec{\alpha} = (\alpha_1, \dots, \alpha_{m-1}) \in \mathbb{F}\_p^{m-1}$, we have a univariate polynomial in $x_m$:
    $$ P_{\vec{\alpha}}(x_m) = \sum_{i=0}^{p-1} Q_i(\vec{\alpha}) \cdot x_m^i $$
    The coefficients $Q_i(\vec{\alpha})$ were computed in step 1.

3.  **Univariate Sweep:** For each of the $p^{m-1}$ vectors $\vec{\alpha}$, we evaluate $P_{\vec{\alpha}}$ on all $p$ values of $x_m$ using the fast univariate algorithm.
    *   Cost: $p^{m-1} \times O(p \cdot \text{poly}(\log p)) = O(p^m \text{poly}(\log p))$.

**Total Time:** Summing the steps yields $O(m \cdot p^m \cdot \text{poly}(\log p))$. $\blacksquare$

## References

1.  **Cooley, J. W., & Tukey, J. W.** (1965). [*An algorithm for the machine calculation of complex Fourier series*](https://doi.org/10.1090/S0025-5718-1965-0178586-1).
2.  **Bluestein, L.** (1970). [*A linear filtering approach to the computation of discrete Fourier transform*](https://ieeexplore.ieee.org/abstract/document/1162132).
3.  **Schönhage, A., & Strassen, V.** (1971). [*Schnelle Multiplikation großer Zahlen*](https://doi.org/10.1007/BF02242355).
4. **von zur Gathen, J. & Gerhard, J.** (2013). [*Modern Computer Algebra*](https://www.cambridge.org/core/books/modern-computer-algebra/DB3563D4013401734851CF683D2F03F0).