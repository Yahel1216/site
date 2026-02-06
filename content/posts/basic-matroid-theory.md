---
title: "Basic Matroid Theory"
date: 2026-01-10
slug: matroid-theory
draft: false
description: "Basics of Matroid Theory and Infinite Extensions"
tags: ["theory", "math", "matroids"]
categories: ["Theory", "Math"]
---
# Basic Matroid Theory and Infinite Extensions

**Prerequisites:**
*   **Linear Algebra:** Vector spaces, basis, dimension, and linear independence.
*   **Graph Theory:** Graphs, connected components, cycles, and spanning trees.
*   **Set Theory:** Basic cardinality and Zorn's Lemma (for the infinite section).

---

In this post, we will discuss the basics of **Matroid Theory**. While often taught merely as a theoretical framework for Greedy Algorithms, matroids are a rich combinatorial structure in their own right. We will start with the finite foundations and then explore how these definitions behave (and break) when extended to the infinite case.

## Motivation
### Vector Spaces
Many results in Linear Algebra rely on the highly convenient tool of a **basis**. In every introductory course, one learns that while a vector space may have many bases, they are all of **the same size**. This fact allows us to define the *dimension* of vector spaces.

The proof of this fact relies on a crucial observation:

> [!important] Observation
> Suppose $A,B$ are linearly independent sets with size $|A|<|B|$. Then there must be some vector $v\in B$ which is not contained in the span of $A$.

*Reasoning:* If not, $B\subseteq \mathrm{Span}(A)$, which implies that the dimension of the space spanned by $B$ is at most $|A|$. Since $|A| < |B|$, this contradicts the linear independence of $B$. Thus, there must exist $v\in B$ which is linearly independent from $A$, and we can add it to $A$ to form a larger independent set.

### Graphs
Graph algorithms often rely on the existence of a *Spanning Tree* (or a spanning *forest* if the graph is disconnected). For a graph $G=(V,E)$ with $n$ vertices and $c$ connected components, any spanning forest must have exactly $n-c$ edges.

We can prove this using a similar observation:

> [!important] Observation
> Suppose $S,T \subset E$ define two forests (acyclic subgraphs) with $|S|< |T|$. Then there must exist an edge $e\in T$ such that adding $e$ to $S$ doesn't close any cycle.

*Reasoning:* If no such edge existed, then the vertices touched by edges in $T$ ($V_{T}$) would be "covered" connectivity-wise by edges in $S$. Since $|S|<|T|$ and both are forests, a counting argument on connected components leads to a contradiction. We conclude that $e$ can be added to $S$ to obtain a bigger forest.

Notice the pattern: in both cases, there is a fundamental property allowing us to "exchange" elements from a larger independent set into a smaller one.

## Matroids
Matroids are the combinatorial structure that captures this exact notion of "independence", abstracting away the specific details of vectors or edges.

> [!caution] Definition
> A matroid is a pair $\mathcal{M}=\langle S,\mathcal{I} \rangle$, where $S$ is the **ground set** and $\mathcal{I}\subset 2^{S}$ is the collection of **independent sets**, satisfying:
> 1.  $\mathcal{I}\neq\emptyset$.
> 2.  **Hereditary:** If $A\in \mathcal{I}$, then every subset $A'\subseteq A$ satisfies $A'\in \mathcal{I}$.
> 3.  **Exchange:** If $A,B\in \mathcal{I}$ and $|A|<|B| < \infty$, then there exists $a\in B\setminus A$ such that $A\cup \lbrace a\rbrace \in \mathcal{I}$.

For now, we assume the ground set and independent sets are finite. We will discuss the infinite case later. First, let us prove that this structure forces all maximal independent sets to have the same size.

> [!important] Lemma
> If $A,B\in \mathcal{I}$ are maximal (meaning there is no $C\in \mathcal{I}$ such that $A\subsetneq C$), then $|A|=|B|$.

**Proof.** Suppose otherwise, i.e., that $|A|>|B|$ (we can switch roles if the opposite holds). By the Exchange property, there exists $a\in A\setminus B$ such that $B\cup \lbrace a\rbrace \in \mathcal{I}$. This contradicts the maximality of $B$. $\blacksquare$

This lemma implies that the following concept is well-defined:

> [!caution] Definition
> A **basis** of a finite matroid $\mathcal{M}=\langle S,\mathcal{I} \rangle$ is a maximal element in $\mathcal{I}$. Every basis has the same size, allowing us to define the **rank** of $\mathcal{M}$, denoted $r(\mathcal{M})$, as the size of any basis.

### Examples
1.  **The Vector Matroid:** For a vector space $V$, let $S\subset V$ be a finite set of vectors, and let $\mathcal{I}$ be the collection of linearly independent subsets of $S$.
2.  **The Graphic Matroid:** For a graph $G=(V,E)$, let $S=E$ and define $\mathcal{I}$ as the collection of edge sets that induce a forest (contain no cycles).

### Circuits
> [!caution] Definition
> A **dependent** set in a matroid is a set $A\subset S$ such that $A\notin \mathcal{I}$. A **circuit** is a **minimal dependent set**. A set $C$ is a circuit if it is dependent, but every proper subset $C'\subsetneq C$ is independent.

Circuits act as minimal **proofs** of dependence:
1.  In the vector matroid, if $v_3 = v_1 + v_2$, then $\lbrace v_1, v_2, v_3\rbrace $ is a circuit. Note that unlike bases, **circuits are not necessarily of the same size**. A set $\lbrace v, 2v\rbrace $ is a circuit of size 2, while $\lbrace v_1, v_2, v_1+v_2\rbrace $ is a circuit of size 3.
2.  In the graphic matroid, a circuit corresponds exactly to a **simple cycle**.

## The Dual Matroid
> [!caution] Definition
> Given a **finite** matroid $\mathcal{M}=(S,\mathcal{I})$, we define the **dual matroid** $\mathcal{M}^{\*}=(S,\mathcal{I}^{\*})$ by:
> $$A\in \mathcal{I}^{\*} \iff S\setminus A \text{ contains a basis of }\mathcal{M}.$$

We use the prefix "co-" for dual terms: dual bases are **cobases**, dual independent sets are **coindependent**, and dual circuits are **cocircuits**.
Since a basis $A$ of $\mathcal{M}^{\*}$ satisfies $S\setminus A=B$ for a basis $B$ of $\mathcal{M}$, the rank is given by $r(\mathcal{M}^{\*})=|S|-r(\mathcal{M})$.

> [!important] Lemma
> If $\mathcal{M}$ is a finite matroid, then $\mathcal{M}^{\*}$ is also a matroid.

**Proof.** The fact that $\mathcal{I}^{\*}$ is non-empty and hereditary is straightforward. We focus on the **Exchange Property**.
Let $A,B\in \mathcal{I}^{\*}$ with $|A|>|B|$. We must show there is $a\in A\setminus B$ such that $B\cup \lbrace a\rbrace \in \mathcal{I}^{\*}$, i.e., that $S\setminus (B\cup \lbrace a\rbrace )$ contains a basis of $\mathcal{M}$.

Let $C,D$ be bases of $\mathcal{M}$ such that $C\subseteq S\setminus B$ and $D\subseteq S\setminus A$. Note that $D\setminus B\in \mathcal{I}$ (hereditary), so there exists a basis $E$ such that $D\setminus B\subseteq E$.
We claim $A\setminus B \not\subset E$. Assume the converse ($A\setminus B\subset E$). Then:
$$|D|= |D\cap B|+ |D\setminus B| \overset{D\subseteq S\setminus A}{\le} |B\setminus A|+ |D\setminus B| \overset{|A|>|B|}{<} |A\setminus B|+ |D\setminus B|\le |E|$$
(The last inequality holds because $A \cap D = \emptyset$ and $A\setminus B, D\setminus B\subseteq E$ by hypothesis).
This implies $|D| < |E|$, contradicting that all bases have the same size.

We conclude $A\setminus B \not\subset E$, hence there exists $a\in (A\setminus B) \setminus E$. Thus $E \subseteq S \setminus (B \cup \lbrace a\rbrace )$, meaning $B \cup \lbrace a\rbrace  \in \mathcal{I}^*$. $\blacksquare$

### Dual Graph Matroid
Recall that in the graphic matroid, independent sets are forests and circuits are simple cycles. What is the dual?

*   **Coindependent sets** are subsets $A\subset E$ whose removal leaves a spanning forest (does not break connectivity).
*   **Cocircuits** are minimal subsets $A\subset E$ whose removal breaks connectivity in some connected component.

If $G$ is connected, cocircuits are exactly the **minimal cuts** of the graph.

**Example:**
Consider the following graph: 
![graph](/images/graph1.png)
A possible basis (Spanning Tree) is: $$\lbrace (4,3),(5,3),(3,0),(1,0),(2,0)\rbrace $$
A possible cobasis (edges not in the tree) is: $$\lbrace (5,4),(2,1)\rbrace $$
Two possible cocircuits (minimal cuts) are:
1.  $\lbrace (5,4),(4,3)\rbrace $ (Isolating vertex 4)
2.  $\lbrace (3,0)\rbrace $ (A bridge)

## The Infinite Case
In linear algebra, the standard definition of linear independence applies to finite sets. To handle infinite dimensions, we extend the definition:

> [!caution] Definition
> Let $V$ be a vector space. A set $S\subset V$ is linearly independent if every **finite** subset $S'\subset S$ is linearly independent.

Using Zorn's lemma, one can show every vector space has a basis. However, this algebraic basis is often less useful than topological bases (like Hilbert bases) which allow for convergence arguments.

We can attempt to define infinite matroids similarly:

> [!caution] Definition (Infinite Matroid Attempt)
> Let $\mathcal{M}=(S,\mathcal{I})$. We require the standard matroid axioms, plus:
> $A\in \mathcal{I}$ if and only if every finite subset $A'\subset A$ satisfies $A'\in \mathcal{I}$.

A consequence of this definition is that **circuits must be finite**. If a circuit were infinite, every finite subset of it would be independent (by minimality), which would make the whole set independent by our definition—a contradiction.

I turns out that this definition, while appealing for its simplicity, is not very useful, because certain concepts like that of the dual matroid fail to translate to the infinite case. We will show this with the next canonical example.

### The Infinite Ladder Failure
Let's see where this definition breaks down. Consider the **Ladder Graph**:
$$G=(V,E), \quad V=\mathbb{Z}\times\lbrace 0,1\rbrace $$
$$E = E_1 \cup E_2$$
Where $E_1 = \lbrace \lbrace (i,x),(i+1,x)\rbrace  \mid i\in\mathbb{Z}, x\in \lbrace 0,1\rbrace  \rbrace$ are the "rails" and $E_2 = \lbrace  \lbrace (i,0),(i,1)\rbrace  \mid i\in\mathbb{Z} \rbrace $ are the "rungs". The level (rung) is determined by $i\in \mathbb{Z}$ while the side (rail) is determined by $x\in \lbrace 0,1\rbrace$.

1.  **Independent Set:** $A = E_1 \cup \lbrace (0,0),(0,1)\rbrace $. This is the set of all rails plus one rung. It contains no cycles, and any finite subset is a forest.
2.  **Coindependent Set:** $B = A^c = E_2 \setminus \lbrace (0,0),(0,1)\rbrace $. This is the set of all rungs except one. Removing these leaves $A$ intact, so the graph remains connected.

Now consider the set of **all** rungs, $E_2$.
In the dual context, $E_2$ is a **cocircuit**. Why?
-   It is a cut: Removing all rungs disconnects the two rails.
-   It is minimal: If we keep even one rung $e \in E_2$, the graph is connected.

**The Problem:** $E_2$ is a **cocircuit**, but it is **infinite**.
We established that under our definition, circuits must be finite. Thus, the dual of this infinite matroid **is not a matroid**.

Finding a definition of infinite matroids that preserves duality was a long-standing open problem, resolved by *Bruhn et al., 2013*.

## Algorithmic Applications
Matroids are intimately tied to **Greedy Algorithms**. An optimization algorithm is "greedy" if it constructs a solution step-by-step, making the locally optimal choice at each step with the hope of finding a global optimum.

**Example (Fractional Knapsack).** Given items with values and weights $\lbrace(v_{i},w_{i})\rbrace _{i=1}^{N}$ and a capacity $W$, we want to maximize total value. We are allowed to take fractions of items.

*Greedy Strategy:* Sort items by value-per-weight ratio. Take as much of the best item as possible, then the next, until the capacity $W$ is full.


For matroids, we can solve the following problem: Given a finite matroid $\mathcal{M}=\langle S,\mathcal{I} \rangle$ and weights $\mu:S\to \mathbb{R}_{+}$, find a **basis** of maximal weight.

> [!caution] Greedy Matroid Algorithm
> 1. Sort $S$ in descending order: $\mu(s_{1})\ge \mu(s_{2})\ge\ldots\ge\mu(s_{n})$.
> 2. Initialize $A\gets \emptyset$.
> 3. For $i=1,\ldots,n$:
>    - If $A\cup \lbrace s_{i}\rbrace \in \mathcal{I}$, update $A\gets A\cup \lbrace s_{i}\rbrace $.
> 4. Return $A$.

### Proof of Optimality
We first note that the algorithm returns a basis (if it returned a smaller independent set, we could extend it by the Exchange property, contradicting the logic of the loop).

> [!important] Lemma
> The algorithm returns a basis of maximal weight.

**Proof.** Let $A$ be the solution returned by the algorithm and $B$ be an optimal basis with $\mu(B) > \mu(A)$.
Write $A=(a_1, \dots, a_r)$ and $B=(b_1, \dots, b_r)$ sorted by weight.
Since $\mu(B) > \mu(A)$, there must be an index $i$ such that $\mu(b_i) > \mu(a_i)$. Let $i$ be the **minimal** such index.

Consider the element $b_i$. Since $i$ is minimal, for all $j < i$, we have $\mu(a_j) \ge \mu(b_j) \ge \mu(b_i)$. Thus, when the algorithm considered $b_i$, it had already considered and selected $\lbrace a_1, \dots, a_{i-1}\rbrace $ (or elements with even higher weights).

By the Exchange property, since $|\lbrace b_1, \dots, b_i\rbrace | > |\lbrace a_1, \dots, a_{i-1}\rbrace |$, we can add an element from the first set to the second. Specifically, we can add $b_i$ to $\lbrace a_1, \dots, a_{i-1}\rbrace $ while maintaining independence (if $b_i$ was dependent on previous elements, it wouldn't be in the independent set $\lbrace b_1 \dots b_i\rbrace $).

Since $b_i$ could have been added, and it has higher weight than $a_i$ (and subsequent elements), the greedy algorithm **would** have added it. This contradicts our assumption that the algorithm produced $A$. $\blacksquare$

**Remark.** The runtime is dominated by sorting ($\Omega(n \log n)$) and the $n$ calls to the **Independence Oracle** (checking $A \cup \lbrace s_i\rbrace  \in \mathcal{I}$).

### Kruskal's Algorithm
Kruskal's Algorithm for Minimum Spanning Trees is a classic manifestation of the Greedy Matroid Algorithm applied to the **Graphic Matroid** (sorting edges by weight). The independence oracle checks for cycles, which can be implemented efficiently using a **Union-Find** data structure.

## References
1.  **Oxley, J.** (2011). [*Matroid Theory*](https://global.oup.com/academic/product/matroid-theory-9780199603398).
2.   **Bruhn, H., Diestel, R., Kriesell, M., Pendavingh, R., & Wollan, P.** (2013). [*Axioms for infinite matroids*](https://arxiv.org/abs/1003.3919).