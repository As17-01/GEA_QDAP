# Logistics Optimization Model with Genetic Algorithm

## Problem Overview

This project defines and solves a logistics assignment and sequencing problem using a genetic algorithm (GA).

The objective is to assign $J$ tasks (customers) to $I$ facilities (agents) while minimizing total cost, respecting capacity constraints, and accounting for spatial relationships.

The problem combines:
- Assignment decisions (which facility serves which task)
- Sequencing decisions (order of tasks)
- Distance-based costs

---

## Sets and Indices

- $I$ — number of facilities  
- $J$ — number of tasks  

---

## Model Parameters

### Assignment Cost ($c_{ij}$)

$c_{ij} \in \mathbb{R}^{I \times J}$

$c_{ij}$ is the cost of assigning task $j$ to facility $i$.

---

### Resource Consumption ($a_{ij}$)

$a_{ij} \in \mathbb{R}^{I \times J}$

$a_{ij}$ is the resource usage if facility $i$ serves task $j$.

---

### Facility Capacity ($b_i$)

$b_i \in \mathbb{R}^I$

$b_i$ is the capacity of facility $i$.

Capacity constraint:

$$
\sum_{j=1}^{J} a_{ij} x_{ij} \leq b_i \quad \forall i
$$

---

### Distance Between Facilities (DIS)

$DIS \in \mathbb{R}^{I \times I}$

$DIS_{ik}$ is the distance between facilities $i$ and $k$.

---

### Distance Between Tasks ($F$)

$F \in \mathbb{R}^{J \times J}$

$F_{jk}$ is the distance between tasks $j$ and $k$.

---

## Decision Variables

### Assignment Variable ($x_{ij}$)

$$
x_{ij} =
\begin{cases}
1 & \text{if task } j \text{ is assigned to facility } i \\
0 & \text{otherwise}
\end{cases}
$$

---

### Permutation

A permutation of size $J$ defining the order of tasks.

---

## Objective (Conceptual)

Minimize total cost:

$$
\min \sum_{i=1}^{I} \sum_{j=1}^{J} c_{ij} x_{ij}
$$

Optionally including sequencing costs:

$$
\sum_{j,k} F_{jk} \cdot \text{order interaction}
$$

---

## Genetic Algorithm Representation

Each solution (Individual) contains:

- `permutation` — ordering of tasks
- `xij` — assignment matrix
- `cost` — objective value
- `diversity` — hidden objective value
- `cvar` — load/risk per facility

---

## Data Description

Facilities and tasks are placed in 2D space.

Distances are computed as:

$$
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

---

## Summary

The problem is:

> Assign tasks to facilities and determine their order while minimizing cost under capacity constraints and spatial effects.

It combines:
- Assignment problem
- Knapsack constraints
- Routing / sequencing (TSP-like)
