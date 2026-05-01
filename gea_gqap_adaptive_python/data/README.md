# Logistics Optimization Model with Genetic Algorithm

## Problem Overview

This project defines and solves a logistics assignment and sequencing problem using a genetic algorithm (GA).

The objective is to assign J tasks (or customers) to I facilities (or agents) while minimizing total cost, respecting capacity constraints, and accounting for spatial relationships between both facilities and tasks.

The problem combines:
- Assignment decisions (which facility serves which task)
- Sequencing decisions (order in which tasks are processed)
- Distance-based costs (between facilities and between tasks)

---

## Sets and Indices

I — number of facilities (agents, depots)  
J — number of tasks (customers, jobs)

---

## Model Parameters

### Assignment Cost (cij)

cij ∈ ℝ^(I×J)

cij[i, j] is the cost of assigning task j to facility i.

This may represent transportation cost, service cost, or processing cost.

---

### Resource Consumption (aij)

aij ∈ ℝ^(I×J)

aij[i, j] is the amount of resource consumed if facility i serves task j.

This can represent time, workload, or energy.

---

### Facility Capacity (bi)

bi ∈ ℝ^I

bi[i] is the total capacity of facility i.

Typical constraint:
sum over j of aij[i, j] * xij[i, j] ≤ bi[i]

---

### Distance Between Facilities (DIS)

DIS ∈ ℝ^(I×I)

DIS[i, k] is the Euclidean distance between facilities i and k.

Used for modeling interactions or coordination costs between facilities.

---

### Distance Between Tasks (F)

F ∈ ℝ^(J×J)

F[j, k] is the distance between tasks j and k.

Used in sequencing and routing decisions.

---

## Decision Representation (Genetic Algorithm)

Each solution (Individual) consists of:

### permutation

An array of size J representing the order of tasks.

This defines the sequence in which tasks are processed.

---

### xij

Matrix of size (I×J)

xij[i, j] = 1 if task j is assigned to facility i, otherwise 0.

---

### cost

A scalar value representing the total objective value of the solution.

---

### cvar

Vector of size I representing a risk or load-related measure per facility.

---

## Algorithm Configuration

### Core Parameters

iterations — number of generations  
population_size — number of individuals in the population  

---

### Genetic Operators

crossover_rate — probability of crossover  
mutation_rate — probability of mutation  

---

### Scenario-Based Parameters

scenario_crossover_rate — crossover probability for scenario-based operators  
scenario_mutation_rate — mutation probability for scenario-based operators  

enable_scenario — tuple of booleans indicating which scenarios are active  

---

### Probabilities

p_fixed_x — probability of keeping assignment fixed  
p_scenario1, p_scenario2, p_scenario3 — probabilities of applying different strategies  

---

### Additional Controls

time_limit — maximum runtime in seconds  
deduplicate — whether to remove duplicate individuals  

---

## Adaptive Algorithm Configuration

Extends the base configuration with adaptive parameters:

adaptive_epsilon — small threshold for stability  
adaptive_alpha — learning rate  
adaptive_lambda_min — minimum adaptive parameter value  
adaptive_lambda_max — maximum adaptive parameter value  

---

## Statistics

### AlgorithmStats

best_cost_trace — list of best cost values over iterations  
contribution_rate — contribution of different operators  

---

### AdaptiveAlgorithmStats

lambda_history — evolution of adaptive parameter  
delta_history — update magnitudes  

---

## Results

### AlgorithmResult

best_cost — best objective value found  
best_individual — best solution  
population — final population  
stats — collected statistics  
elapsed_time — total runtime  

---

## Data Description (Example Instance)

Facilities (I = 15) have coordinates (X, Y), used to compute DIS.

Tasks (J = 20) have coordinates (XX, YY), used to compute F.

Distances are Euclidean:
distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)

---

## Problem Interpretation

This model combines several classical optimization problems:

1. Assignment Problem  
Minimize sum over i, j of cij[i, j] * xij[i, j]

2. Capacity Constraints  
sum over j of aij[i, j] * xij[i, j] ≤ bi[i]

3. Sequencing / Routing Component  
Defined by permutation and task distance matrix F

4. Spatial Interaction  
Defined by facility distance matrix DIS

---

## Summary

The problem can be described as:

Assign tasks to facilities and determine their processing order while minimizing cost, respecting capacity constraints, and considering spatial distances.

This makes it a hybrid problem combining elements of:
- Assignment problems
- Knapsack constraints
- Traveling salesman / routing problems
