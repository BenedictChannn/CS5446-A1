# CS5446 Assignment 2: Two-Player Zero-Sum Games (Normal Form)

> My notes / walkthrough — CS5446 AI Planning and Decision Making (NUS)

## What I did here

I worked through the **normal-form game** for **Tian Ji’s horse racing**: two players each assign **n** horses to **n** races (a **permutation** = one pure strategy). I built intuition for why the payoff matrix is **n! × n!** (for **n = 6**, that’s **720 × 720**), implemented **best response**, a **saddle-point residual** to measure distance from **Nash**, then **regret matching** so **time-averaged** mixed strategies approximate an equilibrium without me solving the thing by hand.

The story is basically **Colonel Blotto**-flavoured: assignment matters more than raw strength alone. The notebook uses **Olympic 100 m** times (2020 vs 2016) for speeds and **V = [1,…,6]** for race weights.

---

## What’s in this folder

```
assignment_2/
├── environment.yml           # Conda env I used (numpy, matplotlib, jupyter, …)
├── Assignment2-NFG.ipynb     # Where all the code lives
└── README.md                 # This file
```

---

## How I think about the pipeline

I start from speeds and the list of **all permutations** as actions, then:

```
compute_payoff_matrix  →  A
compute_payoff(A, x, y)     # P1’s expected payoff xᵀAy
compute_best_response(A, y)
compute_saddle_point_residual(A, x, y)   # my “are we at Nash?” check
get_strategy_from_regret / update_regrets   # RM step
compute_nash_equilibrium   # loops RM, averages strategies, plots SPR
```

When the notebook plots **SPR**, that’s the **saddle-point residual** of the **running average** `(x_avg, y_avg)` — how far that average profile is from equilibrium. Watching it drift down on a log scale was my sanity check that RM was doing something sensible.

---

## Tasks (my checklist)

| Piece | Coursemology | What it is | Status |
|--------|----------------|------------|--------|
| Payoff matrix + `compute_payoff` | — | Given in the notebook; I used them everywhere | Done |
| **Task 1** | Q1 | `compute_best_response` — I take **`A @ y`**, **argmax**, one-hot | Done |
| **Task 2** | Q2 | `compute_saddle_point_residual` — I used **max(Ay) − min(xᵀA)** style residual | Done |
| **Task 3** | Q3 | `get_strategy_from_regret` — positive regrets normalized, else uniform | Done |
| **Task 4** | Q4 | `update_regrets` — I used **`compute_payoff`** for the baseline **`u1`**, then **`A@y`** and **`x@A`** for counterfactuals | Done |
| Final run | — | Assert SPR below tolerance after enough iterations | Done |

**Grading (from the assignment brief):** Task 1 (3), Task 2 (2), Task 3 (6), Task 4 (4).

---

*Assignment 2 finished; all implementation tasks done.*
