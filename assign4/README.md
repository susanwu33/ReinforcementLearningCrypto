# Crypto Bandit Experiment — Multi-Armed Bandits on Cryptocurrency Returns


## Analysis: Prior Sensitivity and Convergence

In this experiment, we compare the convergence behavior under an informative prior (Track A) and an uninformative prior (Track B) using bandit algorithms on crypto return data. Empirically, both priors eventually lead to similar long-term performance, but they differ in their early-stage behavior.

The informative prior accelerates convergence in the initial phase by encoding stronger prior beliefs about the success probabilities of each arm. This allows the algorithm to reduce uncertainty more quickly and make more confident decisions with fewer observations. As a result, early regret is typically lower under Track A. In contrast, the uninformative prior starts with higher uncertainty and treats all arms more equally, leading to more exploration and slower early convergence.

However, as more data is observed, the influence of the prior diminishes. This is clearly reflected in the posterior distributions, which become increasingly concentrated and similar across both tracks over time. By later stages (e.g., t=500), the posterior is dominated by observed data rather than the initial prior.

These findings align with the prior sensitivity bounds in Liu & Li (2015), which show that the effect of prior misspecification on regret is bounded and decreases over time. In particular, the difference in cumulative regret induced by different priors grows sublinearly (typically logarithmically), meaning that priors mainly impact early performance but not asymptotic behavior.

Overall, the informative prior improves sample efficiency in the early stage, while the uninformative prior provides a more robust and unbiased starting point. In practice, when reliable prior knowledge is available, it can significantly accelerate learning, but its long-term impact remains limited as data accumulates.



## 1. Overview

This project implements and evaluates several multi-armed bandit algorithms in the context of cryptocurrency trading. Each cryptocurrency (BTC, ETH, SOL, BNB, USDC) is treated as an arm, and the goal is to sequentially select the asset with the highest probability of a positive return.

At each time step, the agent selects one asset and observes a binary reward:

* (Y_t = 1) if the 8-hour log return is positive
* (Y_t = 0) otherwise

We compare four algorithms:

* UCB1 (frequentist baseline)
* Bayes-UCB
* Thompson Sampling
* Greedy (baseline)

The evaluation metric is **cumulative regret**, computed relative to the best-performing arm in hindsight within each sampled trajectory.

---

## 2. Project Structure

```
project/
│
├── data/                  # Raw crypto price CSV files
│   ├── BTC-USD_DataHr.csv
│   ├── ETH-USD_DataHr.csv
│   ├── SOL-USD_DataHr.csv
│   ├── BNB-USD_DataHr.csv
│   └── USDC-USD_DataHr.csv
│
├── src/
│   ├── data_loader.py     # Data preprocessing → Bernoulli outcomes
│   ├── posterior.py       # Beta posterior implementation
│   ├── algorithms.py      # Arm selection logic
│   ├── simulation.py      # Bandit simulation + regret tracking
│   ├── plots.py           # Figures 1–3
│   └── table.py           # Final summary table
│
├── main.py                # Entry point
├── final_regret_table.csv # Output table
├── figure1_cumulative_regret.png
├── figure2_posterior_*.png
├── figure3_information_ratio.png
└── README.md
```

---

## 3. Methodology

### 3.1 Data Processing

Implemented in 

* Hourly price data is resampled into **non-overlapping 8-hour intervals**
* Log returns are computed:

$$
r_t = \log(P_t) - \log(P_{t-1})
$$

* Binary reward:

$$
Y_t = \mathbb{1}(r_t > 0)
$$

All assets are aligned into a single matrix 

$$
Y \in {0,1}^{T \times K}
$$

---

### 3.2 Bayesian Modeling

Implemented in 

Each arm uses a **Beta-Bernoulli model**:

* Prior:
  
$$
\theta_k \sim \text{Beta}(3,3)
$$

* Update:

  * success → $\alpha + 1$
  * failure → $\beta + 1$

This enables:

* Posterior mean (Greedy)
* Sampling (Thompson)
* Quantiles (Bayes-UCB)

---

### 3.3 Algorithms

Implemented in 

| Algorithm         | Strategy                               |
| ----------------- | -------------------------------------- |
| Greedy            | Select arm with highest posterior mean |
| UCB1              | Empirical mean + exploration bonus     |
| Bayes-UCB         | Posterior upper quantile               |
| Thompson Sampling | Sample from posterior and select max   |

Note:

* UCB1 uses **empirical mean**, not Bayesian mean
* Each arm is forced to be explored at least once

---

### 3.4 Simulation Design

Implemented in 

Key components:

#### Randomized Replications

To ensure valid confidence intervals:

* Each replication samples a **random contiguous window of length 500**
* Avoids deterministic repetition

#### Regret Definition

We follow the assignment definition:

$$
\text{Regret}(t) = \sum_{i=1}^{t} [r(Y_i(a^*)) - r(Y_i(a_i))]
$$

Where:

* $a^*$ = best arm (estimated using empirical mean within each window)
* $a_i$ = chosen arm

---

### 3.5 Information Ratio Approximation

Also in 

We estimate:

$$
(E[\Delta_t])^2
$$

This corresponds to the **numerator** of the information ratio in Eq. 3.1 (Ghavamzadeh et al.).

Note:

> We do not estimate the denominator (I_t), so this is an approximation.

---

## 4. Experiments

### Settings

* Time horizon per run: **T = 500**
* Total data used: **T_total = 1000**
* Replications: **100 independent runs**

---

## 5. Results

### Figure 1 — Cumulative Regret

* Shows mean regret ± 95% CI across replications
* Thompson Sampling achieves the lowest regret
* UCB1 shows highest regret due to over-exploration

---

### Figure 2 — Posterior Distributions

* Shows Beta distributions at:

  * t = 50, 100, 200, 500
* Demonstrates convergence of posterior beliefs
* Clear separation between strong and weak arms

---

### Figure 3 — Information Ratio Numerator

* Shows $(E[\Delta_t])^2$ over time
* Decreases as the algorithm learns
* Indicates reduced uncertainty and improved decisions

---

### Final Regret Table

Generated using 

Includes:

* Final cumulative regret
* Standard error (SE)
* 95% confidence intervals

---

## 6. Key Observations

* **Thompson Sampling performs best**, consistent with theory
* **UCB1 over-explores**, leading to higher regret
* **Greedy can perform competitively** when early signals are informative
* Posterior distributions converge as more data is observed
* Variance across replications confirms stochastic behavior

---

## 7. How to Run

```bash
pip install numpy pandas matplotlib scipy
python main.py
```

This will generate:

* `final_regret_table.csv`
* Figure 1, 2, 3 plots

---

## 8. Notes

* All algorithms share the same data preprocessing pipeline
* Replications are randomized via time-window sampling
* Results are reproducible with fixed random seeds (optional extension)

---

## 9. Summary

This project demonstrates how bandit algorithms can be applied to financial time-series decision-making. The results align with theoretical expectations and highlight the effectiveness of Bayesian approaches, particularly Thompson Sampling, in balancing exploration and exploitation.

---
