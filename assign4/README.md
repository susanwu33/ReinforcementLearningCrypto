# Assignment 4: Bayesian Bandits for Crypto Asset Selection

## 1. Problem Setup

This submission uses the Crypto Track.

- Arms: BTC, ETH, SOL, BNB, USDC
- Number of arms: K = 5
- Unknown parameter for each arm: \(\theta_k = \Pr(\text{positive 8-hour return})\)
- Prior for every arm: Beta(3,3)

Binary observation:

\[
Y_t =
\begin{cases}
1, & \log(P_{t+1}/P_t) > 0 \\
0, & \text{otherwise}
\end{cases}
\]

- Horizon per run: T = 500

## 2. Implemented Algorithms

The following algorithms are implemented:

- UCB1: empirical mean with exploration bonus.
- Bayes-UCB: selects the arm with the highest posterior quantile.
- Thompson Sampling: samples from posterior and selects the best sample.
- Greedy: selects the arm with the highest posterior mean.

Implementation locations:

- `src/algorithms.py`
- `src/posterior.py`
- `src/simulation.py`

## 3. Experimental Setup

- 100 independent replications are run.
- Each replication samples a random contiguous window of length T = 500 from historical data.
- Cumulative regret is tracked over time.

We use the following regret definition:

\[
\text{Regret}_t = \max_k Y_t(k) - Y_t(a_t)
\]

This is stepwise realized regret, meaning that at each time step, the chosen arm is compared against the best realized reward among all arms in that same period.

- 95% confidence bands are computed as mean \(\pm 1.96 \times SE\).
- Bayes-UCB uses:
  `scipy.stats.beta.ppf(1 - 1/t, alpha, beta)`
- The Thompson Sampling information-ratio plot is an empirical approximation of the numerator term in Eq. 3.1 of Ghavamzadeh et al., rather than an exact computation.

## 4. How to Run

```bash
cd assign4
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scipy
python main.py
```

Required data files under `assign4/data/`:

- `data/BTC-USD_DataHr.csv`
- `data/ETH-USD_DataHr.csv`
- `data/SOL-USD_DataHr.csv`
- `data/BNB-USD_DataHr.csv`
- `data/USDC-USD_DataHr.csv`

## 5. Deliverables

- `figure1_cumulative_regret.png`: mean cumulative regret with 95% confidence bands
- `figure2_posterior_t50.png`: posterior distributions at t=50
- `figure2_posterior_t100.png`: posterior distributions at t=100
- `figure2_posterior_t200.png`: posterior distributions at t=200
- `figure2_posterior_t500.png`: posterior distributions at t=500
- `figure3_information_ratio.png`: empirical information-ratio trajectory for Thompson Sampling
- `final_regret_table.csv`: final cumulative regret summary (mean, SE, CI)

## 6. Results Summary

From `final_regret_table.csv`:

- UCB1: 187.06 (SE 0.738, CI [185.61, 188.51])
- Bayes-UCB: 180.66 (SE 0.775, CI [179.14, 182.18])
- Thompson Sampling: 179.84 (SE 0.889, CI [178.10, 181.58])
- Greedy: 179.09 (SE 0.746, CI [177.63, 180.55])

Interpretation:

- Greedy achieves the lowest cumulative regret, followed closely by Thompson Sampling.
- UCB-based methods perform worse in this setting.

This behavior arises because:

- Rewards are binary and noisy, derived from short-term price movements.
- The reward probabilities across assets are relatively similar, making exploration less beneficial.
- The use of stepwise realized regret penalizes exploration at every time step.

As a result, exploration-heavy strategies (UCB, Thompson Sampling) incur additional regret, while Greedy quickly exploits early observations without suffering large long-term penalties.

## 7. Prior Sensitivity Analysis

The Crypto Track uses a Beta(3,3) prior, which is centered at 0.5 and relatively uninformative. This choice reflects limited prior knowledge about short-term price movements and encourages exploration in early stages. As a result, algorithms must rely more heavily on observed data to distinguish between arms, which typically leads to slower initial convergence.

In contrast, the Housing Track uses informative priors derived from historical cap rates. When these priors are well aligned with the true reward distribution, they reduce posterior uncertainty early on and allow the algorithm to focus on promising arms more quickly. This leads to faster convergence and lower early-stage regret.

However, informative priors introduce sensitivity to misspecification. If the prior is inaccurate, it may bias early decisions toward suboptimal arms, increasing regret before sufficient data is collected to correct the posterior.

This behavior aligns with the results of Liu and Li (2015), which show that prior influence is strongest in the early phase of learning and diminishes as more observations are collected. Over time, the posterior becomes increasingly dominated by empirical data rather than the initial prior.

In our Crypto Track experiments, the weak Beta(3,3) prior leads to increased exploration early on. Under stepwise realized regret, this exploration is immediately penalized, which contributes to higher observed regret for exploration-heavy strategies.

## 8. Notes and Limitations

- The information-ratio figure is an empirical approximation, not an exact reconstruction of the theoretical quantity.
- Specifically, the numerator term is approximated using empirical samples across replications.
- Results depend on the sampled historical windows and may vary across datasets.
- USDC serves as a low-volatility baseline arm.
- The model uses binary reward signals, which simplifies real-world return dynamics.
