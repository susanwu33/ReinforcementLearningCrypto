# Reinforcement Learning for Crypto Market Stress

Five-slide presentation draft for April 30, 2026.

## Slide 1: Introduction and Professional Background

**Reinforcement Learning for Crypto Market Stress and Strategic Trading**

**Team:** [Name 1], [Name 2], [Name 3]

**Background:** Cornell Tech students combining machine learning, finance, data engineering, and policy/regulatory analytics.

**Project arc:** Across three assignments, we built a crypto RL framework that moved from a single rational investor to strategic trader archetypes, then to Bayesian asset-selection bandits.

**Core question:** How do learning-based investors respond to steady markets, price crashes, price surges, and strategic manipulation?

Speaker note: Replace the bracketed names and background with your actual professional profiles.

## Slide 2: Benefits to the End User

**For a regulator such as DOB**

- Stress-test crypto markets before real crises by simulating rational, manipulative, and herd-following behavior.
- Identify early-warning indicators: turnover velocity, momentum herding, volatility spikes, and cross-asset risk-off signals.
- Use SHAP explanations to see which market variables are driving agent behavior instead of treating the model as a black box.

**For a BlackRock-like asset manager**

- Compare trading policies under steady, crash, and surge regimes.
- Quantify execution risk, drawdown exposure, and turnover costs across RL policies.
- Use bandit models to allocate attention across crypto assets when short-term opportunities are noisy and similar.

**End-user value:** A decision-support layer for market surveillance, strategy testing, and adaptive risk management.

## Slide 3: Data Description

**Main RL assets:** AAVE, ETC, and SUI hourly crypto data.

**Reference assets:** BTC hourly data, S&P 500 futures (`ES=F`), and Gold futures (`GC=F`) for macro and cross-asset context.

**Raw observations**

- AAVE: 17,248 hourly rows
- ETC: 17,247 hourly rows
- SUI: 17,248 hourly rows
- BTC: 17,250 hourly rows
- ES futures: 11,213 hourly rows
- Gold futures: 11,258 hourly rows

**Processed feature observations**

- AAVE: 11,619 rows; ETC: 11,617 rows; SUI: 11,627 rows
- Modeling split per asset: about 8.1k train, 1.7k validation, 1.7k test rows

**Features:** 1h returns, 4h/24h momentum, 4h/24h volatility, 7d drawdown, cross-crypto returns, macro returns, trading-hour availability flags, and time-of-day sin/cos.

**Assignment 4 bandit data:** BTC, ETH, SOL, BNB, and USDC; binary reward is whether the next 8-hour return is positive; 100 replications with horizon T = 500.

## Slide 4: Methodology

**Environment**

- Gymnasium crypto trading environment with 3 actions: sell, hold, buy.
- Position changes in 0.5 increments; transaction cost = 5 bps per position change.
- Reward uses next-hour return to avoid leakage and scales by rolling volatility.
- Regimes are defined from 24-hour momentum: crash, steady, and surge.

**Assignment 2**

- TD Q-learning and DQN model a rational crypto investor under steady-state, crash, and surge conditions.
- Backtests evaluate cumulative return, Sharpe, turnover, and behavior by regime.

**Assignment 3**

- REINFORCE with baseline: two-layer policy network plus two-layer value network.
- DQN comparison: replay buffer, target network, epsilon-greedy exploration.
- Trader archetypes: rational arbitrageur, manipulator, herd-following retail.
- SHAP explains top features for each trader type and model.

**Assignment 4**

- Bayesian bandits for crypto asset selection: UCB1, Bayes-UCB, Thompson Sampling, and Greedy.
- Beta(3,3) prior over each asset's probability of positive 8-hour return.

## Slide 5: Key Findings

**1. Trader incentives change market risk.** Manipulators and herd-following retail produce higher turnover or momentum-chasing behavior, increasing stress events and volatility risk relative to rational behavior.

**2. Explainability matched the trader design.**

- Retail agents consistently relied on 24h momentum features.
- Manipulators emphasized short-term momentum, volatility, position, and turnover-linked signals.
- Rational traders used drawdown, time-of-day, cross-asset, and macro-risk features.

**3. Policy gradient was smoother, DQN was more sample-efficient but more abrupt.** REINFORCE's stochastic policy made exploration smoother, while DQN often adapted faster but with sharper policy shifts and higher turnover.

**4. Assignment 2 backtests were asset-specific.**

- ETC DQN outperformed TD on cumulative return and Sharpe.
- AAVE and SUI TD was less negative than DQN in the historical test window.
- This supports using regime-aware validation rather than assuming one RL method dominates.

**5. Bandit asset selection favored low exploration in this setup.** Greedy had the lowest final regret, 179.09, followed by Thompson Sampling, 179.84; UCB1 was highest, 187.06. With noisy binary crypto rewards and similar arms, exploration was immediately penalized by stepwise regret.

**Regulatory takeaway:** Surveillance should adapt to strategic behavior, especially when momentum herding and turnover spikes coincide with macro risk-off signals.
