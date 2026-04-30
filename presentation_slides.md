# Reinforcement Learning for Crypto Market Stress

Five-slide presentation draft for April 30, 2026.

## 3-Minute Storyline

**Main message:** Our project turns crypto trading data into a small stress-testing lab: instead of asking only which RL algorithm earns more return, we ask how different learned trader incentives can create market stress that regulators or asset managers should monitor.

**0:00-0:25 - Hook**

Crypto markets do not only move because prices change; they move because different participants react differently to the same signal. Our project studies that behavior with reinforcement learning. We started from a rational crypto trader, then added strategic archetypes like manipulators and herd-following retail, and finally used Bayesian bandits for short-term asset selection.

**0:25-0:55 - Data and setup**

We used hourly data for AAVE, ETC, and SUI, with BTC, S&P 500 futures, and Gold futures as cross-market context. The state includes short-term returns, 4-hour and 24-hour momentum, volatility, 7-day drawdown, macro returns, and time-of-day signals. The environment has three actions: sell, hold, and buy, with transaction costs and rewards based on next-hour returns, so the agent cannot look ahead.

**0:55-1:35 - What we built**

The key design choice was to train different trader types with different incentives. A rational agent maximizes risk-adjusted return. A manipulator is rewarded more for turnover and short-term movement. A retail-herding agent is pushed to follow 24-hour momentum. We trained both DQN and REINFORCE-style policy-gradient agents, then used SHAP-style feature attribution to check whether the learned behavior matched the intended trader design.

**1:35-2:25 - Most important finding**

The main finding is that trader incentives changed market risk more clearly than any single algorithm label. Manipulators produced much higher turnover across assets. For example, policy-gradient manipulator turnover was about 0.27 for AAVE, 0.37 for ETC, and 0.34 for SUI, while retail turnover stayed around 0.08 to 0.10. SHAP also supported the story: retail agents consistently relied on 24-hour momentum, while manipulators relied more on short-term signals, position, volatility, and timing. This means the framework is not just producing returns; it is producing interpretable stress signals.

**2:25-2:50 - Bandit result**

In the final bandit experiment, we treated BTC, ETH, SOL, BNB, and USDC as arms with a Beta(3,3) prior over the probability of a positive 8-hour return. Greedy had the lowest final cumulative regret, 179.09, just ahead of Thompson Sampling at 179.84, while UCB1 was highest at 187.06. The takeaway is that when crypto arms are noisy and similar, extra exploration can be costly under this regret definition.

**2:50-3:00 - Close**

Our final takeaway is regulatory: surveillance should not only track volatility after it appears. It should track behavior that creates fragility, especially turnover spikes, momentum herding, and macro risk-off signals appearing together.

## Slide Priority for a 3-Minute Version

1. Spend the most time on Slide 7: strategic behavior changes market risk.
2. Mention Slide 5 quickly as evidence that the dataset is real and broad.
3. Use Slide 6 only to explain the trader archetypes, not every model detail.
4. Treat the bandit result as a short final experiment, not the main story.

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
