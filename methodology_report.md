# Reinforcement Learning Crypto Framework: Assignment 3 Methodology & Results

## 1. Introduction

This report outlines the methodology behind extending a reinforcement learning (RL) crypto-trading environment to incorporate Policy Gradient (REINFORCE) with a baseline, contrasting its performance with Deep Q-Network (DQN) models. Furthermore, it details the design and deployment of three unique strategic trader archetypes to assess systematic market impacts.

By analyzing Shapley values across different traders, we provide a mathematically rigorous explanation for feature importance. Finally, we model and visualize subsequent market changes like crisis frequency to aid regulators in adapting to multi-agent strategic interactions in digital asset markets.

---

## 2. Refactoring and Architecture

To ensure scalability and maintainability, the initial prototype notebooks (`train.ipynb`, `features_ver2.ipynb`, etc.) were refactored into a modular Python codebase:

- **`/src/data/`**: Manages the construction of unified hourly dataset pipelines, imputes missing macro observations (e.g., S&P 500, Gold futures), calculates technical momentum, and applies z-score normalization on a strict train/val/test split.
- **`/src/env/crypto_env.py`**: Houses `CryptoInvestEnv`, a Gymnasium-compatible reinforcement learning environment that has been expanded to support adaptable reward systems depending on the trader archetype.
- **`/src/agents/`**: Contains the network architectures and training loops for the DQN value-based approach and the REINFORCE stochastic policy-based approach.
- **`/src/evaluation/`**: Automates SHAP-based model explainability and handles simulated population-level plotting logic.

---

## 3. RL Agents: Policy Gradient (REINFORCE) vs. DQN

Our core framework evaluates two distinct paradigms for deep reinforcement learning.

### 3.1 Policy Gradient (REINFORCE) with Baseline

The implemented PG method utilizes an actor-critic paradigm using two separate two-layer neural networks:

- **Actor (Policy Net)**: Output is a softmax probability distribution over 3 discrete actions (sell, hold, buy).
- **Critic (Value Net)**: Learns the expected discounted cumulative return of a state, serving as a baseline.
  We subtract the Value Net's estimate from our Monte Carlo returns to calculate the _advantage function_. This baseline greatly minimizes variance in the gradient updates, maintaining stability in highly stochastic cryptocurrency price series. Returns were also batch-standardized (zero mean, unit variance) to further stabilize backpropagation.

### 3.2 Deep Q-Network (DQN)

The DQN paradigm utilizes a single two-layer target-network architecture. A replay buffer decoupled observations to mitigate catastrophic forgetting and a target network updating every $N$ steps prevented oscillating Q-values.

### 3.3 Contrast and Tradeoffs

- **Exploration**: DQN explores via epsilon-greedy strategy, which often leads to abrupt regime shifts in policy space. REINFORCE maintains a continuous probability space, allowing for smoother exploration and often resulting in policies with less erratic turnover.
- **Convergence speed**: DQN generally exhibited faster sample efficiency due to off-policy replay buffer usage, but REINFORCE effectively managed non-markovian temporal abstractions better natively.

---

## 4. Multi-Trader Market Simulations

To model varying systematic behaviors within the crypto ecosystem, we constructed three distinct agent archetypes. The environment's reward mechanism was systematically altered depending on the assigned `--trader_type`.

### 4.1 Rational Arbitrageurs

- **Objective**: Maximize risk-adjusted log returns.
- **Reward Mechanism**: The native Sharpe ratio `(net_port_r - R_f) / safe_vol`, penalized additionally by macro risk components if Bitcoin and S&P 500 were both negative.
- **Design Philosophy**: Acts as the traditional "efficient-market" participant, smoothing out large discrepancies across assets.

### 4.2 Manipulators (Front-runners and Wash Traders)

- **Objective**: Exploit short-term momentum cascades and inflate trading volume.
- **Reward Mechanism**: Weighted structurally to ignore broad portfolio drawdowns and instead focus heavily on absolute order volume (`delta_pos`) and immediate short-term ticks: `reward = base_reward * 0.5 + delta_pos * 0.1`
- **Design Philosophy**: Incentivized to rapidly oscillate positions without regard for long-term holding Sharpe.

### 4.3 Herd-following Retail

- **Objective**: Minimize deviation from prevailing momentum ("FOMO").
- **Reward Mechanism**: The environment artificially zeroes-out the reward if the agent takes a position counter to the 24-hour momentum (`mom24h`). `herd_penalty = -abs(new_pos - target_pos(mom))`
- **Design Philosophy**: Reflects late-adopters that buy into surges and panic-sell during crashes.

---

## 5. Shapley Explanations (Explainable RL)

To peek into the "black box" of the models, we utilized `shap.DeepExplainer` on both the REINFORCE soft-policy outputs and the DQN Q-value outputs to calculate absolute marginal feature importance. We passed random historical states as background references and extracted SHAP values on the test manifold.

- **Rational Arbitrageurs Focus**: SHAP values uniformly ranked Volatility (`vol24h`), Macro Futures (`ES_r1h`), and short term 4-hour technical momentum indicators highest. The agent properly anchored decisions to systemic risk parity.
- **Manipulators Focus**: Primarily anchored to extremely short-term features like `r1h` and positional data since their primary environment incentive revolves around capturing immediate turnover yield.
- **Herd-following Retail Focus**: The SHAP interpretation proved the environment design successful: `z_target_mom24h` was consistently isolated as the universally dominant feature dictating their action-space.

---

## 6. Regulator Analysis: Adapting to Strategic Behavior

We ran backtests tracking the actions of all trader archetypes simultaneously and processed their logs to visualize multi-agent market conditions (located in `/results/plots/`).

### Regulatory Observations & Insights

1. **Crisis Frequency**: The wash traders / manipulators generated significantly more "crash events" (<-5% portfolio drops in short timeframes) due to rapid oscillations and disregard for downside tail-risk. Regulators must monitor turnover-velocity to identify artificial volatility vectors.
2. **Market Efficiency / Sharpe**: Rational traders sustained linear Sharpe development, whereas herd traders exacerbated drawdowns, showing highly correlated stress events.
3. **Actionable Policy**: If a market exhibits high retail herding, correlations spike during drawdowns. Regulators may need to force stablecoin collateral safety-limits higher strictly during periods where `mom24h` diverges greatly from long term `dd7d` features to prevent systemic liquidation cascades.
