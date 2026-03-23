!pip -q install gymnasium torch pandas numpy
!pip -q install tqdm matplotlib


import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from collections import deque

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from tqdm import trange


from google.colab import drive
drive.mount('/content/drive')


DATA_DIR = Path("/content/drive/MyDrive/ReinforcementLearningCrypto/processed")


# quick check
print("torch:", torch.__version__)
print("gymnasium:", gym.__version__)
print("cuda available:", torch.cuda.is_available())


def load_split(sym: str):
    train = pd.read_csv(DATA_DIR / f"{sym}_train.csv", parse_dates=["datetime"])
    val   = pd.read_csv(DATA_DIR / f"{sym}_val.csv",   parse_dates=["datetime"])
    test  = pd.read_csv(DATA_DIR / f"{sym}_test.csv",  parse_dates=["datetime"])
    return train, val, test


def infer_obs_cols(df: pd.DataFrame):
    drop = {"datetime", "split", "regime"}
    # Only use scaled columns for observation
    obs_cols = [c for c in df.columns if c.startswith("z_") and pd.api.types.is_numeric_dtype(df[c])]
    return obs_cols


def sanity_check_log(log_df, name=""):
    col = "net_port_r" if "net_port_r" in log_df.columns else "port_r"
    r = log_df[col].to_numpy()

    pos = log_df["pos"].to_numpy()
    turnover = float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 1 else 0.0

    print(f"[{name}] {col} mean={r.mean():.6g} std={r.std():.6g} min={r.min():.6g} max={r.max():.6g}")
    print(f"[{name}] avg_pos={pos.mean():.3f} turnover={turnover:.3f}")
    print(f"[{name}] pos=1 {(pos==1.0).mean():.3f} | pos=0.5 {(pos==0.5).mean():.3f} | pos=0 {(pos==0.0).mean():.3f}")

    # If you log action in info (recommended), show action distribution
    if "action" in log_df.columns:
        # action mapping: 0=sell, 1=hold, 2=buy
        a = log_df["action"].to_numpy()
        print(f"[{name}] action sell {(a==0).mean():.3f} | hold {(a==1).mean():.3f} | buy {(a==2).mean():.3f}")

    # Optional: behavior by regime
    if "regime" in log_df.columns:
        reg_mean_pos = log_df.groupby("regime")["pos"].mean()
        print(f"[{name}] mean pos by regime:\n{reg_mean_pos}\n")


class CryptoInvestEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        obs_cols: list[str],
        r_col: str,
        vol_col: str,
        cost: float = 0.0005,
        rf_per_hour: float = 0.0,
        add_position_to_obs: bool = True,
        clip_reward: float | None = None,
        risk_lambda: float = 0.0,
        step_size: float = 0.5,
        random_start: bool = False,
        horizon: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.obs_cols = obs_cols
        self.r_col = r_col
        self.vol_col = vol_col
        self.cost = cost
        self.rf = rf_per_hour
        self.add_position_to_obs = add_position_to_obs
        self.clip_reward = clip_reward
        self.risk_lambda = risk_lambda
        self.step_size = step_size

        # NEW
        self.random_start = random_start
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)
        obs_dim = len(self.obs_cols) + (1 if self.add_position_to_obs else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.has_regime = "regime" in self.df.columns
        self.t = 0
        self.start_t = 0
        self.end_t = len(self.df) - 2   # because we use t+1 in reward
        self.pos = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0.0

        # choose start index
        max_start = len(self.df) - 2
        if self.horizon is not None:
            max_start = len(self.df) - 2 - self.horizon
        max_start = max(0, max_start)

        if self.random_start and max_start > 0:
            self.start_t = int(self.rng.integers(0, max_start + 1))
        else:
            self.start_t = 0

        self.t = self.start_t

        # set episode end
        if self.horizon is None:
            self.end_t = len(self.df) - 2
        else:
            self.end_t = self.start_t + self.horizon

        return self._get_obs(), {}

    def _get_obs(self):
        x = self.df.loc[self.t, self.obs_cols].to_numpy(dtype=np.float32)
        if self.add_position_to_obs:
            x = np.concatenate([x, np.array([self.pos], dtype=np.float32)])
        return x

    def step(self, action: int):
        pos_prev = self.pos

        # action: 0=sell, 1=hold, 2=buy
        if action == 0:
            new_pos = max(0.0, self.pos - self.step_size)
        elif action == 2:
            new_pos = min(1.0, self.pos + self.step_size)
        else:
            new_pos = self.pos

        delta_pos = abs(new_pos - self.pos)

        # next-step return (prevents leakage)
        r_next = float(self.df.loc[self.t + 1, self.r_col])
        vol    = float(self.df.loc[self.t + 1, self.vol_col])
        vol = max(vol, 1e-6)

        port_r = new_pos * r_next
        trade_cost = self.cost * delta_pos
        net_port_r = port_r - trade_cost

        # Macro risk penalty adjustment (BTC and ES down => risk-off)
        current_lmbda = self.risk_lambda
        try:
            if self.df.loc[self.t, "z_BTC_r1h"] < 0 and self.df.loc[self.t, "z_ES_r1h"] < 0:
                current_lmbda *= 5.0
        except KeyError:
            pass

        risk_penalty = current_lmbda * (new_pos * r_next) ** 2

        # Net Sharpe
        safe_vol = max(vol, 1e-4)
        reward = (net_port_r - self.rf) / safe_vol

        if self.clip_reward is not None:
            reward = float(np.clip(reward, -self.clip_reward, self.clip_reward))

        info = {
            "t": self.t,
            "pos": new_pos,
            "r_next": r_next,
            "vol": vol,
            "port_r": port_r,
            "net_port_r": net_port_r,
            "cost": trade_cost,
            "risk_penalty": risk_penalty,

            "pos_prev": pos_prev,
            "action": action,
            "delta_pos": delta_pos,
        }
        if self.has_regime:
            info["regime"] = self.df.loc[self.t, "regime"]

        self.pos = new_pos
        self.t += 1

        # NEW termination: end of window OR end of data
        terminated = (self.t >= self.end_t) or (self.t >= len(self.df) - 2)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, info


def make_env(sym: str, split_df: pd.DataFrame, cost=0.0005, risk_lambda=0.0,
             random_start=False, horizon=None, seed=42):
    obs_cols = infer_obs_cols(split_df)
    r_col = f"{sym}_r1h"
    vol_col = f"{sym}_vol24h"
    env = CryptoInvestEnv(
        split_df, obs_cols,
        r_col=r_col, vol_col=vol_col,
        cost=cost,
        risk_lambda=risk_lambda,
        clip_reward=None,
        random_start=random_start,
        horizon=horizon,
        seed=seed,
    )
    return env, obs_cols


def make_bins(train_df, cols, n_bins=4):
    bins = {}
    for c in cols:
        x = train_df[c].to_numpy()
        qs = np.quantile(x, np.linspace(0, 1, n_bins+1)[1:-1])  # internal cut points
        bins[c] = qs
    return bins


def discretize_row(row, cols, bins):
    out = []
    for c in cols:
        out.append(int(np.digitize(row[c], bins[c])))
    return tuple(out)


def q_learning_train(env: CryptoInvestEnv, train_df: pd.DataFrame, disc_cols: list[str],
                     n_bins=4, episodes=3, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, verbose=True):
    bins = make_bins(train_df, disc_cols, n_bins=n_bins)
    Q = {}  # dict: state -> np.array(|A|)

    def get_Q(s):
        if s not in Q:
            Q[s] = np.zeros(env.action_space.n, dtype=np.float32)
        return Q[s]

    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * (ep / max(1, episodes-1))
        obs, _ = env.reset()
        done = False

        ep_reward = 0.0
        ep_steps = 0

        while not done:
            t = env.t
            row = env.df.loc[t]
            # include position bucket
            state = discretize_row(row, disc_cols, bins) + (int(env.pos * 2),)  # pos 0/0.5/1 -> 0/1/2

            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = int(np.argmax(get_Q(state)))

            obs2, r, done, _, info = env.step(a)

            ep_reward += r
            ep_steps += 1

            if not done:
                t2 = env.t
                row2 = env.df.loc[t2]
                state2 = discretize_row(row2, disc_cols, bins) + (int(env.pos * 2),)
                td_target = r + gamma * np.max(get_Q(state2))
            else:
                td_target = r

            qsa = get_Q(state)[a]
            get_Q(state)[a] = qsa + alpha * (td_target - qsa)

        if verbose:
            avg_r = ep_reward / max(1, ep_steps)
            print(f"[TD] ep {ep+1}/{episodes} | eps={eps:.3f} | avg_reward={avg_r:.6g} | steps={ep_steps} | states={len(Q)}")

    return Q, bins


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)


@torch.no_grad()
def select_action(qnet, obs, eps, n_actions, device):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = qnet(x)
    return int(torch.argmax(q, dim=1).item())


@torch.no_grad()
def eval_dqn(env: CryptoInvestEnv, qnet, device):
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    steps = 0
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a = int(torch.argmax(qnet(x), dim=1).item())
        obs, r, done, _, _ = env.step(a)
        total_r += r
        steps += 1
    return total_r / max(1, steps)


def train_dqn(env_train: CryptoInvestEnv, env_val: CryptoInvestEnv,
              steps=200_000, warmup=5_000, batch_size=64,
              gamma=0.99, lr=1e-3,
              eps_start=1.0, eps_end=0.05, eps_decay_steps=150_000,
              target_update=2_000, buffer_size=200_000,
              log_every = 2000, eval_every=10000,
              device="cuda" if torch.cuda.is_available() else "cpu"):

    obs_dim = env_train.observation_space.shape[0]
    n_actions = env_train.action_space.n

    qnet = QNet(obs_dim, n_actions).to(device)
    tnet = QNet(obs_dim, n_actions).to(device)
    tnet.load_state_dict(qnet.state_dict())

    opt = optim.Adam(qnet.parameters(), lr=lr)
    rb = ReplayBuffer(buffer_size)

    obs, _ = env_train.reset()
    episode_steps = 0

    # Track episode return / length
    ep_return = 0.0
    ep_len = 0
    recent_returns = deque(maxlen=20)
    recent_lens = deque(maxlen=20)

    losses = deque(maxlen=200)

    pbar = trange(steps, desc=f"DQN train ({device})", mininterval=1.0)

    def eps_by_step(s):
        if s >= eps_decay_steps:
            return eps_end
        return eps_start + (eps_end - eps_start) * (s / eps_decay_steps)

    for step in pbar:
        eps = eps_by_step(step)
        a = select_action(qnet, obs, eps, n_actions, device)
        obs2, r, done, _, info = env_train.step(a)
        rb.push(obs, a, r, obs2, done)
        obs = obs2

        ep_return += r
        ep_len += 1

        if done:
            recent_returns.append(ep_return)
            recent_lens.append(ep_len)
            obs, _ = env_train.reset()
            ep_return = 0.0
            ep_len = 0

        if len(rb) < warmup:
            # show warmup status
            if step % log_every == 0:
                pbar.set_postfix({
                    "phase": "warmup",
                    "buf": len(rb),
                    "eps": round(eps, 3),
                    "avg_epR": round(np.mean(recent_returns), 3) if recent_returns else None
                })
            continue

        # sample batch
        s, a_b, r_b, s2, d_b = rb.sample(batch_size)
        s  = torch.tensor(s, dtype=torch.float32, device=device)
        a_b = torch.tensor(a_b, dtype=torch.int64, device=device)
        r_b = torch.tensor(r_b, dtype=torch.float32, device=device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=device)
        d_b = torch.tensor(d_b, dtype=torch.float32, device=device)

        q = qnet(s).gather(1, a_b.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q2 = tnet(s2).max(dim=1).values
            target = r_b + gamma * (1.0 - d_b) * q2

        loss = nn.functional.smooth_l1_loss(q, target)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))

        if step % target_update == 0:
            tnet.load_state_dict(qnet.state_dict())

        # ---- progress display ----
        if step % log_every == 0:
            pbar.set_postfix({
                "eps": round(eps, 3),
                "buf": len(rb),
                "loss": round(np.mean(losses), 4) if losses else None,
                "avg_epR": round(np.mean(recent_returns), 3) if recent_returns else None,
                "avg_len": int(np.mean(recent_lens)) if recent_lens else None,
            })

        if step % eval_every == 0 and step > 0:
            val_score = eval_dqn(env_val, qnet, device)
            pbar.set_postfix({
                "eps": round(eps, 3),
                "loss": round(np.mean(losses), 4) if losses else None,
                "avg_epR": round(np.mean(recent_returns), 3) if recent_returns else None,
                "val_avgR": round(val_score, 4),
            })

    return qnet


def summarize(log_df: pd.DataFrame):
    col = "net_port_r" if "net_port_r" in log_df.columns else "port_r"
    r = log_df[col].to_numpy()
    cum = float(np.exp(r.sum()) - 1.0)
    sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(24*365))
    turnover = float(np.mean(np.abs(np.diff(log_df["pos"].to_numpy()))))
    return {"cum_return": cum, "sharpe": sharpe, "turnover": turnover}

def summarize_by_regime(log_df: pd.DataFrame):
    if "regime" not in log_df.columns:
        return None
    out = {}
    for reg, g in log_df.groupby("regime"):
        out[reg] = summarize(g)
    return pd.DataFrame(out).T


def choose_disc_cols(obs_cols):
    keys = ["mom4h", "vol4h", "dd7d", "BTC_r1h", "ES_r1h", "GC_r1h"]
    disc = [c for c in obs_cols if any(k in c for k in keys)]  # works because z_BTC_r1h contains BTC_r1h
    return disc[:3]


def run_policy_tabular(env: CryptoInvestEnv, Q, bins, disc_cols):
    obs, _ = env.reset()
    done = False
    infos = []
    while not done:
        t = env.t
        row = env.df.loc[t]
        state = discretize_row(row, disc_cols, bins) + (int(env.pos * 2),)
        q = Q.get(state, np.zeros(env.action_space.n))
        a = int(np.argmax(q))
        obs, r, done, _, info = env.step(a)
        infos.append(info)
    return pd.DataFrame(infos)


@torch.no_grad()
def run_policy_dqn(env: CryptoInvestEnv, qnet, device="cuda" if torch.cuda.is_available() else "cpu"):
    obs, _ = env.reset()
    done = False
    infos = []
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a = int(torch.argmax(qnet(x), dim=1).item())
        obs, r, done, _, info = env.step(a)
        infos.append(info)
    return pd.DataFrame(infos)


def run_one_crypto(
    sym: str,
    td_episodes: int = 5,
    dqn_steps: int = 120_000,
    warmup: int = 5_000,
    cost: float = 0.0005,
    lmbda: float = 0.0,
    out_dir: Path | None = None,
    horizon_train=168,
    seed: int = 42,
):
    """
    Runs TD(Q-learning) baseline + DQN for a single crypto symbol (sym).
    Returns:
      - td_log_test, dqn_log_test
      - summary dicts
    Also optionally saves logs + summaries to out_dir.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ---- load data
    train_df, val_df, test_df = load_split(sym)

    # ---- build envs
    env_train, obs_cols = make_env(sym, train_df, cost=cost, risk_lambda=lmbda, random_start=True, horizon=horizon_train, seed=seed)
    env_val, _  = make_env(sym, val_df, cost=cost, risk_lambda=lmbda, random_start=False, horizon=None, seed=seed)
    env_test, _ = make_env(sym, test_df, cost=cost, risk_lambda=lmbda, random_start=False, horizon=None, seed=seed)

    # Set transaction cost consistently
    env_train.cost = cost
    env_val.cost = cost
    env_test.cost = cost

    # ---- TD baseline (tabular Q-learning)
    disc_cols = choose_disc_cols(obs_cols)
    Q, bins = q_learning_train(
        env_train, train_df,
        disc_cols=disc_cols,
        episodes=td_episodes,
        alpha=0.10,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05
    )
    td_test_log = run_policy_tabular(env_test, Q, bins, disc_cols)
    sanity_check_log(td_test_log, f"{sym} TD test")

    # ---- DQN
    qnet = train_dqn(
        env_train=env_train,
        env_val=env_val,
        steps=dqn_steps,
        warmup=warmup,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        target_update=2_000,
        buffer_size=200_000
    )
    dqn_test_log = run_policy_dqn(env_test, qnet)
    sanity_check_log(dqn_test_log, f"{sym} DQN test")

    # ---- Summaries
    td_overall = summarize(td_test_log)
    dqn_overall = summarize(dqn_test_log)

    td_by_regime = summarize_by_regime(td_test_log)
    dqn_by_regime = summarize_by_regime(dqn_test_log)

    results = {
        "sym": sym,
        "td_overall": td_overall,
        "dqn_overall": dqn_overall,
        "td_by_regime": td_by_regime,
        "dqn_by_regime": dqn_by_regime,
        "disc_cols": disc_cols,
    }

    # ---- Save artifacts (optional)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        td_test_log.to_csv(out_dir / f"{sym}_td_test_log.csv", index=False)
        dqn_test_log.to_csv(out_dir / f"{sym}_dqn_test_log.csv", index=False)

        # save summaries as json-friendly
        summary_json = {
            "sym": sym,
            "td_overall": td_overall,
            "dqn_overall": dqn_overall,
            "disc_cols": disc_cols,
        }
        with open(out_dir / f"{sym}_summary.json", "w") as f:
            json.dump(summary_json, f, indent=2)

        # save by-regime tables
        if td_by_regime is not None:
            td_by_regime.to_csv(out_dir / f"{sym}_td_by_regime.csv")
        if dqn_by_regime is not None:
            dqn_by_regime.to_csv(out_dir / f"{sym}_dqn_by_regime.csv")

    return results, td_test_log, dqn_test_log


def main(symbols=("AAVE", "ETC", "SUI"),
         out_dir=Path("/content/drive/MyDrive/ReinforcementLearningCrypto/results"),
         td_episodes=50,
         dqn_steps=150_000,
         warmup=5_000,
         cost=0.0005,
         lmbda=0.0):
    all_results = []
    for sym in symbols:
        print(f"\n=== Running {sym} ===")
        res, td_log, dqn_log = run_one_crypto(
            sym=sym,
            td_episodes=td_episodes,
            dqn_steps=dqn_steps,
            warmup=warmup,
            cost=cost,
            lmbda=lmbda,
            out_dir=out_dir,
        )
        all_results.append(res)

        print(f"{sym} TD overall:", res["td_overall"])
        print(f"{sym} DQN overall:", res["dqn_overall"])
        if res["dqn_by_regime"] is not None:
            print(f"{sym} DQN by regime:\n", res["dqn_by_regime"])

    # Build a compact summary table
    rows = []
    for r in all_results:
        rows.append({
            "sym": r["sym"],
            "td_cum": r["td_overall"]["cum_return"],
            "td_sharpe": r["td_overall"]["sharpe"],
            "td_turnover": r["td_overall"]["turnover"],
            "dqn_cum": r["dqn_overall"]["cum_return"],
            "dqn_sharpe": r["dqn_overall"]["sharpe"],
            "dqn_turnover": r["dqn_overall"]["turnover"],
        })
    summary_df = pd.DataFrame(rows)
    summary_path = Path(out_dir) / "all_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nSaved summary to:", summary_path)
    return summary_df, all_results


summary_df, all_results = main(symbols=("AAVE",), cost=0.001, lmbda=10)
summary_df


def run_fixed_pos(env, target_pos: float):
    """
    Baseline policy for incremental-action env:
    tries to keep position at target_pos (0, 0.5, or 1).
    """
    obs, _ = env.reset()
    done = False
    infos = []

    # because step_size=0.5, target_pos should be one of {0, 0.5, 1}
    target_pos = float(target_pos)

    while not done:
        # choose action to move toward target
        if env.pos < target_pos - 1e-9:
            action = 2  # buy
        elif env.pos > target_pos + 1e-9:
            action = 0  # sell
        else:
            action = 1  # hold

        obs, r, done, _, info = env.step(action)
        infos.append(info)

    return pd.DataFrame(infos)


coins = ["AAVE", "SUI", "ETC"]

results = {}

for coin in coins:

    train_df, val_df, test_df = load_split(coin)

    env_test, _ = make_env(
        coin,
        test_df,
        cost=0.001,
        risk_lambda=1.0
    )

    bh_log = run_fixed_pos(env_test, 1.0)   # buy & hold
    cash_log = run_fixed_pos(env_test, 0.0) # cash

    bh_summary = summarize(bh_log)
    cash_summary = summarize(cash_log)

    results[coin] = {
        "buy_hold": bh_summary,
        "cash": cash_summary
    }

    print(f"\n===== {coin} =====")
    print("buy_hold:", bh_summary)
    print("cash:", cash_summary)

summary_df, all_results = main(symbols=("ETC",), cost=0.001, lmbda=10)
summary_df

summary_df, all_results = main(symbols=("SUI",), cost=0.001, lmbda=10)
summary_df

# =========================
# Cross-asset correlation
# =========================

coins = ["AAVE", "SUI", "ETC"]

returns = {}

for coin in coins:
    _, _, test_df = load_split(coin)
    r_col = f"{coin}_r1h"
    returns[coin] = test_df[r_col].values

returns_df = pd.DataFrame(returns)

# Normal correlation
corr_matrix = returns_df.corr()

print("Normal Correlation")
print(corr_matrix)

# Stress correlation (bottom 10% returns)
threshold = returns_df.quantile(0.10)

stress_df = returns_df[
    (returns_df["AAVE"] < threshold["AAVE"]) |
    (returns_df["SUI"] < threshold["SUI"]) |
    (returns_df["ETC"] < threshold["ETC"])
]

stress_corr = stress_df.corr()

print("\nStress Correlation")
print(stress_corr)

cost = 0.001
summary_df, all_results = main(symbols=("AAVE", "ETC", "SUI"), cost=cost)
summary_df

