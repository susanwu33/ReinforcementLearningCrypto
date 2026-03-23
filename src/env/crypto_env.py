import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

def infer_obs_cols(df: pd.DataFrame):
    drop = {"datetime", "split", "regime"}
    obs_cols = [c for c in df.columns if c.startswith("z_") and pd.api.types.is_numeric_dtype(df[c])]
    return obs_cols

from typing import Optional

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
        clip_reward: Optional[float] = None,
        risk_lambda: float = 0.0,
        step_size: float = 0.5,
        random_start: bool = False,
        horizon: Optional[int] = None,
        seed: int = 42,
        trader_type: str = "rational"
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
        self.trader_type = trader_type

        self.random_start = random_start
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3) # 0=sell, 1=hold, 2=buy
        obs_dim = len(self.obs_cols) + (1 if self.add_position_to_obs else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.has_regime = "regime" in self.df.columns
        self.t = 0
        self.start_t = 0
        self.end_t = len(self.df) - 2   # t+1 in reward
        self.pos = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0.0

        max_start = len(self.df) - 2
        if self.horizon is not None:
            max_start = len(self.df) - 2 - self.horizon
        max_start = max(0, max_start)

        if self.random_start and max_start > 0:
            self.start_t = int(self.rng.integers(0, max_start + 1))
        else:
            self.start_t = 0

        self.t = self.start_t

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

        if action == 0:
            new_pos = max(0.0, self.pos - self.step_size)
        elif action == 2:
            new_pos = min(1.0, self.pos + self.step_size)
        else:
            new_pos = self.pos

        delta_pos = abs(new_pos - self.pos)

        r_next = float(self.df.loc[self.t + 1, self.r_col])
        vol    = float(self.df.loc[self.t + 1, self.vol_col])
        vol = max(vol, 1e-6)

        port_r = new_pos * r_next
        trade_cost = self.cost * delta_pos
        net_port_r = port_r - trade_cost

        current_lmbda = self.risk_lambda
        try:
            if self.df.loc[self.t, "z_BTC_r1h"] < 0 and self.df.loc[self.t, "z_ES_r1h"] < 0:
                current_lmbda *= 5.0
        except KeyError:
            pass

        risk_penalty = current_lmbda * (new_pos * r_next) ** 2
        safe_vol = max(vol, 1e-4)

        base_reward = (net_port_r - self.rf) / safe_vol

        # Apply trader type behaviors
        if self.trader_type == "rational":
            reward = base_reward - risk_penalty
        elif self.trader_type == "manipulator":
            # Reward turnover and absolute volume, less sensitive to risk
            reward = base_reward * 0.5 + delta_pos * 0.1
        elif self.trader_type == "retail":
            # Herd behavior: align position with 24h momentum
            mom_col = next((c for c in self.obs_cols if "mom24h" in c), None)
            mom = float(self.df.loc[self.t, mom_col]) if mom_col else 0.0
            
            target_pos = 1.0 if mom > 0 else 0.0
            herd_penalty = -abs(new_pos - target_pos)
            
            # Less focus on pure return, more on following the herd
            reward = base_reward * 0.1 + herd_penalty * 0.5
        else:
            reward = base_reward

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

        terminated = (self.t >= self.end_t) or (self.t >= len(self.df) - 2)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, truncated, info

def make_env(sym: str, split_df: pd.DataFrame, cost=0.0005, risk_lambda=0.0,
             random_start=False, horizon=None, seed=42, trader_type="rational"):
    obs_cols = infer_obs_cols(split_df)
    r_col = f"{sym}_r1h"
    vol_col = f"{sym}_vol24h"
    env = CryptoInvestEnv(
        df=split_df, obs_cols=obs_cols,
        r_col=r_col, vol_col=vol_col,
        cost=cost,
        risk_lambda=risk_lambda,
        clip_reward=None,
        random_start=random_start,
        horizon=horizon,
        seed=seed,
        trader_type=trader_type
    )
    return env, obs_cols
