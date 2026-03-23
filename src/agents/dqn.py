import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tqdm import trange

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
def eval_dqn(env, qnet, device):
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

def train_dqn(env_train, env_val,
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
            if step % log_every == 0:
                pbar.set_postfix({
                    "phase": "warmup",
                    "buf": len(rb),
                    "eps": round(eps, 3),
                    "avg_epR": round(np.mean(recent_returns), 3) if recent_returns else None
                })
            continue

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

        if step % log_every == 0:
            pbar.set_postfix({
                "eps": round(eps, 3),
                "buf": len(rb),
                "loss": round(np.mean(losses), 4) if losses else None,
                "avg_epR": round(np.mean(recent_returns), 3) if recent_returns else None,
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

@torch.no_grad()
def run_policy_dqn(env, qnet, device="cuda" if torch.cuda.is_available() else "cpu"):
    obs, _ = env.reset()
    done = False
    infos = []
    import pandas as pd
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a = int(torch.argmax(qnet(x), dim=1).item())
        obs, r, done, _, info = env.step(a)
        infos.append(info)
    return pd.DataFrame(infos)
