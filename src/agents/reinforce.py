import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import trange

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def select_action_pg(policy_net, obs, device):
    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    probs = policy_net(x)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def train_reinforce(env_train, env_val, 
                    episodes=1000, max_steps_per_episode=1000,
                    gamma=0.99, lr_pi=1e-3, lr_v=1e-3,
                    eval_every=50,
                    device="cuda" if torch.cuda.is_available() else "cpu"):
    
    obs_dim = env_train.observation_space.shape[0]
    n_actions = env_train.action_space.n

    policy_net = PolicyNetwork(obs_dim, n_actions).to(device)
    value_net = ValueNetwork(obs_dim).to(device)

    opt_pi = optim.Adam(policy_net.parameters(), lr=lr_pi)
    opt_v = optim.Adam(value_net.parameters(), lr=lr_v)

    recent_returns = deque(maxlen=20)
    losses_pi = deque(maxlen=200)
    losses_v = deque(maxlen=200)

    pbar = trange(episodes, desc=f"REINFORCE train ({device})")

    for ep in pbar:
        obs, _ = env_train.reset()
        done = False
        
        log_probs = []
        values = []
        rewards = []
        
        ep_return = 0.0
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            action, log_prob = select_action_pg(policy_net, obs, device)
            
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            value = value_net(x).squeeze(0)
            
            obs, reward, done, _, _ = env_train.step(action)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            ep_return += reward
            steps += 1
            
        recent_returns.append(ep_return)
        
        # Calculate Returns (Discounted Cumulative Rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Standardize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        
        # Advantage calculation
        advantages = returns - values.detach()
        
        # Compute losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Optimize networks
        opt_pi.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        opt_pi.step()
        
        opt_v.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        opt_v.step()
        
        losses_pi.append(policy_loss.item())
        losses_v.append(value_loss.item())
        
        if ep % 10 == 0:
            pbar.set_postfix({
                "avg_epR": round(np.mean(recent_returns), 3) if recent_returns else None,
                "loss_pi": round(np.mean(losses_pi), 4) if losses_pi else None,
                "loss_v": round(np.mean(losses_v), 4) if losses_v else None,
            })
            
        if ep % eval_every == 0 and ep > 0:
            val_score = eval_reinforce(env_val, policy_net, device)
            pbar.set_postfix({
                "avg_epR": round(np.mean(recent_returns), 3),
                "val_avgR": round(val_score, 4),
            })
            
    return policy_net

@torch.no_grad()
def eval_reinforce(env, policy_net, device):
    obs, _ = env.reset()
    done = False
    total_r = 0.0
    steps = 0
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        probs = policy_net(x)
        action = torch.argmax(probs, dim=1).item()
        obs, r, done, _, _ = env.step(action)
        total_r += r
        steps += 1
    return total_r / max(1, steps)

@torch.no_grad()
def run_policy_reinforce(env, policy_net, device="cuda" if torch.cuda.is_available() else "cpu"):
    obs, _ = env.reset()
    done = False
    infos = []
    import pandas as pd
    while not done:
        x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        probs = policy_net(x)
        action = torch.argmax(probs, dim=1).item()
        obs, r, done, _, info = env.step(action)
        infos.append(info)
    return pd.DataFrame(infos)
