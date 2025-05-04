import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from config import Config

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor head 
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, obs):
        if len(obs.shape) == 1: 
            obs = obs.unsqueeze(0)
        
        features = self.shared_layers(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        
        if len(value.shape) > 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        
        return action_logits, value

class PPOTrainer:
    def __init__(self, model, lr=3e-4, clip_ratio=0.2, entropy_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def store(self, obs, action, reward, log_prob, value, done):
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, gamma=0.99, lambda_=0.95):
        returns = []
        advantages = []
        last_gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            last_gae = delta + gamma * lambda_ * (1 - self.dones[t]) * last_gae
            advantages.insert(0, last_gae)
            
            if self.dones[t]:
                returns.insert(0, self.rewards[t])
            else:
                returns.insert(0, self.rewards[t] + gamma * next_value)
        
        return returns, advantages
    
    def update(self, epochs=10):
        observations = torch.FloatTensor(np.array(self.observations)).to(Config.DEVICE)
        actions = torch.LongTensor(self.actions).to(Config.DEVICE)
        old_log_probs = torch.FloatTensor(self.log_probs).to(Config.DEVICE)
        
        returns, advantages = self.compute_returns_and_advantages()
        returns = torch.FloatTensor(returns).to(Config.DEVICE)
        advantages = torch.FloatTensor(advantages).to(Config.DEVICE)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            logits, values = self.model(observations)
            dist = Categorical(logits=logits)
            
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = ((values.squeeze() - returns) ** 2).mean()
            entropy_loss = -dist.entropy().mean()
            
            total_loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        self.reset()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }