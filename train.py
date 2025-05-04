import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import math

from ppo import ActorCriticNetwork, PPOTrainer
from environment import create_environment, get_observation_info
from config import Config

def main():

    # all logs, checkpoints, and videos will be saved to this directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/roundabout_kinematics_{timestamp}"
    os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{experiment_dir}/videos", exist_ok=True)
    
    env = create_environment(max_steps=Config.MAX_STEPS)
    obs_shape = get_observation_info(env)
    input_dim = obs_shape[0] * obs_shape[1]
    action_dim = env.action_space.n
    
    model = ActorCriticNetwork(input_dim, action_dim).to(Config.DEVICE)
    trainer = PPOTrainer(model, lr=Config.LEARNING_RATE, clip_ratio=Config.CLIP_RATIO)
    
    episode_rewards = []
    timesteps = 0
    
    episode = 0
    while timesteps < Config.TIMESTEPS:
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done and timesteps < Config.TIMESTEPS:
            obs_flat = obs.flatten()
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=Config.DEVICE)
            
            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits.unsqueeze(0))
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            # TODO: make sure to take out the first vehicle here
            min_distance_to_ego = float('inf')
            for next_step_vehicle_obs in next_obs[1:]:
                if next_step_vehicle_obs[0] != 0:
                    distance_to_ego = math.sqrt(next_step_vehicle_obs[1] ** 2 + next_step_vehicle_obs[2] ** 2)
                    if distance_to_ego < min_distance_to_ego:
                        min_distance_to_ego = distance_to_ego
            if min_distance_to_ego != float('inf'):
                if min_distance_to_ego < 0.1:
                    if action.item() == 4:
                        reward += 10
                else:
                    reward += 10 * min_distance_to_ego
            done = terminated or truncated
            
            trainer.store(obs_flat, action.item(), reward, log_prob.item(), value.item(), done)
            
            obs = next_obs
            episode_reward += reward
            timesteps += 1
        
        episode_rewards.append(episode_reward)
        update_info = trainer.update(epochs=Config.EPOCHS_PER_UPDATE)
        
        if episode % Config.LOG_FREQ == 0:
            avg_reward = np.mean(episode_rewards[-Config.LOG_FREQ:])
            
            print(f"Episode {episode}")
            print(f"Avg Reward: {avg_reward:.2f}")
        
        if episode % Config.VIDEO_FREQ == 0 and episode > 0:
            video_dir = f"{experiment_dir}/videos/episode_{episode}"
            video_env = create_environment(
                render_mode='rgb_array',
                record_video=True,
                video_dir=video_dir
            )
            
            obs, _ = video_env.reset()
            done = False
            while not done:
                obs_flat = obs.flatten()
                obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=Config.DEVICE)
                with torch.no_grad():
                    logits, _ = model(obs_tensor)
                    action = torch.argmax(logits).item()
                obs, _, terminated, truncated, _ = video_env.step(action)
                done = terminated or truncated
            video_env.close()
        
        if episode % Config.CHECKPOINT_FREQ == 0 and episode > 0:
            checkpoint_path = f"{experiment_dir}/checkpoints/model_episode_{episode}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'Training Progress - Episode {episode}')
            plt.grid(True)
            plt.savefig(f"{experiment_dir}/checkpoints/rewards_episode_{episode}.png")
            plt.close()
        
        episode += 1
    
    torch.save(model.state_dict(), f"{experiment_dir}/final_model.pt")
    env.close()

if __name__ == "__main__":
    main()