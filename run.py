import torch
import numpy as np
import time
from ppo import ActorCriticNetwork
from environment import create_environment, get_observation_info
from config import Config

def load_model(checkpoint_path, input_dim, action_dim):
    model = ActorCriticNetwork(input_dim, action_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()
    return model

def main():
    env = create_environment(render_mode='human', max_steps=Config.MAX_STEPS)
    
    obs_shape = get_observation_info(env)
    input_dim = obs_shape[0] * obs_shape[1]
    action_dim = env.action_space.n
    model = load_model("./final_model.pt", input_dim, action_dim)
    model.to(Config.DEVICE)
    
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        
        while not done:
            obs_flat = obs.flatten()
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=Config.DEVICE)
            
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                action = torch.argmax(logits).item()  
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            env.render()
            time.sleep(0.05)  # slow down to real time
    
    env.close()

if __name__ == "__main__":
    main()