import gymnasium as gym
import highway_env
from gymnasium.wrappers import TimeLimit, RecordVideo
import os
import numpy as np

def create_environment(render_mode='rgb_array', max_steps=500, record_video=False, video_dir=None):
    config = {
        "action": {
            "type": "DiscreteMetaAction"
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": [
                "presence",
                "x",
                "y",
                "vx",
                "vy",
                "heading",
                "cos_h",
                "sin_h",
                "cos_d",
                "sin_d"
            ],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "normalize": True,
            "absolute": False,
            "see_behind": True
        },
        # "collision_reward": -1,
        # "high_speed_reward": 0.2,
        # "right_lane_reward": 0,
        # "lane_change_reward": -0.05,
        "normalize_reward": False,
        "duration": 20,
        "policy_frequency": 10,
    }
    env = gym.make('roundabout-v0', render_mode=render_mode, config=config)
    
    env = TimeLimit(env, max_episode_steps=max_steps)
    
    if record_video and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix="roundabout_kinematics"
        )
    
    return env

def get_observation_info(env):
    obs_sample, _ = env.reset()
    return obs_sample.shape