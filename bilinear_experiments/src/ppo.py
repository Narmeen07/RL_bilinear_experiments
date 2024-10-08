import torch
import os
from src.plotter import EigenvectorPlotter
from src.heist import load_model
import imageio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.helpers import action_space, evaluate_model
from collections import Counter
import random
import torch
import numpy as np
import imageio
from procgen import ProcgenEnv
from src.vec_env import VecExtractDictObs, VecMonitor, VecNormalize
from src.bilinear_impala import BimpalaCNN  # Adjust the import path based on your actual module location
from tqdm import tqdm
#import cv2
from src.maze_env import (
    EnvState,
    get_wall_presence_tensor,
    get_object_presence_tensor,
    get_next_n_moves_graph,
    get_neighbouring_wall_positions,
    maze_grid_to_graph,
    get_mouse_pos,
    MAZE_ACTION_INDICES
)

class PPO:
    def __init__(self, model, device= "cpu"):
        self.model = model.to(device)
        self.device = device
    
    def batch_act(self, observations):
        with torch.no_grad():
            # Ensure the observations are on the correct device and in the correct dtype
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
            dist, value = self.model(observations)
            return dist.sample().cpu().numpy()
    
    def get_next_n_actions_labels(self, grid, n):
        graph = maze_grid_to_graph(grid)
        mouse_pos = get_mouse_pos(grid)
        next_moves = get_next_n_moves_graph(grid, graph, n, mouse_pos)
        return np.array([MAZE_ACTION_INDICES[move][0] for move in next_moves])
    
    def create_dataset_with_labels(self, env, num_episodes=100, max_steps_per_episode=1000, n_actions=5):
        dataset = []
        
        for episode in tqdm(range(num_episodes), desc="Generating dataset"):
            obs = env.reset()
            done = False
            step = 0
            
            while not done and step < max_steps_per_episode:
                action = self.batch_act(obs)
                next_obs, _, dones, _ = env.step(action)
                
                # Generate labels
                env_state = EnvState(env.env.callmethod("get_state")[0])
                grid = env_state.full_grid()
                
                labels = {
                    'cheese_presence': get_object_presence_tensor(grid, 'cheese').flatten(),
                    'wall_presence': get_wall_presence_tensor(grid).flatten(),
                    'next_n_actions': self.get_next_n_actions_labels(grid, n_actions),
                    'mouse_location': np.array(get_mouse_pos(grid)),
                    'neighbouring_walls': get_neighbouring_wall_positions(grid, get_mouse_pos(grid))[1]
                }
                
                dataset.append({
                    'observation': obs,
                    'next_observation': next_obs,
                    'action': action,
                    'labels': labels
                })
                
                obs = next_obs
                done = dones.any()
                step += 1
        
        return dataset
    
# Configuration
class Config:
    env_name = 'maze'
    num_envs = 1
    num_levels = 0  # Set to 0 for infinite levels or specify for deterministic levels
    start_level = 0
    distribution_mode = 'easy'
    gpu = 0
    model_file = "/mnt/ssd-1/mechinterp/narmeen/bilinear_experiments_official/bilinear_experiments/bilinear_models/bimpala_maze_simplified.pt"

# Create environment
def create_venv(config):
    venv = ProcgenEnv(num_envs=config.num_envs, env_name=config.env_name,
                      num_levels=config.num_levels, start_level=config.start_level,
                      distribution_mode=config.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)

# Run a single episode and collect frames
def rollout_episode(agent, env, return_info= False):
    obs = env.reset()
    frames = []
    done = False
    while not done:
        action = agent.batch_act(obs)
        obs, reward, dones, infos = env.step(action)
        # Render the current environment state as an RGB array
        frame = env.render(mode='rgb_array')
        frames.append(frame)  # Append the RGB frame directly obtained from the environment
        done = dones.any()
    if return_info:
        return frames, reward, infos[0]['episode']
    return frames, reward
def rollout_episode_obs(agent, env):
    obs = env.reset()
    observations = [obs]
    done = False
    while not done:
        action = agent.batch_act(obs)
        obs, _, dones, infos = env.step(action)
        # Attempt to clone or serialize the environment state
        observations.append(obs)
        done = dones.any()

    return observations
def random_rollout_episode(agent, env, max_random_steps = 200):
    obs = env.reset()
    done = False

 
    while not done:
        # Decide randomly the number of steps to take (at least 1)
        num_steps = random.randint(0, max_random_steps)

        for _ in range(num_steps):
            action = agent.batch_act(obs)
            obs, _, dones, infos = env.step(action)

            # Check if the game has ended after each step
            done = dones.any()
            if done:
                break


    return env

# Create a GIF from frames
def create_gif(frames, filename='episode.gif'):
    with imageio.get_writer(filename, mode='I', duration=0.033) as writer:
        for frame in frames:
            writer.append_data(frame)

