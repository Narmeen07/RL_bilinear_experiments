import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import struct
import os
from procgen import ProcgenGym3Env
from gym3 import ToBaselinesVecEnv
from typing import Tuple, Dict, List, Optional

# Constants
KEY_COLORS = {0: 'blue', 1: 'green', 2: 'red'}
MOUSE = 0
KEY = 2
LOCKED_DOOR = 1
WORLD_DIM = 25
EMPTY = 100
BLOCKED = 51
GEM = 9

ENTITY_COLORS = {"blue": 0, "green": 1, "red": 2}
ENTITY_TYPES = {"key": 2, "lock": 1, "gem": 9, "player": 0}

class StateValue:
    def __init__(self, val, idx):
        self.val = val
        self.idx = idx

class EnvState:
    def __init__(self, state_bytes: bytes):
        self.state_bytes = state_bytes
        self.key_indices = {"blue": 0, "green": 1, "red": 2}

    @property
    def state_vals(self):
        return _parse_maze_state_bytes(self.state_bytes)

    @property
    def world_dim(self):
        return self.state_vals["world_dim"].val

    def full_grid(self, with_mouse=True):
        world_dim = self.world_dim
        grid = np.array([dd["i"].val for dd in self.state_vals["data"]]).reshape(world_dim, world_dim)
        if with_mouse:
            grid[self.mouse_pos] = MOUSE
        return grid

    def inner_grid(self, with_mouse=True):
        return inner_grid(self.full_grid(with_mouse=with_mouse))

    @property
    def mouse_pos(self) -> Tuple[int, int]:
        ents = self.state_vals["ents"][0]
        return int(ents["y"].val), int(ents["x"].val)

    def set_mouse_pos(self, x: int, y: int):
        state_vals = self.state_vals
        state_vals["ents"][0]["x"].val = float(y) + 0.5
        state_vals["ents"][0]["y"].val = float(x) + 0.5
        self.state_bytes = _serialize_maze_state(state_vals)

    def get_key_colors(self):
        key_colors = []
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2:
                key_index = ents["image_theme"].val
                key_color = KEY_COLORS.get(key_index, "Unknown")
                if key_color not in key_colors:
                    key_colors.append(key_color)
        return key_colors

    def add_entity(self, entity_type, entity_theme, x, y):
        state_values = self.state_vals
        new_entity = {
            "x": StateValue(float(y) + 0.5, 0),
            "y": StateValue(float(x) + 0.5, 0),
            "type": StateValue(entity_type, 0),
            "image_type": StateValue(entity_type, 0),
            "image_theme": StateValue(entity_theme, 0),
            # Add other necessary entity properties here
        }
        state_values["ents"].append(new_entity)
        state_values["ents.size"].val += 1
        self.state_bytes = _serialize_maze_state(state_values)

    def remove_all_entities(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            ents["x"].val = -1
            ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def remove_gem(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 9:
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_keys(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 2:
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

    def delete_locks(self):
        state_values = self.state_vals
        for ents in state_values["ents"]:
            if ents["image_type"].val == 1:
                ents["x"].val = -1
                ents["y"].val = -1
        self.state_bytes = _serialize_maze_state(state_values)

def inner_grid(grid: np.ndarray) -> np.ndarray:
    bl = 0
    while ((grid[bl, :] == BLOCKED).all() and (grid[-bl - 1, :] == BLOCKED).all() and
           (grid[:, bl] == BLOCKED).all() and (grid[:, -bl - 1] == BLOCKED).all()):
        bl += 1
    return grid[bl:-bl, bl:-bl] if bl > 0 else grid

def get_legal_mouse_positions(grid: np.ndarray, entities: List[Dict[str, StateValue]]):
    occupied_positions = set()
    for entity in entities:
        x, y = entity["x"].val, entity["y"].val
        ex, ey = int(x), int(y)
        occupied_positions.add((ex, ey))

    legal_positions = [
        (x, y)
        for x in range(grid.shape[0])
        for y in range(grid.shape[1])
        if grid[x, y] == EMPTY and (x, y) not in occupied_positions
    ]
    return legal_positions

def create_venv(num: int, start_level: int, num_levels: int, num_threads: int = 1, distribution_mode: str = "easy"):
    venv = ProcgenGym3Env(
        num=num,
        env_name="heist",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        num_threads=num_threads,
        render_mode="rgb_array",
    )
    venv = ToBaselinesVecEnv(venv)
    return venv

def state_from_venv(venv, idx: int = 0) -> EnvState:
    state_bytes_list = venv.env.callmethod("get_state")
    return EnvState(state_bytes_list[idx])

def _parse_maze_state_bytes(state_bytes: bytes) -> Dict[str, StateValue]:
    # Implementation of _parse_maze_state_bytes function
    # (This function is quite long, so I've omitted its content for brevity)
    pass

def _serialize_maze_state(state_vals: Dict[str, StateValue]) -> bytes:
    # Implementation of _serialize_maze_state function
    # (This function is quite long, so I've omitted its content for brevity)
    pass

def create_classified_dataset(num_samples_per_category=5, num_levels=0):
    dataset = {
        "gem": [], "blue_key": [], "green_key": [], "red_key": [],
        "blue_lock": [], "green_lock": [], "red_lock": []
    }
    key_indices = {"blue": 0, "green": 1, "red": 2}

    while any(len(samples) < num_samples_per_category for samples in dataset.values()):
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)
        key_colors = state.get_key_colors()

        if not key_colors:
            if len(dataset["gem"]) < num_samples_per_category:
                state_bytes = state.state_bytes
                if state_bytes is not None:
                    venv.env.callmethod("set_state", [state_bytes])
                    obs = venv.reset()
                    dataset["gem"].append(obs[0].transpose(1,2,0))
        else:
            for color in ["red", "green", "blue"]:
                if color in key_colors:
                    if len(dataset[f"{color}_key"]) < num_samples_per_category:
                        if color == "red":
                            state.delete_keys_and_locks(3)
                        elif color == "green":
                            state.delete_keys_and_locks(2)
                        state_bytes = state.state_bytes
                        if state_bytes is not None:
                            venv.env.callmethod("set_state", [state_bytes])
                            obs = venv.reset()
                            dataset[f"{color}_key"].append(obs[0].transpose(1,2,0))
                    if len(dataset[f"{color}_lock"]) < num_samples_per_category:
                        state.delete_keys()
                        if color in ["red", "green"]:
                            state.delete_specific_locks([key_indices[c] for c in key_indices if c != color])
                        state_bytes = state.state_bytes
                        if state_bytes is not None:
                            venv.env.callmethod("set_state", [state_bytes])
                            obs = venv.reset()
                            dataset[f"{color}_lock"].append(obs[0].transpose(1,2,0))
        venv.close()

    return dataset

def create_empty_maze_dataset(num_samples=5, num_levels=0, keep_player=True):
    dataset = {"empty_maze": []}

    while len(dataset["empty_maze"]) < num_samples:
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)

        state.remove_gem()
        state.delete_keys()
        state.delete_locks()
        if not keep_player:
            state.remove_player()
        
        state_bytes = state.state_bytes
        if state_bytes is not None:
            venv.env.callmethod("set_state", [state_bytes])
            obs = venv.reset()
            dataset["empty_maze"].append(obs[0].transpose(1,2,0))

        venv.close()

    return dataset

def set_mouse_to_center(state):
    full_grid = state.full_grid(with_mouse=False)
    entities = state.state_vals["ents"]
    legal_mouse_positions = get_legal_mouse_positions(full_grid, entities)

    if not legal_mouse_positions:
        return None

    middle_x = sum(pos[0] for pos in legal_mouse_positions) / len(legal_mouse_positions)
    middle_y = sum(pos[1] for pos in legal_mouse_positions) / len(legal_mouse_positions)

    closest_position = min(legal_mouse_positions, key=lambda pos: (pos[0] - middle_x) ** 2 + (pos[1] - middle_y) ** 2)

    state.set_mouse_pos(*closest_position)

    return state

def create_direction_dataset(num_samples_per_category=5, num_levels=100):
    dataset = {
        "top_left": [], "top_right": [],
        "bottom_left": [], "bottom_right": []
    }

    while any(len(samples) < num_samples_per_category for samples in dataset.values()):
        venv = create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels)
        state = state_from_venv(venv, 0)

        state.remove_all_entities()
        state = set_mouse_to_center(state)
        if state is None:
            venv.close()
            continue

        full_grid = state.full_grid(with_mouse=False)
        entities = state.state_vals["ents"]
        legal_positions = get_legal_mouse_positions(full_grid, entities)

        if len(legal_positions) < 4:
            venv.close()
            continue

        top_left_corner = min(legal_positions, key=lambda pos: pos[0] + pos[1])
        top_right_corner = max(legal_positions, key=lambda pos: pos[0] - pos[1])
        bottom_left_corner = max(legal_positions, key=lambda pos: -pos[0] + pos[1])
        bottom_right_corner = max(legal_positions, key=lambda pos: pos[0] + pos[1])

        for corner_pos, category in zip(
            [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner],
            ["top_left", "top_right", "bottom_left", "bottom_right"]
        ):
            if len(dataset[category]) < num_samples_per_category:
                state.set_gem_position(*corner_pos)
                state_bytes = state.state_bytes
                if state_bytes is not None:
                    venv.env.callmethod("set_state", [state_bytes])
                    obs = venv.reset()
                    dataset[category].append(obs[0].transpose(1, 2, 0))

        venv.close()

    return dataset