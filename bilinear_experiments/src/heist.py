from src.bilinear_impala import BimpalaCNN
import torch
from procgen import ProcgenEnv
from procgen import ProcgenGym3Env
from src.vec_env import VecExtractDictObs
from src.vec_env import VecMonitor
from src.vec_env import VecNormalize
from gym3 import ToBaselinesVecEnv
from src.vec_env.procgen_wrappers import TransposeFrame
from src.vec_env.procgen_wrappers import ScaledFloatFrame
import random

def wrap_venv(venv) -> ToBaselinesVecEnv:
    "Wrap a vectorized env, making it compatible with the gym apis, transposing, scaling, etc."

    venv = ToBaselinesVecEnv(venv)  
    venv = VecExtractDictObs(venv, "rgb")

    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv  

def create_venv(
    num: int, start_level: int, num_levels: int, num_threads: int = 1, distribution_mode: str = "easy"
):
    """
    Create a wrapped venv. See https://github.com/openai/procgen#environment-options for params

    num=1 - The number of parallel environments to create in the vectorized env.

    num_levels=0 - The number of unique levels that can be generated. Set to 0 to use unlimited levels.

    start_level=0 - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
    """
    venv = ProcgenGym3Env(
        num=num,
        env_name="heist",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        num_threads=num_threads,
        render_mode="rgb_array",
    )
    venv = wrap_venv(venv)
    return venv
    

#load the model and the state dict
def load_model(model_path,kernel_size):
    # Initialize your model
    venv = ProcgenEnv(
        num_envs=1,
        env_name="heist",
        num_levels= 100000,
        start_level=0,
        distribution_mode="easy",
        num_threads=4,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
 

    model = BimpalaCNN(
        obs_space=venv.observation_space,
        num_outputs=venv.action_space.n,
        kernel_size = kernel_size
    )
    # Load the state dict
    state_dict = torch.load(model_path)
    # Load the state dict into your model
    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")
    return model

def create_dataset(num_samples= 100, num_levels= 0):
    dataset = []
    for i in range(num_samples):
        dataset.append(create_venv(num=1, start_level=random.randint(1000, 10000), num_levels=num_levels))
    return dataset
