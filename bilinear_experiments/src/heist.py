from src.bilinear_impala_simplified import BimpalaCNN
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

def create_venv(num_envs=1, num_levels=0, start_level=0,env_name = "maze"):
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name,
                      num_levels=num_levels, start_level=start_level,
                      distribution_mode="easy")
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)
    

#load the model and the state dict
def load_model(model_path,kernel_size, env_name ="maze"):
    # Initialize your model
    venv = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels= 0,
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



