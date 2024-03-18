from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import gymnasium as gym


def get_training_env() -> gym.Env:
    env = make_atari_env("ALE/MsPacman-v5", n_envs=4, seed=42)
    env = VecFrameStack(env, n_stack=4)
    return env


def get_render_env() -> gym.Env:
    env = make_atari_env("ALE/MsPacman-v5", n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)
    return env
