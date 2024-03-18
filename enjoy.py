# See the Agent live

from env import get_render_env
from stable_baselines3 import DQN

import time

env = get_render_env()

model = DQN.load("pacman_10m_1e-4_32_005.zip", env, device="cuda")

obs = env.reset()

# let the Agent play (with 20 "fps")
for step in range(5000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, infos = env.step(action)
    env.render("human") 
    time.sleep(1/20)

    if done.any():
        obs = env.reset()

env.close()