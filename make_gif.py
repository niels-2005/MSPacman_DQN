# create a Gif 
from env import get_render_env
from stable_baselines3 import DQN
import imageio
import numpy as np
import time

# get env
env = get_render_env()

# load model (needs)
model = DQN.load("pacman_10m_3e-4_32_005.zip", env, device="cuda")

images = []
obs = model.env.reset()

# let the agent play and save "images"
for _ in range(1000):
    img = model.env.render(mode="rgb_array")
    time.sleep(1/120)
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, done, _ = model.env.step(action)
    if done:
        obs = model.env.reset()

# Save Gif with unlimited Loop
imageio.mimsave("mspacman.gif", [np.array(img) for img in images if img is not None], loop=0)