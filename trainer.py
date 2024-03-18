from dqn import get_model

total_timesteps = 10000000
tb_log_name = "pacman_10m"
progress_bar = True 

# get the model
model = get_model()

# model learns for 10m Steps
model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, progress_bar=progress_bar)

# save the trained model
model.save("pacman_10m.zip")