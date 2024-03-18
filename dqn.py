from env import get_training_env
from stable_baselines3 import DQN

policy = "CnnPolicy"
env = get_training_env()
learning_rate = 0.0001
buffer_size = 500000
learning_starts = 250000
batch_size = 32
tau = 1.0
gamma = 0.99
train_freq = 4
gradient_steps = 1
optimize_memory_usage = False 
target_update_interval = 1000
exploration_fraction = 0.5
exploration_initial_eps = 1.0
exploration_final_eps = 0.1
max_grad_norm = 10
tensorboard_logs = "./logs/"
verbose = 0
seed = 42

# create model with hyperparameters
model = DQN(
    policy=policy,
    env=env,
    learning_rate=learning_rate,
    buffer_size=buffer_size,
    learning_starts=learning_starts,
    batch_size=batch_size,
    tau=tau,
    gamma=gamma,
    train_freq=train_freq,
    gradient_steps=gradient_steps,
    optimize_memory_usage=optimize_memory_usage,
    target_update_interval=target_update_interval,
    exploration_fraction=exploration_fraction,
    exploration_initial_eps=exploration_initial_eps,
    exploration_final_eps=exploration_final_eps,
    max_grad_norm=max_grad_norm,
    tensorboard_log=tensorboard_logs,
    verbose=verbose,
    seed=seed
)

def get_model() -> DQN:
    return model