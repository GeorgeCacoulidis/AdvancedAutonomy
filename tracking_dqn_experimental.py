import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CheckpointCallback
import torch as th
from scheduling import linear_schedule

save_dir = "./tracking_dqn_depr_4_4_23"

# Create a DummyVecEnv for car tracking airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airsim-car-tracking-v1",
                ip_address="127.0.0.1",
                step_length=7,
                image_shape=(11,),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
#env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=linear_schedule(0.1),
    verbose=1,
    batch_size=64,
    train_freq=2,
    target_update_interval=1000,
    learning_starts=10000,
    buffer_size=50000,
    max_grad_norm=10,
    exploration_fraction=0.2,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log=f"{save_dir}/tb_logs/"
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=f"{save_dir}/best_model",
    log_path=f"{save_dir}/eval_logs",
    eval_freq=1000,
)
callbacks.append(eval_callback)

# Add a checkpoint callback 
checkpoint_callback = CheckpointCallback(
    save_freq=1000, 
    save_path=f"{save_dir}/checkpoint_logs/"
)
callbacks.append(checkpoint_callback)

# Create a progress bar callback to estimate time left
progress_bar_callback = ProgressBarCallback()
callbacks.append(progress_bar_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=1e5,
    tb_log_name=f"{save_dir}" + str(time.time()),
    **kwargs
)

# Save policy weights when training is done
model.save(f"{save_dir}/final_save")