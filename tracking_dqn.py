import setup_path
import gym
import airgym
import time
import logging
import os
import traceback

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
import torch as th
from scheduling import linear_schedule
# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airsim-car-tracking-v1",
                ip_address="127.0.0.1",
                step_length=1,
                image_shape=(5,),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
#env = VecTransposeImage(env)

# Initialize RL algorithm type and parametersll
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=linear_schedule(0.1),
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=1,
    learning_starts=1000,
    buffer_size=1000000,
    gradient_steps=10000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tracking_tb_logs/"
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=f"./tracking_training_best_model3",
    log_path=f"./tracking_training_eval_logs3",
    eval_freq=1000,
)
callbacks.append(eval_callback)

# Create a progress bar callback to estimate time left
progress_bar_callback = ProgressBarCallback()
callbacks.append(progress_bar_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
directory = "./UE5_PATH_TRAVERSAL_LIDAR_T1000_best_model/"
folder = ""
learned = 0
while (learned == 0):
    learned = 1
    try:
        model.learn(
            total_timesteps=1e5,
            tb_log_name="UE5_PATH_TRAVERSAL_LIDAR_T1000_" + str(time.time()),
            **kwargs
        )
    except Exception as e:
        #logging.error(traceback.format_exc(e))
        learned = 0
        for filename in os.scandir(directory):
            if folder == "":
                folder = str(filename.name)
            elif str(filename.name) > folder:
                folder = str(filename.name)
        if(folder == ""):
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=linear_schedule(0.1),
                verbose=1,
                batch_size=32,
                train_freq=4,
                target_update_interval=1,
                learning_starts=1000,
                buffer_size=1000000,
                gradient_steps=10000,
                max_grad_norm=10,
                exploration_fraction=0.1,
                exploration_final_eps=0.01,
                device="cuda",
                tensorboard_log="./tracking_tb_logs/"
            )
        else:
            model = DQN.load(directory + folder + "/best_model.zip")
        model.set_env(env)
# Save policy weights
model.save("tracking_training_final_save3")
