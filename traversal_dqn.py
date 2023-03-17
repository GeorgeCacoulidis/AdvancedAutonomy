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
from scheduling import linear_schedule


# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v1",
                ip_address="127.0.0.1",
                step_length=1,
                image_shape=(19,),
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
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=1000000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/"
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=f"./Traversal_DQN_reloaded_best_model",
    log_path=f"./Traversal_DQN_reloaded_eval_logs",
    eval_freq=5000,
)
callbacks.append(eval_callback)

# Create a progress bar callback to estimate time left
progress_bar_callback = ProgressBarCallback()
callbacks.append(progress_bar_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
directory = "./Traversal_DQN/"
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
        logging.error(traceback.format_exc(e))
        learned = 0
        for filename in os.scandir(directory):
            if folder == "":
                folder = filename.name
            elif filename.name > folder:
                folder = filename
        model = DQN.load(directory + folder + "/best_model.zip")
        model.set_env(env)

# Save policy weights

model.save("DQN_ALPHA_reloaded")


model.save(f"{save_dir}/final_save")