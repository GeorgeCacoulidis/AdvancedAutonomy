import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from scheduling import linear_schedule

# Reload the previous model: "best_model.zip path", "tb_logs path"
model = DQN.load("./DQN_ALPHA2_best_model/best_model.zip", tensorboard_log="./tb_logs/DQN_ALPHA2_1676611422.436817_1")

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v1",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(19,),
            )
        )
    ]
)

# Set the env 
model.set_env(env)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=f"./DQN_ALPHA2_best_model",
    log_path=f"./DQN_ALPHA2_eval_logs",
    eval_freq=5000,
)
callbacks.append(eval_callback)

# Create a progress bar callback to estimate time left
progress_bar_callback = ProgressBarCallback()
callbacks.append(progress_bar_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=1e5,
    reset_num_timesteps=False,
    #tb_log_name="./DQN_ALPHA2_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("DQN_ALPHA2")

