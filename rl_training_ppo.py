import setup_path
import gym
import airgym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v1",
                ip_address="127.0.0.1",
                step_length=1,
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=0.0003,
    verbose=1,
    batch_size=32,
    normalize_advantage=True,
    max_grad_norm=.5,
    ent_coef=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=f"./UE5_PATH_TRAVERSAL_LIDAR_PPO_best_model/{str(time.time())}",
    log_path=f"./UE5_PATH_TRAVERSAL_LIDAR_PPO_eval_logs/{str(time.time())}",
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
    tb_log_name="UE5_PATH_TRAVERSAL_LIDAR_PPO_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("UE5_PATH_TRAVERSAL_LIDAR_PPO_POLICY_WEIGHTS")
