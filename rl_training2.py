import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v1",
                ip_address="127.0.0.3",
                step_length=0.25,
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
    learning_rate=0.0001,
    n_steps=2048,
    batch_size=64,
    verbose=1, 
    tensorboard_log="./tb_logs/", 
    device="cuda"
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=100,
    best_model_save_path=".",
    log_path=".",
    eval_freq=100,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=2e4,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("ppo_airsim_drone_policy")
