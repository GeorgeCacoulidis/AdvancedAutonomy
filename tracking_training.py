import setup_path
import gym
import airgym
import time

#from stable_baselines3 import DQN
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
import torch as th
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
#lenv = VecTransposeImage(env)
# Initialize RL algorithm type and parametersll
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=1,
    learning_starts=10000,
    buffer_size=1000000,
    gradient_steps=10000,
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
    best_model_save_path=f"./best_models/{str(time.time())}",
    log_path=f"./eval_logs/{str(time.time())}",
    eval_freq=10000,
)
callbacks.append(eval_callback)

# Create a progress bar callback to estimate time left
progress_bar_callback = ProgressBarCallback()
callbacks.append(progress_bar_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e6,
    tb_log_name="tracking_training" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("tracking_training")
