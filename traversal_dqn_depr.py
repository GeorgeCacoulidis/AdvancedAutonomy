import setup_path
import gym
import airgym
import time
import pytz
import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CheckpointCallback
from scheduling import linear_schedule


# Setup things for nice timezone formatting 
tz = pytz.timezone("US/Eastern")
now = datetime.datetime.now(tz)
formatted_date = now.strftime("%m_%d_%y_%H_%M")

# Change date to current
save_dir = f"./{formatted_date}_DQN_DRONE_TRAVERSAL"

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v1",
                ip_address="127.0.0.1",
                step_length=7,
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
    log_path=f"{save_dir}/eval_logs/",
    eval_freq=1000,
)
callbacks.append(eval_callback)

# Add a checkpoint callback to save the model and buffer every N steps
checkpoint_callback = CheckpointCallback(
    save_freq=500, 
    save_path=f"{save_dir}/checkpoint_save",
    save_vecnormalize=True
)
callbacks.append(checkpoint_callback)

# Create a progress bar callback to estimate time left
progress_bar_callback = ProgressBarCallback()
callbacks.append(progress_bar_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=1e6,
    tb_log_name=f"{save_dir}/tb_logs/",
    **kwargs
)

# Save policy weights

model.save(f"{save_dir}/final_save")