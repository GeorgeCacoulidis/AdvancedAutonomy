import setup_path
import gym
import airgym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CheckpointCallback
from scheduling import linear_schedule

save_dir="./tracking_ppo_depr_3_24"

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

# Initialize RL algorithm type and parameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=.001,
    n_steps=1024,
    verbose=1,
    batch_size=32,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=False,
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=.3,
    device="cuda",
    tensorboard_log=f"{save_dir}/tb_logs/",
    _init_setup_model=True
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=save_dir,
    log_path=save_dir,
    eval_freq=5000,
)
callbacks.append(eval_callback)

# Add a checkpoint callback to save the model and buffer every N steps
checkpoint_callback = CheckpointCallback(
    save_freq=1000, 
    save_path=f"{save_dir}/checkpoint_save",
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
    tb_log_name=f"{save_dir}",
    **kwargs
)

# Save policy weights
model.save(f"{save_dir}/final_save")
