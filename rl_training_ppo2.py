import setup_path
import gym
import airgym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v1",
                ip_address="127.0.0.2",
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
    learning_rate=0.0003,
    n_steps=2048,
    verbose=1,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=None,
    device="cuda",
    tensorboard_log="./logs_actual/",
    _init_setup_model=True
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
#eval_callback = EvalCallback(
#    env,
#    callback_on_new_best=None,
#    n_eval_episodes=100,
#    best_model_save_path=".",
#    log_path=".",
#    eval_freq=500,
#)
#callbacks.append(eval_callback)

#Add a checkpoint callback 
checkpoint_callback = CheckpointCallback(
    save_freq=500, 
    save_path='./checkpoint_logs/',
    name_prefix='ppo_rl_model',
    save_vecnormalize=True
)
callbacks.append(checkpoint_callback)

progress_bar_callback = ProgressBarCallback()
callbacks.append(progress_bar_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=2e3,
    tb_log_name="ppo_airsim_drone_run_landscape_across_lake" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("ppo_airsim_drone_policy_landscape_across_lake")
