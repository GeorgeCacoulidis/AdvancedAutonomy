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


#model = DQN.load("F:\\AdvancedAutonomy\\UE5_PATH_TRAVERSAL_LIDAR_T1000_best_model\\1675152517.535831\\best_model.zip")
model = PPO.load("./PPO_Fixed_best_model/best_model.zip")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    if(_states["inSight"] == 0):
       break
    movement = env._interpret_action(action)
    env.sanityCheck(movement)    
    obs = env._get_obs()
    env.render()

