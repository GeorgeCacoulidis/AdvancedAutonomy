import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import trackingTestEnv, traversalTestEnv

def droneTraversal():
    # Create a DummyVecEnv for main airsim gym env
    env = traversalTestEnv(
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

    model = DQN.load("./DQN_ALPHA2_best_model/best_model.zip")
    obs = env.reset()
    while env.getDist() > 10:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

def detectionModel():
    #TODO
    print("TEST")

def carTracking():
    env = traversalTestEnv(
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

    model = DQN.load("./tracking_training_best_model3.zip")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        if(env.inSight() == 0):
            break
        obs, rewards, dones, info = env.step(action)
        env.render()

mode = 0

while True:
    if mode == 0:
        droneTraversal()
    if mode == 1:
        detectionModel()
    if(mode == 3):
        carTracking()

