import setup_path
import gym
import airgym
import time
import math
import object_detection_orbit

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import trackingTestEnv, traversalTestEnv

mode = 0

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

def calc_dist(dist):
    return math.sqrt(pow(dist.x_val, 2) + pow(dist.y_val, 2) + pow(dist.z_val, 2))

def droneTraversal():
    done = 0
    # Create a DummyVecEnv for main airsim gym env
    
    model = DQN.load("./DQN_ALPHA2_best_model/best_model.zip")

    # Debug
    print("Loaded DQN model")

    obs = env.reset()
    while done != 1:
        # Dbug 
        print("Entered while loop")

        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        print(info[-1]["curr_dist"])

        dist = info[-1]["curr_dist"]
        euclid_dist = calc_dist(dist)

        if(euclid_dist < 85):
            print("entered done if")
            global mode
            mode = 1
            done = 1
        env.render()
        

    

def detectionModel():
    in_sight, x_min, x_max, y_min, y_max = object_detection_orbit.orbit()
    global mode 
    mode = 2

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

def main():
    global mode
    while True:
        if mode == 0:
            # debug
            print("Main mode 0 entered")
            droneTraversal()
        if mode == 1:
            detectionModel()
        if(mode == 2):
            carTracking()
main()