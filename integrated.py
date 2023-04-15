import torch
import setup_path
import gym
import airgym
import time
import math
import object_detection_orbit_demo

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import trackingTestEnv, traversalTestEnv

mode = 0
env = None

yoloModel = None

changedToCarTracking = False
carTrackingModel = None
carTrackingObs = None

def loadYoloModel():
    global yoloModel

    # Hover drone so it doesnt fall while yolo model is loading
    object_detection_orbit_demo.hover()

    # Load Yolo Model if it has not been loaded yet
    if not yoloModel:
        yoloModel = torch.hub.load('ultralytics/yolov5', 'custom', 'police_model_v4')

def changeToTraversal():
    global env

    env = DummyVecEnv(
        [
            lambda: Monitor(
                gym.make(
                    "airgym:airsim-drone-traversal-demo-v0",
                    ip_address="127.0.0.1",
                    step_length=1,
                    image_shape=(19,),
                )
            )
        ]
    )

def changeToTracking():
    global env
    env = DummyVecEnv(
        [
            lambda: Monitor(
                gym.make(
                    "airgym:airsim-car-tracking-demo-v0",
                    ip_address="127.0.0.1",
                    step_length=6,
                    image_shape=(11,),
                    model = yoloModel,
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
    in_sight, x_min, x_max, y_min, y_max = object_detection_orbit_demo.orbit(yoloModel)
    global mode 
    mode = 2

def carTracking():
    global mode
    global carTrackingModel
    global carTrackingObs
    global changedToCarTracking
    
    if changedToCarTracking == False:
        carTrackingModel = DQN.load("./car_tracking_mvp.zip")
        carTrackingObs = env.reset()
        changedToCarTracking = True

    while True:
        action, _states = carTrackingModel.predict(carTrackingObs)
        carTrackingObs, rewards, dones, info = env.step(action)
        env.render()
        print(info)
        # If confidence was not high, switch back to mode 1
        if(info[-1]["Conf"] < 0.5):
            mode = 1
            break

def main():
    global mode
    global changedToCarTracking

    while True:
        if mode == 0:
            # debug
            print("Main mode 0 entered")
            changeToTraversal()
            print("Called env change")
            print("calling traversal fncn")
            droneTraversal()
        if mode == 1:
            # debug
            loadYoloModel()
            print("Main mode 1 entered")
            detectionModel()
        if(mode == 2):
            # debug
            print("Main mode 2 entered")
            if changedToCarTracking == False:
                changeToTracking()
            carTracking()
                    
main()