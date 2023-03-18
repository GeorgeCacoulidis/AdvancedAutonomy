import cv2
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
import time
import torch
import traceback

from pynput.keyboard import Key, Controller


#Bounding Box centering limit
BOX_LIM_X_MIN = 500
BOX_LIM_X_MAX = 750
BOX_LIM_Y_MIN = 250
BOX_LIM_Y_MAX = 450
keyboard = Controller()

class  DroneCarTrackingEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.negative_reward = 0
        self.threshold_start_time = time.time()
        self.detectionModel = self.load_model()
        self.boxSize = 0

        self.state = {
            "xMin": 0,
            "xMax": 0,
            "yMin": 0,
            "yMax": 0,
            "inSight": 1,
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(11)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def inSight(self):
        return self.state["inSight"]

    def _setup_flight(self):
        self.drone.reset()

        # keyboard reset used to be here 
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        #self.starting_position = airsim.Vector3r(-0.55265, -3.9786, -19.0225) # should this be declared in init? 
        #self.drone.moveToPositionAsync(self.starting_position.x_val, self.starting_position.y_val, self.starting_position.z_val, 10).join()
        #self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

        #Setting point of origin
        self.origin = self.drone.getMultirotorState().kinematics_estimated.position
        keyboard.press("l")
        keyboard.release("l")

    # pretty much just the current state of the drone the img, prev position, velocity, prev dist, curr dist, collision
    def _get_obs(self):
        #responses = self.drone.simGetImages([self.image_request])
        #image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.getModelResults()

        return [self.state["xMin"], self.state["xMax"], self.state["yMin"], self.state["yMax"], self.state["inSight"]]

    def getModelResults(self):
        image = self.raw_image_snapshot()
        ambulance_found, x_min, x_max, y_min, y_max = self.detection(image)

        self.state["inSight"] = int(ambulance_found)
        self.state["xMin"] = x_min
        self.state["xMax"] = x_max
        self.state["yMin"] = y_min
        self.state["yMax"] = y_max
    

    # the actual movement of the drone
    def _do_action(self, action):
        quad_offset, rotate = self.interpret_action(action)
        if rotate == 0:
            quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
            self.drone.moveByVelocityAsync(
                quad_vel.x_val + quad_offset[0],
                quad_vel.y_val + quad_offset[1],
                quad_vel.z_val + quad_offset[2],
                .5,
            ).join()
        else:
            self.drone.moveByRollPitchYawThrottleAsync(quad_offset[0], quad_offset[2], quad_offset[1], 1, .5)
        #print(self.state["position"]) # debug 

    def isCentered(self):
        if(self.state["xMin"] < BOX_LIM_X_MIN):
            return False
        elif(self.state["xMax"] > BOX_LIM_X_MAX):
            return False
        elif(self.state["yMin"] < BOX_LIM_Y_MIN):
            return False
        elif(self.state["yMax"] > BOX_LIM_Y_MAX):
            return False
        else:
            return True
    
    def calcBoxSize(self):
        x = self.state["xMax"] - self.state["xMin"]
        y = self.state["yMax"] - self.state["yMin"]
        return x * y

    def calcOffset(self):
        dist = 0
        if(self.state["xMin"] < BOX_LIM_X_MIN):
            dist = dist + BOX_LIM_X_MIN - self.state["xMin"]
        if(self.state["xMax"] > BOX_LIM_X_MAX):
            dist = dist + self.state["xMax"] - BOX_LIM_X_MAX
        if(self.state["yMin"] < BOX_LIM_Y_MIN):
            dist = dist + BOX_LIM_Y_MIN - self.state["yMin"]
        if(self.state["yMax"] > BOX_LIM_Y_MAX):
            dist = dist + self.state["yMax"] - BOX_LIM_Y_MAX
        
        return dist / 5



    def _compute_reward(self):
        reward = 0
        done = 0
        if(self.state["inSight"] == 0):
            self.reset()
            print("Testing: " + str(done))
            return -100, 1

        if(self.isCentered()):
            reward = reward + 20
        else:
            reward = reward - self.calcOffset()   

        box = self.calcBoxSize()
        if(box < self.boxSize):
            reward - 10
        if(box < 100):
            reward = reward - 100
            done = 1
        self.boxSize = box
                    
        return reward, done

    def step(self, action):
        self.getModelResults()
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'police_model_test_environment')
        print(model)
        return model

    def raw_image_snapshot(self):
        camera_name = "0"
        image_type = airsim.ImageType.Scene
        raw_image = self.drone.simGetImage(camera_name, image_type)

        return raw_image

    def detection(self, raw_image):
        png = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)

        result = self.detectionModel(png, size = 1216)
        ambulance_found = False
        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1
        for box in result.xyxy[0]: 
            if box[5]==0:
                ambulance_found = True
                x_min = float(box[0])
                y_min = float(box[1])
                x_max = float(box[2])
                y_max = float(box[3])        

        return ambulance_found, x_min, x_max, y_min, y_max


    # based on the action passed it does another action associated
    def interpret_action(self, action):
        rotate = 0
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        elif action == 6:
            rotate = 1
            quad_offset = (.52, 0, 0)
        elif action == 7:
            rotate = 1
            quad_offset = (0, .52, 0)
        elif action == 8:
            rotate = 1
            quad_offset = (-.52, 0, 0)
        elif action == 9:
            rotate = 1
            quad_offset = (0, -.52, 0)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset, rotate
