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

#Bounding Box centering limit
BOX_LIM_X_MIN = 156
BOX_LIM_X_MAX = 356
BOX_LIM_Y_MIN = 156
BOX_LIM_Y_MAX = 356

class  DroneCarTrackingEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.negative_reward = 0
        self.threshold_start_time = time.time()
        self.detectionModel = self.load_model()

        self.state = {
            "xMin": 0,
            "xMax": 0,
            "yMin": 0,
            "yMax": 0,
            "inSight": True,
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        #self.starting_position = airsim.Vector3r(-0.55265, -3.9786, -19.0225) # should this be declared in init? 
        #self.drone.moveToPositionAsync(self.starting_position.x_val, self.starting_position.y_val, self.starting_position.z_val, 10).join()
        #self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

        #Setting point of origin
        self.origin = self.drone.getMultirotorState().kinematics_estimated.position

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])


    # pretty much just the current state of the drone the img, prev position, velocity, prev dist, curr dist, collision
    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.getModelResults()

        return image

    def getModelResults(self):
        image = self.raw_image_snapshot()
        ambulance_found, x_min, x_max, y_min, y_max = self.detection(image)

        self.state["inSight"] = ambulance_found
        self.state["xMin"] = x_min
        self.state["xMax"] = x_max
        self.state["yMin"] = y_min
        self.state["yMax"] = y_max
    

    # the actual movement of the drone
    def _do_action(self, action):
        print("wirweioprhjuwoir")
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            1,
        ).join()
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
    
    def calcOffeset(self):
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
        if(self.state["inSight"] == False):
            self.reset()
            print("Testing: " + str(done))
            return -100, 1

        if(self.isCentered()):
            reward = reward + 20
        else:
            reward = reward - self.calcOffset()   
                    
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
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'best')    
        return model

    def raw_image_snapshot(self):
        camera_name = "0"
        image_type = airsim.ImageType.Scene
        raw_image = self.drone.simGetImage(camera_name, image_type)

        return raw_image

    def detection(self, raw_image):
        png = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)

        result = self.detectionModel(png)
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
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
