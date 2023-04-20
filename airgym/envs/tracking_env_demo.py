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
BOX_LIM_X_MIN = 300
BOX_LIM_X_MAX = 900
BOX_LIM_Y_MIN = 200
BOX_LIM_Y_MAX = 500
MIN_BOX_SIZE = 7000
BOX_STANDARDIZATION = 50000

class  DroneCarTrackingDemo(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, model):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.negative_reward = 0
        self.threshold_start_time = time.time()
        self.detectionModel = model
        self.obstacle = 0
        
        self.state = {
            "xMin": 0,
            "xMax": 0,
            "yMin": 0,
            "yMax": 0,
            "Conf": 1,
            "pxMin": 0,
            "pxMax": 0,
            "pyMin": 0,
            "pyMax": 0,
            "BoxSize": 0,
            "PrevBoxSize": 0,
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(5)
        self._setup_flight()

        #listOfSceneObjects = self.drone.simListSceneObjects()
        #for string in listOfSceneObjects:
        #        if string.startswith("StaticMeshActor_UAID_207BD21BE74E387201_1287001399"):
        #            self.drone.simDestroyObject(string)

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        # self.drone.reset()
        print("This is where drone would have been reset in tracking")

    def _setup_flight(self):
        #self.drone.reset()

        # keyboard reset used to be here 
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        #self.drone.takeoffAsync()
        # self.height = self.drone.getMultirotorState().gps_location.altitude
        # Angling -60 degrees downward
        
        #self.drone.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-0.7854, 0, 0)))

        # Set home position and velocity
        #self.starting_position = airsim.Vector3r(-0.55265, -3.9786, -19.0225) # should this be declared in init? 
        #self.drone.moveToPositionAsync(self.starting_position.x_val, self.starting_position.y_val, self.starting_position.z_val, 10).join()
        #self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

        #Setting point of origin
        #self.origin = self.drone.getMultirotorState().kinematics_estimated.position

    # pretty much just the current state of the drone the img, prev position, velocity, prev dist, curr dist, collision
    def _get_obs(self):
        #responses = self.drone.simGetImages([self.image_request])
        #image = self.transform_obs(responses)

        #self.drone_state = self.drone.getMultirotorState()


        self.getModelResults()

        return [self.state["xMin"]/1216, self.state["xMax"]/1216, self.state["yMin"]/684, self.state["yMax"]/684, self.state["Conf"],
                self.state["pxMin"]/1216, self.state["pxMax"]/1216, self.state["pyMin"]/684, self.state["pyMax"]/684, 
                self.state["BoxSize"]/BOX_STANDARDIZATION, self.state["PrevBoxSize"]/BOX_STANDARDIZATION]

    def getModelResults(self):
        image = self.raw_image_snapshot()
        conf, x_min, x_max, y_min, y_max = self.detection(image)

        self.state["pxMin"] = self.state["xMin"]
        self.state["pxMax"] = self.state["xMax"]
        self.state["pyMin"] = self.state["yMin"]
        self.state["pyMax"] = self.state["yMax"]
        self.state["Conf"] = conf
        self.state["xMin"] = x_min
        self.state["xMax"] = x_max
        self.state["yMin"] = y_min
        self.state["yMax"] = y_max
    

    # the actual movement of the drone
    def _do_action(self, action):
        quad_offset, rotate = self.interpret_action(action)
        if rotate == 0:
            quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
            self.drone.moveByVelocityBodyFrameAsync(
                quad_offset[0],
                quad_offset[1],
                quad_offset[2],
                1,
            )
        else:
            self.drone.rotateByYawRateAsync(quad_offset, 1)
        #print(self.state["position"]) # debug 

    def isCentered(self):
        xCenter = (self.state["xMin"] + self.state["xMax"]) / 2
        yCenter = (self.state["yMin"] + self.state["yMax"]) / 2
        print("(", xCenter, ", ", yCenter, ")")
        if(xCenter < BOX_LIM_X_MIN):
            return False
        elif(xCenter > BOX_LIM_X_MAX):
            return False
        elif(yCenter < BOX_LIM_Y_MIN):
            return False
        elif(yCenter > BOX_LIM_Y_MAX):
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
        
        return dist / 10

    def sanityCheck(self):

        lidar_results = np.zeros(4)
        flag = 0
        lidarData = self.drone.getLidarData(lidar_name="LidarSensor1", vehicle_name= "SimpleFlight")
        if(len(lidarData.point_cloud)):
            lidar_results[0] = 1
        else:
            lidar_results[0] = 0
        lidarData = self.drone.getLidarData(lidar_name="LidarSensor2", vehicle_name= "SimpleFlight")
        if(len(lidarData.point_cloud)):
            lidar_results[1] = 1
        else:
            lidar_results[1] = 0
        lidarData = self.drone.getLidarData(lidar_name="LidarSensor3", vehicle_name= "SimpleFlight")
        if(len(lidarData.point_cloud)):
            lidar_results[2] = 1
        else:
            lidar_results[2] = 0
        lidarData = self.drone.getLidarData(lidar_name="LidarSensor4", vehicle_name= "SimpleFlight")
        if(len(lidarData.point_cloud)):
            lidar_results[3] = 1
        else:
            lidar_results[3] = 0
        
        if(lidar_results[0] == 1):
            # quad_offset = (-self.step_length, 0, 0)
            flag = 1
            self.state["Conf"] = 0
        elif(lidar_results[1] == 1):
            quad_offset = (0, -self.step_length, 0)
            self.drone.moveByVelocityBodyFrameAsync(
                quad_offset[0],
                quad_offset[1],
                quad_offset[2],
                .5,
            ).join()
        elif(lidar_results[3] == 1):
            quad_offset = (0, self.step_length, 0)
            self.drone.moveByVelocityBodyFrameAsync(
                quad_offset[0],
                quad_offset[1],
                quad_offset[2],
                .5,
            ).join()

        return flag

    def _compute_reward(self):
        reward = 0
        done = 0
        print("Confidence: ", self.state["Conf"])
        if(self.state["Conf"] < .4):
            #self.reset()
            print("Testing: " + str(done))
            return -100, 0
        elif(self.state["Conf"] > .6):
            reward = reward + 25
        elif(self.state["Conf"] <= .6 and self.state["Conf"] >= .4):
            time.sleep(0.1)
            self.getModelResults()
            if(self.state["Conf"] < .6):
                #self.reset()
                print("Testing: " + str(done))
                return -100, 0
            
        if(self.isCentered()):
            reward = reward + 50
            print("******Centered!")
        else:
            #reward = reward - self.calcOffset()   
            reward = reward - 50
            print("Uncentered!")

        self.state["PrevBoxSize"] = self.state["BoxSize"]
        self.state["BoxSize"] = self.calcBoxSize()
        
        print("Box Size: ", self.state["BoxSize"])
        if(self.state["BoxSize"] < self.state["PrevBoxSize"]):
            reward = reward - 50
        else:
            reward = reward + 50
        if(self.state["BoxSize"] < MIN_BOX_SIZE):
            reward = reward - 100
            done = 0
        if(self.obstacle == 1):
            self.state["Conf"] = 0
                    
        return reward, done

    def step(self, action):
        self.getModelResults()
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        print("**********************")
        print("Obs: ", obs)
        print("Reward: ", reward)
        print("**********************")
        return obs, reward, 0, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'police_model_v3.5.pt')
        print(model)
        return model

    def raw_image_snapshot(self):
        camera_name = "0"
        image_type = airsim.ImageType.Scene
        raw_image = self.drone.simGetImage(camera_name, image_type)

        while not raw_image:
            print("Image from drone invalid. Retrying after 5ms")
            time.sleep(0.005)
            raw_image = self.drone.simGetImage(camera_name, image_type)

        return raw_image

    def detection(self, raw_image):
        png = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)

        result = self.detectionModel(png, size = 1216)
        conf = 0
        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1
        for box in result.xyxy[0]: 
            if box[5]==0:
                conf = float(box[4])
                x_min = float(box[0])
                y_min = float(box[1])
                x_max = float(box[2])
                y_max = float(box[3])        

        return conf, x_min, x_max, y_min, y_max


    # based on the action passed it does another action associated
    def interpret_action(self, action):
        rotate = 0
        quad_offset = (0, 0, 0)
        flag = self.sanityCheck()
        if flag == 0:
            if action == 0:
                # Go straight
                quad_offset = (self.step_length, 0, 0)
            elif action == 1:
                # Go right
                quad_offset = (0, self.step_length, 0)
            elif action == 2:
                # Go down
                quad_offset = (0, 0, self.step_length)
            elif action == 3:
                # Go left
                quad_offset = (0, -self.step_length, 0)
            

        print("Action: ", action, " - ", "quad_offset: ", quad_offset)
        return quad_offset, rotate

    # Removes the car in the environment and waits until its there, if not already
    def removeCar(self):
        carFound = False
        while not carFound:
            listOfSceneObjects = self.drone.simListSceneObjects()

            for string in listOfSceneObjects:
                if string.startswith("carActor_Lambo"):
                    carFound = True
                    self.drone.simDestroyObject(string)

            # If the car was not found, then we sleep to give it time to load in
            if not carFound:
                print("Car was not found. Sleeping for 5ms before next check")
                time.sleep(0.005)