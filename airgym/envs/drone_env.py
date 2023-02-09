import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
import time
from metric_gui import MetricsGui
from PyQt5.QtWidgets import QApplication
import sys
import pprint


class  AirSimDroneEnvV1(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.negative_reward = 0
        self.threshold_start_time = time.time()


        self.state = {
            "prev_position": np.zeros(3),
            "position": np.zeros(3),
            "collision": False,
            "prev_dist": np.zeros(3),
            "curr_dist": np.zeros(3),
            "processed_lidar": np.zeros(4),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )
        #self.app = QApplication(sys.argv)
        #sys.exit(self.app.exec_())
        #self.metricsGUI = MetricsGui()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.starting_position = airsim.Vector3r(0, 0, -19) # should this be declared in init? 
        self.drone.moveToPositionAsync(self.starting_position.x_val, self.starting_position.y_val, self.starting_position.z_val, 10).join()
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

        #Setting point of origin
        self.origin = self.drone.getMultirotorState().kinematics_estimated.position
        self.origin_dist_to_target = self.calc_dist(self.origin, self.get_destination())

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def get_destination(self):
        #last coors for city: (12.326184272766113, 119.89775848388672, -3.789776563644409)
        return airsim.Vector3r(-359.7535095214844, -402.3492126464844, 15.1305513381958)


    def get_dist(self, position):
        return self.get_destination() - position
    
    def detect_obstacle(self, box):
        point_count =  0
        for dist_point in box:
            if dist_point:
                point_count = point_count + 1
        if point_count >= box.size/3:
            return 1
        else:
            return 0

    def parse_lidarData(self, data):

        # reshape array of floats to array of [X,Y,Z]
        points = np.array(data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
       
        return points    
    #Input: 2D numpy array of Point Cloud LIDAR from AirSim API with a preset threshold
    #Funtion: Paritions the LIDAR into boxes then tests each box for threat
    #Output: numpy array of size 9, each element indicating the presence or absence of a threat in given box
    def process_lidar(self, lidar):
        x_step_size = lidar.shape[1]/3
        y_step_size = lidar.shape[0]/3
        obstacles = np.zeros(9)

        #Cutting the LIDAR into 9 boxes
        box00 = lidar[0:x_step_size, 0:y_step_size]
        box01 = lidar[0:x_step_size, y_step_size:y_step_size*2 + 1]
        box02 = lidar[0:x_step_size, y_step_size*2 + 1:y_step_size*3 + 1]
        box10 = lidar[x_step_size:x_step_size*2 + 1, 0:y_step_size]
        box11 = lidar[x_step_size:x_step_size*2 + 1, y_step_size:y_step_size*2 + 1]
        box12 = lidar[x_step_size:x_step_size*2 + 1, y_step_size*2 + 1:y_step_size*3 + 1]
        box20 = lidar[x_step_size*2 + 1:x_step_size*3 + 1, 0:y_step_size]
        box21 = lidar[x_step_size*2 + 1:x_step_size*3 + 1, y_step_size:y_step_size*2 + 1]
        box22 = lidar[x_step_size*2 + 1:x_step_size*3 + 1, y_step_size:y_step_size*3 + 1]

        #Check each individual LIDAR box for obstacle
        obstacles[0] = self.detect_obstacle(box00) 
        obstacles[1] = self.detect_obstacle(box01) 
        obstacles[2] = self.detect_obstacle(box02) 
        
        obstacles[3] = self.detect_obstacle(box10) 
        obstacles[4] = self.detect_obstacle(box11) 
        obstacles[5] = self.detect_obstacle(box12) 
        
        obstacles[6] = self.detect_obstacle(box20) 
        obstacles[7] = self.detect_obstacle(box21) 
        obstacles[8] = self.detect_obstacle(box22) 

        return obstacles
    
    def lidar_processing(self):
        lidar_results = np.zeros(4)
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
        self.state["processed_lidar"] = lidar_results

    # pretty much just the current state of the drone the img, prev position, velocity, prev dist, curr dist, collision
    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()


        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        self.state["prev_dist"] = self.state["curr_dist"]
        self.state["curr_dist"] = self.get_dist(self.state["position"])

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        #self.state["processed_lidar"] = self.process_lidar()

        return image

    # the actual movement of the drone
    def _do_action(self, action):
        self.lidar_processing()
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0] * 10,
            quad_vel.y_val + quad_offset[1] * 10,
            quad_vel.z_val + quad_offset[2] * 10,
            5,
        ).join()        
        self.drone.moveByVelocityAsync(0, 0, 0, .3).join()

    def calc_dist(self, pointA, pointB):
        return math.sqrt(pow(pointA.x_val - pointB.x_val, 2) + pow(pointA.y_val - pointB.y_val, 2) + pow(pointA.z_val - pointB.z_val, 2))

    #Punishes the drone for going farther than the original distance from the drone
    def radius_loss_eq(self, curr_dist_to_target):
        dist_change = curr_dist_to_target - self.origin_dist_to_target
        loss = (25) / (1 + pow(self.origin_dist_to_target, 2) * pow(math.e, (-0.5 * dist_change)))
        return loss


    def _compute_reward(self):
        reward = 0
        done = 0
        prev_l = self.state["prev_position"]
        curr_l = self.state["position"]
        target_l = self.get_destination()
        
        # Here we find the distance from the previous location to the target location
        # consider the target location x2 always
        prev_dist_to_target = self.calc_dist(target_l, prev_l)
        
        # Here we find the distance from the current location to the target location
        curr_dist_to_target = self.calc_dist(target_l, curr_l)

        # Calculate range between origin and target
        origin_dist_to_target = self.calc_dist(target_l, self.starting_position)

        # if there has been a collision then huge penalty and reset
        if self.state["collision"]:
            reward = -100
            done = 1
            return reward, done

        # if the drone reaches the target location and didn't collide, huge reward to promote this behavior more often
        if curr_dist_to_target == 0:
            done = 1
            reward += 100

        # if the drone does nothing and is in the same position give them minus 10
        # to show being stagnant is not the best move
        elif prev_dist_to_target == curr_dist_to_target:
            self.negative_reward -= 10
            reward -= 10

        # if the prev_dist is less then curr_dist, then we got further from the target
        # and give them a slight penalty to show they are going in the wrong direction
        elif prev_dist_to_target < curr_dist_to_target:
            if curr_dist_to_target > origin_dist_to_target:
                self.negative_reward -= self.radius_loss_eq(curr_dist_to_target)
                reward -= self.radius_loss_eq(curr_dist_to_target)
            else: 
                self.negative_reward -= 10
                reward -= 10

        
        # do we still need this?
        #Checks if the drone has drifted too far from the original distance and if we have a 
        #negative reward (implying that the drone is far and making no move to correct it)
        #elif(curr_dist_to_target > self.origin_dist_to_target and reward < 0):
        #    reward = reward - self.radius_loss_eq(curr_dist_to_target)
        
        # else the drone move closer to the target then its previous distance which is a +
        # previous dist is greater then curr distance so it'll pass a positive value
        else: 
            reward += (prev_dist_to_target - curr_dist_to_target)

        #Checks if the stopwatch has reached 1 minute. If it has, it checks if the negative reward
        #threshold has been reach, which would trigger the start of a new episode
        if((int) (time.time() - self.threshold_start_time) >= 60):
            if(self.negative_reward >= 100):
                done = 1
            else:
                self.negative_reward = 0
                self.threshold_start_time = time.time()

        # do we need? are we subtracting reward doubly for no reason?
        #else:
        #    if(reward < 0):
        #        self.negative_reward = self.negative_reward + reward

        ###print("Previous distance to target:", prev_dist_to_target)
        ##print("Current distance to target:", curr_dist_to_target)
        #print("Current position is:", self.state["position"])
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        #self.metricsGUI.refresh()
        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

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
