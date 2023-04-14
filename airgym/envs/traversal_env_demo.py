import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
import time
from PyQt5.QtWidgets import QApplication
import sys
from PyQt5.QtWidgets import QApplication
import sys
import pprint


class  DroneTraversalDemo(AirSimEnv):
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
            "processed_lidar": np.zeros(4),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(9)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )
        #self.app = QApplication(sys.argv)
        #sys.exit(self.app.exec_())
        #self.metricsGUI = MetricsGui()
        #self.app = QApplication(sys.argv)
        #sys.exit(self.app.exec_())
        #self.metricsGUI = MetricsGui()

    def __del__(self):
        # self.drone.reset()
        print("This is where drone would have been reset in traversal")

    def _setup_flight(self):
        self.negative_reward = 0
        self.threshold_start_time = time.time()
        # self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.takeoffAsync()

        # Make the the camera face where the drone is facing (since its stabilized now)
        camera_pose = self.drone.simGetVehiclePose()
        self.drone.simSetCameraPose(0, camera_pose)

        # Set home position
        #self.starting_position = airsim.Vector3r(0, 0, -19)
        #self.drone.moveToPositionAsync(self.starting_position.x_val, self.starting_position.y_val, self.starting_position.z_val, 10).join()
        #self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

        #Setting point of origin
        self.origin = self.drone.getMultirotorState().kinematics_estimated.position
        self.origin_dist_to_target = self.calc_dist(self.origin, self.get_destination())

    def get_destination(self):
        #last coors for city: (12.326184272766113, 119.89775848388672, -3.789776563644409)
        # last coors for mountain: (-359.7535095214844, -402.3492126464844, 15.1305513381958)
        # relatively close coors for UE5 City 
        return airsim.Vector3r(70.68778991699219, 198.01834106445312, -17.886749267578125)

    def getDist(self):
        return self.get_destination() - self.state["position"]
        
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
        self.drone_state = self.drone.getMultirotorState()


        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        self.state["prev_dist"] = self.state["curr_dist"]
        self.state["curr_dist"] = self.getDist()

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        #self.state["processed_lidar"] = self.process_lidar()

        self.lidar_processing()

        return [*self.state["prev_position"], *self.state["position"], *self.state["velocity"], *self.state["prev_dist"], *self.state["curr_dist"], *self.state["processed_lidar"]]

        #self.state["processed_lidar"] = self.process_lidar()

    # the actual movement of the drone
    def _do_action(self, action):
        quad_offset, rotate = self.interpret_action(action)
        if rotate == 0:
            quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
            self.drone.moveByVelocityAsync(
                quad_vel.x_val + quad_offset[0],
                quad_vel.y_val + quad_offset[1],
                quad_vel.z_val + quad_offset[2],
                0.5,
            ).join()
        else:
            self.drone.rotateByYawRateAsync(quad_offset, .5)

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
        origin_dist_to_target = self.calc_dist(target_l, self.origin)

        # if there has been a collision then huge penalty and reset
        if self.state["collision"]:
            reward = -100
            done = 1
            return reward, done

        # if the drone reaches the target location and didn't collide, huge reward to promote this behavior more often
        
        # debug 
        print("how far from target: ",curr_dist_to_target)
        if curr_dist_to_target <= 10:
            done = 1
            reward += 500

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
            reward += (prev_dist_to_target - curr_dist_to_target) * 10

        #Checks if the stopwatch has reached 1 minute. If it has, it checks if the negative reward
        #threshold has been reach, which would trigger the start of a new episode
        if((time.time() - self.threshold_start_time) >= 5):
            if(self.negative_reward <= -100):
                reward -= 20
                done = 1
        #else:
            #self.negative_reward = 0
            #self.threshold_start_time = time.time()

        # do we need? are we subtracting reward doubly for no reason?
        #else:
        #    if(reward < 0):
        #        self.negative_reward = self.negative_reward + reward
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()  

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

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
            quad_offset = -30
        elif action == 7:
            rotate = 1
            quad_offset = 30
        else:
            quad_offset = (0, 0, 0)

        return quad_offset, rotate
