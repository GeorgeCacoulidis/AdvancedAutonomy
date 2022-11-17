import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
import time



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
        self.starting_position = airsim.Vector3r(-0.55265, -31.9786, -19.0225) # should this be declared in init? 
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
        return airsim.Vector3r(36, -75, -21)

    def get_dist(self, position):
        return self.get_destination() - position

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

        return image

    # the actual movement of the drone
    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()

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
