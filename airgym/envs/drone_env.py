import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class  AirSimDroneEnvV1(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

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
        self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def get_destination(self):
        return airsim.Vector3r(10, 10, 10)

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

     # in the near future make it so the drones z value and the grounds z value being 0 should always be in a set range
    # if within that range punish
    # its running in the same episode infinitly we need to stop an episode after a certain number of moves, it needs to lose like a video game
    def _compute_reward(self):
        reward = 0
        done = 0
        prev_l = self.state["prev_position"]
        curr_l = self.state["position"]
        target_l = self.get_destination()
        # Here we find the distance from the previous location to the target location
        # consider the target location x2 always
        prev_dist_to_target = math.sqrt(pow(target_l.x_val - prev_l.x_val, 2) + pow(target_l.y_val - prev_l.y_val, 2) + pow(target_l.z_val - prev_l.z_val, 2))
        # Here we find the distance from the current location to the target location
        curr_dist_to_target = math.sqrt(pow(target_l.x_val - curr_l.x_val, 2) + pow(target_l.y_val - curr_l.y_val, 2) + pow(target_l.z_val - curr_l.z_val, 2))

        collision_status = self.drone.simGetCollisionInfo().has_collided

        # if there has been a collision then huge penalty and reset
        if collision_status:
            self.reset()

        # if the drone reaches the target location and didn't collide, huge reward to promote this behavior more often
        if curr_dist_to_target == 0:
            done = 1
            return reward + 100, done

        # if the drone does nothing and is in the same position give them minus 10
        # to show being stagnant is not the best move
        if prev_dist_to_target == curr_dist_to_target:
            return reward-10, done

        # if the prev_dist is less then curr_dist, then we got further from the target
        # and give them a slight penalty to show they are going in the wrong direction
        if prev_dist_to_target < curr_dist_to_target:
            reward - (curr_dist_to_target - prev_dist_to_target)
            return reward, done

        # else the drone move closer to the target then its previous distance which is a +
        # previous dist is greater then curr distance so it'll pass a positive value
        reward + (prev_dist_to_target - curr_dist_to_target)
        return reward, done


    # def _compute_reward(self):
    #    dist_change = self.state["prev_dist"] - self.state["curr_dist"]
    #    net_dist = dist_change.x_val + dist_change.y_val + dist_change.z_val
    #    reward = net_dist * 2
    #    done = 0

    #    if self.state["collision"]:
    #        done = 1
    #        reward = reward - 100

        # Punishing drone for being idle
    #    if (net_dist < 5):
    #        done = 1
    #        reward = reward - 20
        
        # Cap on max height
    #    if (dist_change.y_val > 20):
    #        reward = reward - 20

    #    return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    # based on the action passed it does another action accossiated
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
