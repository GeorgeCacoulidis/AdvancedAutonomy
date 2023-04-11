import setup_path 
import airsim
import cv2
import sys
import math
import time
import numpy as np 
import pprint
import torch
import pandas
import argparse

class Position:
    def __init__(self, pos):
        self.x = pos.x_val
        self.y = pos.y_val
        self.z = pos.z_val

class OrbitNavigator:
    def __init__(self, client, radius = 2, speed = 2, iterations = 1, center = [1,0]):
        self.radius = radius
        self.speed = speed
        self.iterations = iterations
        self.z = None
        self.takeoff = False # whether we did a take off

        if self.iterations <= 0:
            self.iterations = 1

        if len(center) != 2:
            raise Exception("Expecting '[x,y]' for the center direction vector")
        
        # center is just a direction vector, so normalize it to compute the actual cx,cy locations.
        cx = float(center[0])
        cy = float(center[1])
        length = math.sqrt((cx*cx) + (cy*cy))
        cx /= length
        cy /= length
        cx *= self.radius
        cy *= self.radius

        self.client = client

        self.home = self.client.getMultirotorState().kinematics_estimated.position
        # check that our home position is stable
        start = time.time()
        count = 0
        while count < 100:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            if abs(pos.z_val - self.home.z_val) > 1:                                
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print("Drone position is drifting, we are waiting for it to settle down...")
                    start = time
            else:
                count += 1

        self.center = pos
        self.center.x_val += cx
        self.center.y_val += cy

    def start(self, model):
        object_detected = False
        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1
        conf = -1

        yaw = 0
        pitch = -1
        roll = 0

        print("arming the drone...")
        self.client.armDisarm(True)

        
        # AirSim uses NED coordinates so negative axis is up.
        start = self.client.getMultirotorState().kinematics_estimated.position
        landed = self.client.getMultirotorState().landed_state

        print("already flying so we will orbit at current altitude {}".format(start.z_val))
        z = start.z_val # use current altitude then

        print("climbing to position: {},{},{}".format(start.x_val, start.y_val, z))
        #self.client.moveToPositionAsync(start.x_val, start.y_val, z, self.speed).join()
        self.z = z
        
        print("ramping up to speed...")
        count = 0
        self.start_angle = None
        
        # ramp up time
        ramptime = self.radius / 10
        self.start_time = time.time()        

        while count < self.iterations:
        
            image = raw_image_snapshot(self.client)
            object_detected, x_min, x_max, y_min, y_max, conf = detection(image, model)
            if (object_detected == True and conf > .7):
                print(object_detected, x_min, x_max, y_min, y_max, conf)
                return object_detected, x_min, x_max, y_min, y_max
            else:
                print(object_detected, x_min, x_max, y_min, y_max, conf)

            # ramp up to full speed in smooth increments so we don't start too aggressively.
            now = time.time()
            speed = self.speed
            diff = now - self.start_time
            if diff < ramptime:
                speed = self.speed * diff / ramptime
            elif ramptime > 0:
                print("reached full speed...")
                ramptime = 0
                
            lookahead_angle = speed / self.radius            

            # compute current angle
            pos = self.client.getMultirotorState().kinematics_estimated.position
            dx = pos.x_val - self.center.x_val
            dy = pos.y_val - self.center.y_val
            actual_radius = math.sqrt((dx*dx) + (dy*dy))
            angle_to_center = math.atan2(dy, dx)

            camera_heading = (angle_to_center - math.pi) * 180 / math.pi 

            # compute lookahead
            lookahead_x = self.center.x_val + self.radius * math.cos(angle_to_center + lookahead_angle)
            lookahead_y = self.center.y_val + self.radius * math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self.track_orbits(angle_to_center * 180 / math.pi):
                count += 1
                print("completed {} orbits".format(count))
            
            self.camera_heading = camera_heading
            #self.client.moveByVelocityZAsync(vx, vy, z, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, 15))
            self.client.rotateByYawRateAsync(15, 1).join()
            yaw = yaw + .2618
            orientation = airsim.to_quaternion(pitch, roll, yaw)
            self.client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), orientation))

        
        self.client.moveToPositionAsync(start.x_val, start.y_val, z, 2).join()
        return object_detected, x_min, x_max, y_min, y_max

    def track_orbits(self, angle):
        # tracking # of completed orbits is surprisingly tricky to get right in order to handle random wobbles
        # about the starting point.  So we watch for complete 1/2 orbits to avoid that problem.
        if angle < 0:
            angle += 360

        if self.start_angle is None:
            self.start_angle = angle
            self.previous_angle = angle
            self.shifted = False
            self.previous_sign = None
            self.previous_diff = None            
            self.quarter = False
            return False

        # now we just have to watch for a smooth crossing from negative diff to positive diff
        if self.previous_angle is None:
            self.previous_angle = angle
            return False            

        # ignore the click over from 360 back to 0
        if self.previous_angle > 350 and angle < 10:
            return False

        diff = self.previous_angle - angle
        crossing = False
        self.previous_angle = angle

        diff = abs(angle - self.start_angle)
        if diff > 45:
            self.quarter = True

        if self.quarter and self.previous_diff is not None and diff != self.previous_diff:
            # watch direction this diff is moving if it switches from shrinking to growing
            # then we passed the starting point.
            direction = self.sign(self.previous_diff - diff)
            if self.previous_sign is None:
                self.previous_sign = direction
            elif self.previous_sign > 0 and direction < 0:
                if diff < 45:
                    self.quarter = False
                    crossing = True
            self.previous_sign = direction
        self.previous_diff = diff

        return crossing


    def sign(self, s):
        if s < 0: 
            return -1
        return 1

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'police_model_v3_5')
    
    return model

def raw_image_snapshot(client):
    camera_name = "0"
    image_type = airsim.ImageType.Scene
    raw_image = client.simGetImage(camera_name, image_type)

    return raw_image

def connect_to_client():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync()
    
    return client

def detection(raw_image, model):
    png = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)

    result = model(png, size=1216)
    ambulance_found = False
    x_min = -1
    x_max = -1
    y_min = -1
    y_max = -1
    conf = -1

    for box in result.xyxy[0]: 
        if box[5]==0:
           ambulance_found = True
           x_min = float(box[0])
           y_min = float(box[1])
           x_max = float(box[2])
           y_max = float(box[3])
           conf = float(box[4])

    return ambulance_found, x_min, x_max, y_min, y_max, conf

def orbit():
    # Overall, the Position and OrbitNavigator classes are both for the orbit part. The rest is for object detection, but you need both.
    # I also changed the detection function slightly so it retuns the confidence level as well.

    client = connect_to_client()
    model = load_model()
    
    # Change the camera position to look ~60 degrees down. I thought it provided a better view for the drone.
    camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-1, 0, 0))  #PRY in radians
    client.simSetCameraPose(0, camera_pose)

    # Just to move the drone into the air, only for testing
    #client.moveToPositionAsync(0, 0, -15, 2).join()

    # The two statements that you actually need. 
    nav = OrbitNavigator(client, 4, 3, 99, ['1', '0']) #(client, radius, speed, iterations, center (tbh idk what this does))
    object_detected, x_min, x_max, y_min, y_max = nav.start(model)

    return object_detected, x_min, x_max, y_min, y_max

orbit()