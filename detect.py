import setup_path 
import airsim
import cv2
import numpy as np 
import pprint
import torch
import pandas


# connect to the AirSim simulator
client = airsim.VehicleClient()
client.confirmConnection()

# set camera name and image type
camera_name = "0"
image_type = airsim.ImageType.Scene
model = torch.hub.load('ultralytics/yolov5', 'custom', 'test_env_v5s_v4.pt')


def detection(raw_image, model):
    png = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_UNCHANGED)

    result = model(png, size = 1216)
    ambulance_found = False
    xA = -1
    xB = -1
    yA = -1
    yB = -1
    conf = 0
    
    for box in result.xyxy[0]: 
        if box[5]==0:
            conf = float(box[4])
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])      
            cv2.rectangle(png, (xA, yA), (xB, yB), (255,0, 0), 2)
            cv2.putText(png, str(box[4].item()), (xA, yA-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            ambulance_found = True

    
    cv2.imshow("AirSim", png)
    cv2.waitKey(5)

    return ambulance_found, xA, xB, yA, yB, conf
while True:
    raw_image = client.simGetImage(camera_name, image_type)
    ambulance_found, xA, xB, yA, yB, conf = detection(raw_image, model)
    print(ambulance_found, xA, xB, yA, yB, conf)