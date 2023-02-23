import torch
import pandas

model = torch.hub.load('ultralytics/yolov5', 'custom', 'police_model_v2')

img = 'tests/sktSEhIa8q.jpg'

results = model(img, size = 1612)


ambulance_found = False
x_min = -1
x_max = -1
y_min = -1
y_max = -1
for box in results.xyxy[0]: 
    if box[5]==0:
        ambulance_found = True
        x_min = float(box[0])
        y_min = float(box[1])
        x_max = float(box[2])
        y_max = float(box[3])

print(ambulance_found, x_min, x_max, y_min, y_max)
results.print()
print(results.pandas().xyxy[0])