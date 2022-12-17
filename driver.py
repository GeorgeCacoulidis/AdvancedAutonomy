import os 
import detect
import load_model2
import subprocess
from threading import Thread

# Create new threads
t_OBJ = Thread(target=subprocess.run, args=(["python", "detect.py"],))
t_RL = Thread(target=subprocess.run, args=(["python", "load_model2.py"],))

print("threads created")

# Start new Threads
t_OBJ.start()
t_RL.start()

print("threads started")

# Wait until threads are completely executed
t_OBJ.join()
t_RL.join()

print("threads joined")