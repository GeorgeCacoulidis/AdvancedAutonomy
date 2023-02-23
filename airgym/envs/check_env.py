from stable_baselines3.common.env_checker import check_env
from drone_env import AirSimDroneEnvV1
from tracking_env import DroneCarTrackingEnv

check_env(AirSimDroneEnvV1())
check_env(DroneCarTrackingEnv())