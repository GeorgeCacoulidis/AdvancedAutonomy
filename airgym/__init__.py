from gym.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-drone-sample-v1", entry_point="airgym.envs:AirSimDroneEnvV1",
)

register(
    id="airsim-car-tracking-v1", entry_point="airgym.envs:DroneCarTrackingEnv",
)

register(
    id="airsim-drone-traversal-demo-v0", entry_point="airgym.envs:DroneTraversalDemo",

)

register(
    id="airsim-car-tracking-demo-v0", entry_point="airgym.envs:DroneCarTrackingDemo"
)