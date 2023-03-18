from stable_baselines3.common.vec_env import DummyVecEnv

class traversalTestEnv(DummyVecEnv):
    def getDist():
        raise NotImplementedError()
    
class trackingTestEnv(DummyVecEnv):
    def inSight(): 
        raise NotImplementedError()