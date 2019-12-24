import gym.envs.registration as registration
import gym.error

try:
    registration.register(
        id='statespace-v0',
        entry_point='samples.envs.statespace:StateSpaceEnv'
    )
except gym.error.Error as e:
    # this module may be initialized multiple times. ignore gym registration errors.
    pass
