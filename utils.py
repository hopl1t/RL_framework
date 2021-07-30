import gym
import gym_sokoban
from enum import Enum


class ObsType(Enum):
    REGULAR = 1
    ROOM_STATE_VECTOR = 2
    ROOM_STATE_MATRIX = 3


class EnvWrapper():
    """
    Wrapps a Sokoban gym environment s.t. we can use the room_state property instead of regular state
    """

    def __init__(self, env_name, obs_type=ObsType.REGULAR, valid_actions=[], *args, **kwargs):
        """
        Wraps a gym environment s.t. you can control it's input and output
        :param env_name: str, The environments name
        :param obs_type: ObsType, type of output for environment's observations
        :param valid_inputs: list, optional. list of valid action number. If empty defaults to all actions
        :param args: Any args you want to pass to make()
        :param kwargs: Any kwargs you want to pass to make()
        """
        self.obs_type = obs_type
        self.env = gym.make(env_name)
        self.valid_actions = valid_actions
        if obs_type == ObsType.REGULAR:
            self.obs_size = self.env.observation_space.shape[0]
        elif obs_type == ObsType.ROOM_STATE_VECTOR:
            self.obs_size = self.env.room_state.shape[0] ** 2
        elif obs_type == ObsType.ROOM_STATE_MATRIX:
            self.obs_size = self.env.observation_space.shape[0]
        if valid_actions:
            self.num_actions = len(valid_actions)
        else:
            self.num_actions = self.env.action_space.n

    def reset(self):
        obs = self.env.reset()
        return self.process_obs(obs)

    def step(self, action):
        if self.valid_actions:
            # +1 is a temporary fix cause 0 is NOP and we only want actions 1,2,3,4 is Sokoban push only
            obs, reward, done, info = self.env.step(action + 1)
        else:
            obs, reward, done, info = self.env.step(action)
        obs = self.process_obs(obs)
        return obs, reward, done, info

    def process_obs(self, obs):
        if self.obs_type == ObsType.REGULAR:
            return obs
        elif self.obs_type == ObsType.ROOM_STATE_VECTOR:
            return self.env.room_state.flatten()
        elif self.obs_type == ObsType.ROOM_STATE_MATRIX:
            return self.env.room_state
