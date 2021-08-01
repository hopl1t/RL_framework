import gym
import gym_sokoban
from enum import Enum
# import threading
# import queue
import time
import multiprocessing as mp

class ObsType(Enum):
    REGULAR = 1
    ROOM_STATE_VECTOR = 2
    ROOM_STATE_MATRIX = 3


class ActionType(Enum):
    REGULAR = 1
    PUSH_ONLY = 2
    PUSH_PULL = 3


class EnvWrapper():
    """
    Wrapps a Sokoban gym environment s.t. we can use the room_state property instead of regular state
    """

    def __init__(self, env_name, obs_type=ObsType.REGULAR, action_type=ActionType.REGULAR, *args, **kwargs):
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
        self.action_type = action_type
        if obs_type == ObsType.REGULAR:
            self.obs_size = self.env.observation_space.shape[0]
        elif obs_type == ObsType.ROOM_STATE_VECTOR:
            self.obs_size = self.env.room_state.shape[0] ** 2
        elif obs_type == ObsType.ROOM_STATE_MATRIX:
            self.obs_size = self.env.observation_space.shape[0]
        if action_type == ActionType.REGULAR:
            self.num_actions = self.env.action_space.n
        elif action_type == ActionType.PUSH_ONLY:
            self.num_actions = 4
        elif action_type == ActionType.PUSH_PULL:
            self.num_actions = 8

    def reset(self):
        obs = self.env.reset()
        return self.process_obs(obs)

    def step(self, action):
        if self.action_type == ActionType.REGULAR:
            pass
        elif self.action_type == ActionType.PUSH_ONLY:
            # maps from 0-3 to 1-4 since 0 is NOP
            action += 1
        elif self.action_type == ActionType.PUSH_PULL:
            # maps from 0-7 to [1,2,3,4,9,10,11,12]
            action += 1
            if action >= 5:
                action += 4
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


# class AsyncEnvGen(threading.Thread):
class AsyncEnvGen(mp.Process):
    """
    Creates and manages gym environments a-synchroneuosly
    This is used to save time on env.reset() command while playing a game
    """
    def __init__(self, envs, sleep_interval):
        super(AsyncEnvGen, self).__init__()
        self.envs = envs
        self.q = mp.Queue(len(self.envs) - 1)
        self._kill = mp.Event()
        self.env_idx = 0
        self.sleep_interval = sleep_interval

    def run(self):
        while not self._kill.is_set():
            if not self.q.full():
                state = self.envs[self.env_idx].reset()
                self.q.put((state, self.envs[self.env_idx]))
                self.env_idx += 1
                if self.env_idx == len(self.envs):
                    self.env_idx = 0
            elif self.sleep_interval != 0:
                time.sleep(self.sleep_interval)
        self.q.close()
        self.q.cancel_join_thread()

    def kill(self):
        self._kill.set()
