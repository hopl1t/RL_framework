import argparse
import sys
import os
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import models
import utils
import pickle
import time
from datetime import datetime


class A2CAgent:
    """
    Holds a model, an environment and a training session
    Can train or perform actions
    """

    def __init__(self, model, save_path, log_path, **kwargs):
        self.model = model
        self.env = gym.Env
        self.kwargs = kwargs
        self.save_path = save_path
        self.log_path = log_path
        self.all_lengths = []
        self.average_lengths = []
        self.all_rewards = []
        self.all_times = []
        self.log_buffer = []

    def train(self, epochs: int, trajectory_len: int, env_gen: utils.AsyncEnvGen, lr=1e-4,
              discount_gamma=0.99, scheduler_gamma=0.98, beta=1e-3, print_interval=1000, log_interval=1000,
              save_interval=10000, scheduler_interval=1000):
        """
        Trains the model
        :param epochs: int, number of epochs to run
        :param trajectory_len: int, maximal length of a single trajectory
        :param env_gen: AsyncEnvGen, generates environments asynchronicaly
        :param lr: float, learning rate
        :param discount_gamma: float, discount factor
        :param scheduler_gamma: float, LR decay factor
        :param beta: float, information gain factor
        :return:
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            sys.stdout.write('Using CUDA\n')
        else:
            device = torch.device('cpu')
            sys.stdout.write('Using CPU\n')
        self.model.to(device)
        self.model.device = device
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
        entropy_term = torch.zeros(1).to(device)
        q_val = 0

        for episode in range(epochs):
            ep_start_time = time.time()
            log_probs = []
            values = []
            rewards = []
            state, self.env = env_gen.get_reset_env()
            for step in range(trajectory_len):
                value, policy_dist = self.model.forward(state)
                value = value.detach().item()
                dist = policy_dist.detach().squeeze(0)
                action, log_prob, entropy = self.env.process_action(dist, policy_dist.squeeze(0))
                new_state, reward, done, info = self.env.step(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                if log_interval:
                    self.log_buffer.append(info.__repr__() + '\n')

                entropy_term += entropy
                state = new_state

                if done or step == trajectory_len - 1:
                    q_val, _ = self.model.forward(new_state)
                    q_val = q_val.detach().item()
                    self.all_rewards.append(np.sum(rewards))
                    self.all_lengths.append(step)
                    if done:
                        self.all_times.append(time.time() - ep_start_time)
                        if (episode % print_interval == 0) and episode != 0:
                            self.print_stats(episode, print_interval)
                        if (episode % scheduler_interval == 0) and (episode != 0):
                            scheduler.step()
                            sys.stdout.write('stepped scheduler, new lr: {:.5f}\n'.format(scheduler.get_last_lr()[0]))
                        if (episode % save_interval == 0) and (episode != 0):
                            self.save()
                        if log_interval and (episode % log_interval == 0) and (episode != 0):
                            self.log()
                        break

            q_vals = torch.zeros(len(values)).to(device)
            for t in reversed(range(len(rewards))):
                q_val = rewards[t] + discount_gamma * q_val
                q_vals[t] = q_val

            values = torch.FloatTensor(values).to(device)
            log_probs = torch.stack(log_probs, dim=log_probs[0].dim()) # for more than one action dim will be 1

            advantage = q_vals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = (actor_loss + critic_loss + beta * entropy_term)
            optimizer.zero_grad()
            ac_loss.backward()
            optimizer.step()

        sys.stdout.write('-' * 10 + ' Finished training ' + '-' * 10 + '\n')
        utils.kill_process(env_gen)
        sys.stdout.write('Killed env gen process\n')
        self.save()
        if log_interval:
            self.log()

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)
        sys.stdout.write('Saved agent to {}\n'.format(self.save_path))

    def log(self):
        with open(self.log_path, 'a') as f:
            _ = f.writelines(self.log_buffer)
            self.log_buffer = []
        sys.stdout.write('Logged info to {}\n'.format(self.log_path))

    def print_stats(self, episode, print_interval):
        sys.stdout.write(
            "episode: {}, stats for last {} episodes:\tavg reward: {:.3f}\t"
            "avg length: {:.3f}\t avg time: {:.3f}\n"
                .format(episode, print_interval, np.mean(self.all_rewards[-print_interval:]),
                        np.mean(self.all_lengths[-print_interval:]),
                        np.mean(self.all_times[-print_interval:])))

    def act(self, state):
        self.model.eval()
        _, policy_dist = self.model.forward(state)
        dist = policy_dist.detach().squeeze(0)
        action = torch.multinomial(dist, 1).item()
        return action


def main(raw_args):
    parser = argparse.ArgumentParser(description='Creates or load an agent and then trains it')
    parser.add_argument(
        '-load', type=str, nargs='?', help='Weather or not to load an existing agent from the specified path.\n'
                                           'In case of loading all other arguments are ignored')
    parser.add_argument(
        '-async_env', action='store_true', help='Flag. If present use async environment generation, else don\'t',
        default=False)
    parser.add_argument(
        '-env', type=str, nargs='?', help='desired gym environment, example: "Pong-v0"')
    parser.add_argument(
        '-model', type=str, nargs='?', help='Model class name from models.py to use, example: "ModelClassName"')
    parser.add_argument(
        '-save_dir', type=str, nargs='?', help='Path to save and load the agent. Defaults to ./saved_agents',
        default='./saved_agents')
    parser.add_argument(
        '-log_dir', type=str, nargs='?', help='Path to save log files. Defaults to ./logs', default='./logs')
    parser.add_argument('-obs_type', type=str, nargs='?',
                        help='Type of observation to use - either REGULAR, ROOM_STATE_VECTOR or ROOM_STATE_MATRIX',
                        default='REGULAR')
    parser.add_argument('-action_type', type=str, nargs='?',
                        help='Type of action to use - wither REGULAR, PUSH_ONLY, PUSH_PULL', default='REGULAR')
    parser.add_argument('-epochs', type=int, nargs='?', help='Num epochs (episodes) to train', default=3000)
    parser.add_argument('-trajectory_len', type=int, nargs='?', help='Maximal length of single trajectory', default=300)
    parser.add_argument('-lr', type=float, nargs='?', help='Learning rate', default=3e-4)
    parser.add_argument('-discount_gamma', type=float, nargs='?', help='Discount factor', default=0.99)
    parser.add_argument('-scheduler_gamma', type=float, nargs='?', help='Scheduling factor', default=0.999)
    parser.add_argument('-scheduler_interval', type=int, nargs='?', help='Interval to step scheduler', default=1000)
    parser.add_argument('-beta', type=float, nargs='?', help='Info loss factor', default=1e-3)
    parser.add_argument('-print_interval', type=int, nargs='?', help='Print stats to screen evey x steps', default=1000)
    parser.add_argument('-log_interval', type=int, nargs='?', help='Log stats to file evey x steps. '
                                                                   'Set 0 for no logs at all', default=1000)
    parser.add_argument('-max_len', type=int, nargs='?', help='Maximal steps for a single episode', default=5000)
    parser.add_argument('-async_sleep_interval', type=float, nargs='?', help='How long should the env gen thread sleep',
                        default=1e-2)
    parser.add_argument('-num_envs', type=int, nargs='?', help='Number of async envs to use if using async_env.'
                                                               ' default 2', default=2)
    parser.add_argument('-num_discrete', type=int, nargs='?', help='How many discrete actions to generate for a cont.'
                                                                   ' setting using discrete action space', default=100)

    args = parser.parse_args(raw_args)
    assert os.path.isdir(args.save_dir)
    assert os.path.isdir(args.log_dir)
    envs = [utils.EnvWrapper(args.env, utils.ObsType[args.obs_type], utils.ActionType[args.action_type],
            args.max_len, num_discrete=args.num_discrete) for _ in range(args.num_envs)]
    env_gen = utils.AsyncEnvGen(envs, args.async_sleep_interval)
    if args.load:
        with open(args.load, 'rb') as f:
            agent = pickle.load(f)
    else:
        model = getattr(models, args.model)(envs[0].obs_size, envs[0].num_actions, num_discrete=args.num_discrete)
        timestamp = datetime.now().strftime('%y%m%d%H%m')
        save_path = os.path.join(args.save_dir, '{0}_{1}_{2}.pkl'.format(args.model, args.env, timestamp))
        log_path = os.path.join(args.log_dir, '{0}_{1}_{2}.log'.format(args.model, args.env, timestamp))
        agent = A2CAgent(model, save_path, log_path)

    try:
        if args.async_env:
            env_gen.start()
            sys.stdout.write('Started async env_gen process..\n')
        agent.train(args.epochs, args.trajectory_len, env_gen, args.lr,
                    args.discount_gamma, args.scheduler_gamma, args.beta,
                    args.print_interval, args.log_interval, scheduler_interval=args.scheduler_interval)
    finally:
        utils.kill_process(env_gen)
        if env_gen.is_alive():
            env_gen.terminate()
        sys.stdout.write('Killed env gen process\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
