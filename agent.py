import argparse
import sys
import os
import gym
import numpy as np
import torch
from torch.distributions import Categorical
import torch.optim as optim
from torch.optim import lr_scheduler
import models
import utils
import pickle
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
        self.log_buffer = []

    def train(self, epochs: int, trajectory_len: int, env_gen: utils.AsyncEnvGen, lr=1e-4,
              discount_gamma=0.99, scheduler_gamma=0.98, beta=1e-3):
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

        env_gen.start()

        for episode in range(epochs):
            log_probs = []
            values = []
            rewards = []
            state, self.env = env_gen.q.get()
            for step in range(trajectory_len):
                value, policy_dist = self.model.forward(state)
                value = value.detach().item()
                dist = policy_dist.detach().squeeze(0)
                action = torch.multinomial(dist, 1).item()
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = Categorical(probs=dist).entropy()
                new_state, reward, done, info = self.env.step(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                self.log_buffer.append(info.__repr__() + '\n')
                entropy_term += entropy
                state = new_state

                if done or step == trajectory_len - 1:
                    q_val, _ = self.model.forward(new_state)
                    q_val = q_val.detach().item()
                    self.all_rewards.append(np.sum(rewards))
                    self.all_lengths.append(step)
                    self.average_lengths.append(np.mean(self.all_lengths[-10:]))
                    if episode % 10 == 0:
                        sys.stdout.write(
                            "episode: {}, reward: {}, total length: {}, average length: {} \n"
                            .format(episode, np.sum(rewards), step, self.average_lengths[-1]))
                    if (episode % 1000 == 0) and (episode != 0):
                        scheduler.step()
                        sys.stdout.write('stepped scheduler, new lr: {:.5f}\n'.format(scheduler.get_last_lr()[0]))

                    if (episode % 10000 == 0) and (episode != 0):
                        self.save_and_log()
                    # step = trajectory_len + 1
                    break

            q_vals = torch.zeros(len(values)).to(device)
            for t in reversed(range(len(rewards))):
                q_val = rewards[t] + discount_gamma * q_val
                q_vals[t] = q_val

            values = torch.FloatTensor(values).to(device)
            log_probs = torch.stack(log_probs)

            advantage = q_vals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = (actor_loss + critic_loss + beta * entropy_term)

            optimizer.zero_grad()
            ac_loss.backward()
            optimizer.step()

        sys.stdout.write('-' * 10 + ' Finished training ' + '-' * 10 + '\n')
        self.save_and_log()

    def save_and_log(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)
        with open(self.log_path, 'a') as f:
            _ = f.writelines(self.log_buffer)
            self.log_buffer = []
        sys.stdout.write('Saved agent to {}\nLogged info to {}'.format(self.save_path, self.log_path))

    def act(self, state):
        self.model.eval()
        _, policy_dist = self.model.forward(state)
        dist = policy_dist.detach().squeeze(0)
        action = torch.multinomial(dist, 1).item()
        return action


def main(raw_args):
    parser = argparse.ArgumentParser(
        description='Creates or load an agent and then trains it')
    parser.add_argument(
        '-load', type=str, nargs='?', help='Weather or not to load an existing agent from the specified path.\n'
                                           'In case of loading all other arguments are ignored')
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
    parser.add_argument('-valid_actions', type=int, nargs='?',
                        help='Number of actions to use. defaults to all of the env\'s actions', default=0)
    parser.add_argument('-epochs', type=int, nargs='?', help='Num epochs (episodes) to train', default=3000)
    parser.add_argument('-trajectory_len', type=int, nargs='?', help='Maximal length of single trajectory', default=300)
    parser.add_argument('-lr', type=float, nargs='?', help='Learning rate', default=3e-4)
    parser.add_argument('-discount_gamma', type=float, nargs='?', help='Discount factor', default=0.99)
    parser.add_argument('-scheduler_gamma', type=float, nargs='?', help='Scheduling factor', default=0.999)
    parser.add_argument('-beta', type=float, nargs='?', help='Info loss factor', default=1e-3)

    args = parser.parse_args(raw_args)
    envs = [utils.EnvWrapper(args.env, utils.ObsType[args.obs_type], [i for i in range(args.valid_actions)])
            for _ in range(3)]
    env_gen = utils.AsyncEnvGen(envs)
    if args.load:
        with open(args.load, 'rb') as f:
            agent = pickle.load(f)
    else:
        model = getattr(models, args.model)(envs[0].obs_size, envs[0].num_actions)
        timestamp = datetime.now().strftime('%y%m%d%H%m')
        save_path = os.path.join(args.save_dir, '{0}_{1}_{2}.pkl'.format(args.model, args.env, timestamp))
        log_path = os.path.join(args.log_dir, '{0}_{1}_{2}.log'.format(args.model, args.env, timestamp))
        agent = A2CAgent(model, save_path, log_path)

    agent.train(args.epochs, args.trajectory_len, env_gen, args.lr,
                args.discount_gamma, args.scheduler_gamma, args.beta)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
