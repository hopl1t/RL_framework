import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 256

class SimpleActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=HIDDEN_SIZE, device=torch.device('cpu')):
        super(SimpleActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        self.device = device

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

