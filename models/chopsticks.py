import torch
import torch.nn as nn
import torch.nn.functional as F

class ChopsticksMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChopsticksMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value