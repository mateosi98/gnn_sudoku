import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool

import numpy as np
import random
from itertools import permutations

from matplotlib import pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# REINFROCE Network
class REINFORCE_graph(nn.Module):
    def __init__(self, state_space=None,
                       action_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None,
                       learning_rate=None):

        super(REINFORCE_graph, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"

        if hidden_dim is None:
            hidden_dim = state_space * 2
        

        # print(state_space)

        self.conv1 = GCNConv(10, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_space)
        self.layer_norm = LayerNorm(hidden_dim)

        self.roll_out = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def put_data(self, data):
        self.roll_out.append(data)

    def forward(self, x, edge_index, device):
        x = F.relu(self.conv1(x, edge_index))
        x = global_add_pool(self.layer_norm(x), torch.LongTensor([0 for _ in range(81)]).to(device))
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        out = F.log_softmax(x, dim=1)
        return out

    def train_net(self, gamma):
        R = 0
        G = []
        G_t = 0
        
        # Whitening baseline
        for r, prob in self.roll_out[::-1]:
            G_t = r + gamma * G_t
            G.append(G_t)
        G = np.array(G)
        G_mean = G.mean()
        G_std  = G.std()
        self.optimizer.zero_grad()

        for r, prob in self.roll_out[::-1]:
            R = r + gamma * R
            loss = -prob * ((R-G_mean) / G_std)
            loss.backward()
        self.optimizer.step()
        self.roll_out = []

