import os
import time
import sys
dir_name = '/Users/mateosallesize/Google Drive/SRO/Windswept/Code/SB3'
sys.path.insert(0, dir_name)

import torch
import gym
import numpy as np
import random
from torch.distributions import Categorical


from importlib import reload
import graph_data
reload(graph_data)
from graph_data import *
import gnn_agent
reload(gnn_agent)
from gnn_agent import *
import env_sudoku
reload(env_sudoku)
from env_sudoku import *


if __name__ == "__main__":

    # Determine seeds
    model_name = "REINFORCE"
    # env_name = "CartPole-v1"
    seed = 10
    exp_num = 'SEED_'+str(seed)

    # Set gym environment
    env = SudokuEnv()
    s = env.reset()
    # np.array(s['grid'],dtype=np.int64).reshape(-1,81)
    # s_g = create_torch_graph_data(np.array(s['grid'],dtype=np.int64).reshape(-1,81), s['cursor'])
    # env.observation_space['grid']
    # env.observation_space['grid'].sample().reshape(-1,81)
    # env.action_space.n
    # env = gym.make(env_name)

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")

    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).name == "apple M1":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # set parameters
    learning_rate = 0.0005
    episodes = 1000
    discount_rate = 0.99
    print_interval = 10
    Policy = REINFORCE_graph(state_space=env.observation_space['grid'].shape[0]**2,
                             action_space=env.action_space.n,
                             num_hidden_layer=0,
                             hidden_dim=128,
                             learning_rate=learning_rate).to(device)

    score = 0
    score_list = []

    for epi in range(episodes):
        s = env.reset()
        done = False

        step = 0

        while not done:
            # if epi%print_interval == 0:
            #     env.render()

            # Get action
            # print(s)
            s_g = create_torch_graph_data(np.array(s['grid'],dtype=np.int64).reshape(-1,81), s['cursor'])            
            # print(step)
            # print(s_g.x)
            # print(s_g.edge_index)

            a_prob = Policy.forward(s_g.x.to(device), s_g.edge_index.to(device), device)
            a_distrib = Categorical(torch.exp(a_prob))
            a = a_distrib.sample()

            # Interaction with Environment
            s_prime, r, done, _ = env.step(a.item())

            Policy.put_data((r, a_prob[0][a]))
            s = s_prime
            score += r
            step += 1
        
        Policy.train_net(discount_rate)
        score_list.append(score)
        score = 0.0

        # Logging/
        if epi%print_interval==0 and epi!=0:
            print("# of episode :{}, avg score : {}".format(epi, sum(score_list[-print_interval:])/print_interval))
            
    env.close()

    # plt.plot(score_list)
    # plt.title('Reward')
    # plt.ylabel('reward')
    # plt.xlabel('episode')
    # plt.grid()
    # plt.show()