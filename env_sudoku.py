
import gym
from gym import Env
from gym.spaces import Discrete, Box, Tuple, Dict
import numpy as np

class SudokuEnv(Env):

    def __init__(self, max_time=1000, state_initial=[[0,0,0,0,0,0,0,8,1],
                                    [9,0,7,4,8,0,0,5,0],
                                    [0,0,0,3,0,0,9,0,7],
                                    [0,0,0,7,0,0,1,0,5],
                                    [0,3,0,6,0,9,0,7,0],
                                    [8,0,5,0,0,4,0,0,0],
                                    [1,0,6,0,0,5,0,0,0],
                                    [0,9,0,0,4,2,7,0,8],
                                    [7,8,0,0,0,0,0,0,0]]):
        # Moving in the grid and placing 0-9
        self.action_space = Discrete(14)
        # Observation space as mat of 0s to mat of 9s
        # low=[[0 for i in range(9)] for j in range(9)]
        # high=[[9 for i in range(9)] for j in range(9)]
        # self.observation_space = Tuple(Box(low=0, high=9, shape=(9,9), dtype=np.int),
        #                                 Discrete(81))
        self.observation_space = Dict({
            'grid': Box(low=0, high=9, shape=(9, 9), dtype=np.int),
            'cursor': Discrete(81)
        })
        # Initialize the grid as the initial grid
        self.state = {'grid':state_initial[:], 'cursor':0}
        # Constant initial state
        self.initial_state = self.state['grid'][:]
        # Time starts in zero
        self.time = 0
        self.max_time = max_time

    def num_used_in_row(self,grid,row,number):
        if number in grid[row]:
            return True
        return False

    def num_used_in_column(self,grid,col,number):
        for i in range(9):
            if grid[i][col] == number:
                return True
        return False

    def num_consecutive(self,grid,row,col,number):
        ban = False
        for i in [row-1]:
            if i >= 0:
                if grid[i][col] == number+ 1 or grid[i][col] == number-1:
                    ban = True
        for j in [col-1]:
            if j >= 0:
                if grid[row][j] == number+ 1 or grid[row][j] == number-1:
                    ban = True
        return ban

    def num_used_in_subgrid(self,grid,row,col,number):
        sub_row = (row // 3) * 3
        sub_col = (col // 3)  * 3
        for i in range(sub_row, (sub_row + 3)): 
            for j in range(sub_col, (sub_col + 3)): 
                if grid[i][j] == number: 
                    return True
        return False

    def move_cursor(self, row, col, direction):
        if direction == 0:
            row = max(row - 1, 0)
        elif direction == 1:
            row = min(row + 1, 8)
        elif direction == 2:
            col = max(col - 1, 0)
        elif direction == 3:
            col = min(col + 1, 8)
        return row * 9 + col

    def step(self, action): 
        # action is an int between 0 and 13
        cursor = int(self.state['cursor'])
        row, col = divmod(cursor, 9)
        x = self.state['grid']
        reward = -1
        possible = True
        # check if its a cursor move
        if action < 4:
            cursor = self.move_cursor(row, col, action)
        else:
            if x[row][col] != 0:
                possible = False
            if self.num_used_in_row(x,row,action):
                possible = False
            if self.num_used_in_column(x,col,action):
                possible = False
            if self.num_used_in_subgrid(x,row,col,action):
                possible = False
        # make the move if possible
        if possible:
            x[row][col] = action
        # else leave the state untuched
        else:
            pass
        #print(action)
        self.state['grid'] = x
        self.state['cursor'] = cursor
        self.time += 1 
        # Check if sudoku if full
        if np.all(np.not_equal(self.state['grid'],0)):
            reward = 10
            done = True
        elif self.time >= self.max_time:
            done = True
        else:
            done = False
        # Set placeholder for info
        info = {}
        # Return step information
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        # Reset the braid
        self.state['grid'] = self.initial_state[:]
        self.state['cursor'] = 0
        self.time = 0
        # Return the state after reset (initial state)
        return self.state

