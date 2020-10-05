

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


from ..animation import Animation


class CellularAutomata2D:
    def __init__(self):

        # 2D Convolution with PyTorch
        self.conv = nn.Conv2d(1,1,3, stride=1,padding = 1,bias = False)


    def set_gameoflife_filter(self):
        neighbors_filter = torch.ones(3,3)
        neighbors_filter[1,1] = 0
        self.set_convolution_filter(neighbors_filter)


    def set_convolution_filter(self,f):
        self.conv.weight = torch.nn.Parameter(torch.Tensor(f[None,None,:,:]))


    def step(self,state):
        output = self.conv(state)
        state = self.apply_rule(output,state)
        return state

    def apply_rule(self,state):
        # To be overriden in parents classes
        return state


    def run(self,n_steps,init_state = None,n_init = 10,init_size = 100):

        # Prepare init state
        if init_state is None:
            init_state = self.make_random_init_state(init_size,n_init)

        # Run Cellular Automata
        states = []
        states.append(torch.squeeze(init_state).numpy())
        state = init_state

        for i in range(n_steps):
            state = self.step(state)
            states.append(torch.squeeze(state).numpy())

        return Animation(states)

    def make_random_init_state(self,size,n):
        state = torch.zeros(size,size)
        state[np.random.randint(0,size,size = (2,n))] = 1
        state = state[None,None,:,:]
        return state


    def show(self,states,cmap = "viridis",figsize = (8,8)):

        plt.figure(figsize = figsize)
        plt.imshow(states,cmap = cmap)
        plt.axis('off')
        plt.show()






class GameOfLife(CellularAutomata2D):

    def __init__(self):

        super().__init__()
        self.set_gameoflife_filter()

    def apply_rule(self,output,state):
    
        # Compute Alive and Dead rules
        rule_alive = ((output >= 2) * (output <= 3)).type(torch.IntTensor)
        rule_dead = (output == 3).type(torch.IntTensor)
        
        # Vectorized compution of the next state
        next_state = rule_alive * state + rule_dead * (1 - state)
        
        return next_state