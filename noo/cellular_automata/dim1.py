

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class CellularAutomata1D:
    def __init__(self,rule = None):

        # Store rule as attribute
        self.rule = rule 

        # Convolve to convert to binary representation with zero padding
        f = [4,2,1]
        self.conv = nn.Conv1d(1,1,3, stride=1,padding = 1,bias = False)
        self.conv.weight = torch.nn.Parameter(torch.Tensor([[f]]))


    def step(self,state,rule = None):

        # Allow for different rule at each step
        # Otherwise take attribute
        if rule is None:
            rule = self.rule
        assert rule is not None

        # Apply convolution to state
        output = self.conv(state[None,None,:])
        indexes = torch.squeeze(output.data.type(torch.LongTensor))

        # Dummify for matrix multiplication
        # Can also use .scatter_ https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4
        P = torch.zeros(len(state),8)
        P[torch.arange(0,len(indexes)),indexes] = 1

        # Compute next step values
        values = P @ rule
        return values


    def run(self,init_state,n_steps = 100,rule = None):

        states = [init_state]
        state = init_state

        for i in range(n_steps):

            state = self.step(state,rule)
            states.append(state)

        return states


    def show(self,states,title = "1D Cellular Automata",figsize = (8,8),cmap = "viridis"):

        if isinstance(states,list):
            states = torch.stack(states).numpy()

        plt.figure(figsize = figsize)
        plt.title(title)
        plt.imshow(states,cmap = cmap)
        plt.axis('off')
        plt.show()


    def _convert_rule_to_tensor(self,rule):

        if isinstance(rule,list):
            return rule
        elif isinstance(rule,int):
            rule = np.binary_repr(rule)
            rule = "0"*(8-len(rule))+rule
            rule = list(map(int,list(rule)))
            return rule
        else:
            return rule



    def run_random(self,rule = None,size = 100,n_steps = 100,init_state = None,p_init = 0.1,show = True,**kwargs):
        
        # Prepare init state
        # Random with a given probability if not precised
        if init_state is None:
            init_state = np.random.binomial(1,p = p_init,size = size)
            init_state = torch.Tensor(init_state)

            n = size

        # Prepare random rule
        if rule is None:
            rule = np.random.randint(0,2,size = 8,dtype = np.int8)
        else:
            rule = self._convert_rule_to_tensor(rule)

        # Convert to tensor and to binary repr for titles
        rule_str_binary = "".join(map(str,rule))
        rule_str_int = int(rule_str_binary,2)
        rule = torch.Tensor(rule)

        # Run Cellular Automata
        states = self.run(init_state,n_steps,rule)

        if show:
            self.show(states,title = f"Rule {rule_str_int}: {rule_str_binary}",**kwargs)
        else:
            return states






