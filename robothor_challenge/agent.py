from abc import ABC, abstractmethod
import random

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class Agent(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, observations):
        pass


class SimpleRandomAgent(Agent):

    def reset(self):
        pass

    def act(self, observations):
        # observations contains the following keys: rgb(numpy RGB frame), depth (None by default), object_goal(category of target object)
        action = random.choice(['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Stop'])
        return action

#class RandomDistillateNetwork(nn.Module):
#
#    def __init__(self):
#        super(RandomDistillateNetwork, self).__init__()
#        pass
#
#
#class AgentPolicy(nn.Module):
#
#    def __init__(self):
#        super(AgentPolicy, self).__init__() 
#        
#
#
#        self.block0 = nn.Sequential([nn.Conv2D])

#
#    def forward(self, x, one_hot=None, mode="pre"):
#        
#        # calculate action or depth, segmentation, and classification if pre-training
#
#        if mode == "pre":
#            pass
#            #return classes, segmentation, depth_map
#
#        elif mode == "act":
#            assert one_hot is not None, "Need to specify goal object"
#            pass
#            #return act


class MyTempAgent(Agent):

    def __init__(self, conv_depth=16):

        num_actions = 7
        num_objects = 12

        block0 = nn.Sequential(\
                nn.Conv2d(3, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        #240x320
        block1 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))
    
        #120x160
        block2 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))
    
        #60x80
        block3 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))

        #30x40
        block4 = nn.Sequential(\
                nn.Conv2d(conv_depth, conv_depth, kernel_size=3,stride=1, padding=1),\
                nn.BatchNorm2d(conv_depth),\
                nn.ReLU(),\
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_extractor = nn.Sequential(block0, block1, block2,\
                block3, block4)

        dense_in = 15*20*conv_depth + num_objects
        dense_hid = 128

        dense_out = num_actions # number of possible actions

        self.flatten = nn.Flatten()

        self.dense_layers = nn.Sequential(\
                nn.Linear(dense_in, dense_hid),\
                nn.ReLU(),\
                nn.Linear(dense_hid, dense_out))


    def reset(self):
        pass

    def forward(self, x, one_hot):

        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = torch.cat((x, one_hot), dim=1)
        x = self.dense_layers(x)

        # x is the value of each 
        return x

    def act(self, observations):
        img_input = torch.Tensor(observations['rgb'].copy()[np.newaxis,...]).reshape(1,3,480,640)

        target_str = observations['object_goal']

        possible_targets = ['Alarm Clock', \
                            'Apple',\
                            'Baseball Bat',\
                            'Basketball',\
                            'Bowl',\
                            'Garbage Can',\
                            'House Plant',\
                            'Laptop',\
                            'Mug',\
                            'Spray Bottle',\
                            'Television',\
                            'Vase']

        possible_actions =  ['MoveAhead', 'MoveBack', 'RotateRight', \
                'RotateLeft', 'LookUp', 'LookDown', 'Stop']

        target_one_hot = torch.Tensor(\
                np.array([1.0 if target_str in elem else 0.0 for elem in possible_targets]))\
                .reshape(1,12)


        # get action from forward policy
        softmax_logits = self.forward(img_input, target_one_hot)

        act = possible_actions[torch.argmax(softmax_logits)]

        return act



if __name__ == "__main__":
    # run some tests

    my_agent_policy =  AgentPolicy()
    my_agent = MyTempAgent()
