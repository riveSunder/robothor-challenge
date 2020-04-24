
import robothor_challenge
from robothor_challenge import RobothorChallenge
from robothor_challenge.agent import SimpleRandomAgent
from robothor_challenge.agent import MyTempAgent

import numpy as np

import torch
import torch.nn as nn


ALLOWED_ACTIONS =  ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Stop']

class RobothorChallengeEnv(RobothorChallenge):


    def reset(self, depth_frame=False):
        # sample an episode randomly from self.episodes
        num_epds = len(self.episodes)
        self.episode = self.episodes[np.random.randint(num_epds)]
        
        event = self.controller.last_event
        self.controller.initialization_parameters['robothorChallengeEpisodeId'] = self.episode['id']
        self.controller.reset(self.episode['scene'])
        teleport_action = dict(action='TeleportFull')
        teleport_action.update(self.episode['initial_position'])
        self.controller.step(action=teleport_action)
        self.controller.step(action=dict(action='Rotate', rotation=dict(y=self.episode['initial_orientation'], horizon=0.0)))
        self.total_steps = 0
        self.agent.reset()
        self.stopped = False


        obs = dict(object_goal=self.episode['object_type'], depth=event.depth_frame, rgb=event.frame)
        return obs


    def step(self, action ):

        self.total_steps += 1
        event = self.controller.last_event
        event.metadata.clear()
        
        if action not in ALLOWED_ACTIONS:
            raise ValueError('Invalid action: {action}'.format(action=action))

        event =  self.controller.step(action=action)

        obs = dict(object_goal=self.episode['object_type'], depth=event.depth_frame, rgb=event.frame)

        stopped = action == 'Stop'

        if stopped:
            simobj = self.controller.last_event.get_object(self.episode['object_id'])
            reward = 1.0 * simobj['visible']
            if reward == 1.0: print("winner!")
        else:
            reward = 0.0

        simobj = self.controller.last_event.get_object(self.episode['object_id'])
        if simobj['visible']: 
            reward += 0.1
            #print('Target object visible!!!')

        # reward for keeping on
        reward += 0.01

        done = stopped or self.total_steps >= self.config['max_steps']
        info = {}

        return obs, reward, done, info

    def sample_action_space(self):
        pass



if __name__ == "__main__":

    # run a quick testV 

    print("instantiate agent and environment")
    #agent = SimpleRandomAgent()
    agent = MyTempAgent()
    env = RobothorChallengeEnv(agent=agent)
    
    done = False

    print("step through a few episodes in env")
    for trial in range(1000):
        obs = env.reset()
        done = False
        sum_rewards = 0.0
        import pdb; pdb.set_trace()
        while not done:

            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            sum_rewards += reward
        print("sum of rewards/done = {}/{}".format(sum_rewards, done), \
                obs['rgb'].shape, obs['object_goal'])
            
        
