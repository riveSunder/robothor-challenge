import robothor_challenge
from robothor_challenge.env import RobothorChallengeEnv
from robothor_challenge.agent import SimpleRandomAgent
from robothor_challenge.agent import MyTempAgent

import numpy as np

import torch
import torch.nn as nn
import copy

ALLOWED_ACTIONS =  ['MoveAhead', 'MoveBack', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Stop']

class DQN():

    def __init__(self, env, agent_fn):

        self.env = env

        # some training parameters
        self.lr = 1e-3
        self.batch_size = 64
        self.buffer_size = 512 #4096
        self.epsilon_decay = 0.99
        self.starting_epsilon = 0.99
        self.epsilon = self.starting_epsilon * 1.0
        self.min_epsilon = 0.025
        self.update_qt_every = 32
        self.gamma = 0.95

        # train on this device
        self.device = torch.device("cpu")

        # define the value and value target networks q and q_t

        self.q = agent_fn()
        self.qt = agent_fn()

        self.q.to(self.device)
        self.qt.to(self.device)

        for param in self.qt.parameters():
            param.requires_grad = False

    def get_episodes(self, steps=128):

        # define lists for recording trajectory transitions
        l_obs_x = torch.Tensor()
        l_obs_one_hot = torch.Tensor()
        l_rew = torch.Tensor()
        l_act = torch.Tensor()
        l_next_obs_x = torch.Tensor()
        l_next_obs_one_hot = torch.Tensor()
        l_done = torch.Tensor()

        # interaction loop
        done = True
        with torch.no_grad():
            for step in range(steps):

                if done:
                    self.q.reset()
                    observation = env.reset()
                    target_str = observation["object_goal"]

                    target_one_hot = torch.Tensor(\
                            np.array([1.0 if target_str in elem else 0.0 \
                            for elem in self.q.possible_targets]))\
                            .reshape(1,12)

                    obs = (torch.Tensor(observation["rgb"].copy()).reshape(1,3,480,640),\
                            target_one_hot)
                    done = False

                if torch.rand(1) < self.epsilon:
                    act = np.random.randint(len(ALLOWED_ACTIONS))
                    action = ALLOWED_ACTIONS[act]
                    act = torch.Tensor(np.array(1.0*act)).unsqueeze(0)
                else:
                    q_values = self.q.forward(obs[0], obs[1] )
                    action = ALLOWED_ACTIONS[torch.argmax(q_values)]
                    act = 1.0*torch.argmax(q_values).unsqueeze(0)

                prev_obs = obs
                observation, reward, done, info = self.env.step(action)
                target_str = observation["object_goal"]

                target_one_hot = torch.Tensor(\
                        np.array([1.0 if target_str in elem else 0.0 \
                        for elem in self.q.possible_targets]))\
                        .reshape(1,12)

                obs = (torch.Tensor(observation["rgb"].copy()).reshape(1,3,480,640),\
                        target_one_hot)

                if len(prev_obs[0].shape) == 3:
                    print("dimensional problem?")
                    import pdb; pdb.set_trace()



                l_obs_x = torch.cat([l_obs_x, prev_obs[0]], dim=0)
                l_obs_one_hot = torch.cat([l_obs_one_hot, prev_obs[1]], dim=0)
                l_rew = torch.cat([l_rew, torch.Tensor(np.array(1.*reward)).unsqueeze(0)], dim=0)
                l_act = torch.cat([l_act, act], dim=0)
                l_next_obs_x = torch.cat([l_next_obs_x, obs[0]], dim=0)
                l_next_obs_one_hot = torch.cat([l_next_obs_one_hot, obs[1]], dim=0)
                l_done = torch.cat([l_done,torch.Tensor(np.array(1.0*done)).unsqueeze(0)], dim=0)


            return l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, l_next_obs_one_hot, l_done 

    def compute_q_loss(self, t_obs_x, t_obs_one_hot, t_rew, t_act,\
            t_next_obs_x, t_next_obs_one_hot, t_done, double_dqn=True):

        with torch.no_grad():
            qt = self.qt.forward(t_next_obs_x, t_next_obs_one_hot)
            
            if double_dqn:
                qtq = self.q.forward(t_next_obs_x, t_next_obs_one_hot)
                qt_max = torch.gather(qt, -1, torch.argmax(qtq, dim=-1).unsqueeze(-1))
            else:
                qt_max = torch.gather(qt, -1, torch.argmax(qt, dim=-1).unsqueeze(-1))

            yj = t_rew + ((1-t_done) * self.gamma * qt_max)


        t_act = t_act.long().unsqueeze(1)
        q_av = self.q.forward(t_obs_x, t_obs_one_hot)

        q_act = torch.gather(q_av, -1, t_act)

        loss = torch.mean(torch.pow(yj - q_act, 2))

        return loss



    def train(self, max_epochs=1024):


        optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

        self.rewards = []
        self.losses = []

        for epoch in range(max_epochs):

            self.q.zero_grad()

            l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, \
                    l_next_obs_one_hot, l_done = self.get_episodes(steps=self.buffer_size)

            for batch in range(0,self.buffer_size-self.batch_size, self.batch_size):

                loss = self.compute_q_loss(l_obs_x[batch:batch+self.batch_size],\
                        l_obs_one_hot[batch:batch+self.batch_size], \
                        l_rew[batch:batch+self.batch_size], \
                        l_act[batch:batch+self.batch_size],\
                        l_next_obs_x[batch:batch+self.batch_size], \
                        l_next_obs_one_hot[batch:batch+self.batch_size], \
                        l_done[batch:batch+self.batch_size])

                loss.backward()

                optimizer.step()

            print("loss at epoch {}: {:.3e}".format(epoch, loss))

            # update target network every once in a while
            if epoch % self.update_qt_every == 0:

                self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))

                for param in self.qt.parameters():
                    param.requires_grad = False
            
            for my_buffer in [l_obs_x, l_obs_one_hot, l_rew, l_act, l_next_obs_x, l_next_obs_one_hot,\
                    l_done]:
                del my_buffer



            self.epsilon = np.max([self.min_epsilon, self.epsilon*self.epsilon_decay])
        
if __name__ == "__main__":

    agent_fn = MyTempAgent
    agent = MyTempAgent()
    env = RobothorChallengeEnv(agent=agent)
    dqn = DQN(env, agent_fn)


    dqn.train()

    
