import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy


class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Load a saved table of Q-values for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        if (model is None):
            self.model = np.zeros(self.buckets + (actionsize,))
        else:
            self.model = model

    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation  
        """
        upper_bounds = [self.env.observation_space.high[0], 5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        qvals = []
        for state in states:
            indices = self.discretize(state)
            qvals.append(self.model[indices[0]][indices[1]][indices[2]][indices[3]])
        return qvals


    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        if (done):
            target = reward
        else:
            indices = self.discretize(next_state)
            target = reward + self.gamma * np.max(self.model[indices[0]][indices[1]][indices[2]][indices[3]])
        indices = self.discretize(state)
        old = self.model[indices[0]][indices[1]][indices[2]][indices[3]][action]
        self.model[indices[0]][indices[1]][indices[2]][indices[3]][action] = old + self.lr * (target - old)
        loss = (target - old)**2
        return loss


    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(3, 3, 12, 12), actionsize=actionsize, lr=0.1, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'tabular.npy')
