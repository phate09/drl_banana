import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


from dqn_agent import Agent

env = gym.make('LunarLander-v2')
env.seed(0)
env = gym.wrappers.Monitor(env, './video/', force = True)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


agent = Agent(state_size=8, action_size=4, seed=0)
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(10):
    state = env.reset()
    for j in range(2000):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

env.close()