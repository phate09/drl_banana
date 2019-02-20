import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import datetime

from dqn_agent import Agent
from utils.Scheduler import Scheduler

currentDT = datetime.datetime.now()
print(f'Start at {currentDT.strftime("%Y-%m-%d %H:%M:%S")}')
env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size=8, action_size=4, seed=0)
STARTING_BETA = 0.4


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    betas = Scheduler(STARTING_BETA, 1.0, n_episodes)
    eps = Scheduler(eps_start, eps_end, n_episodes * 0.1)
    for i_episode in range(n_episodes):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps.get(i_episode))
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, beta=betas.get(i_episode))
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        # eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} '
            f'eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}'
            , end="")
        if i_episode + 1 % 100 == 0:
            print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} '
                  f'eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}')
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
