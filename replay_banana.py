import torch
from unityagents import UnityEnvironment

from dqn_agent import Agent

env = UnityEnvironment(file_name="/home/edoardo/Downloads/Banana_Linux/Banana.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of actions
action_size = brain.vector_action_space_size
# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)
score = 0  # initialize the score
agent = Agent(state_size=state_size, action_size=action_size, seed=0, alpha=0.6)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
while True:
    state = env_info.vector_observations[0]  # get the current state
    action = agent.act(state)
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    state = next_state
    score += reward
    if done:
        break

print("Score: {}".format(score))
