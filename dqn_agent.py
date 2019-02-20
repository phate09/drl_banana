import numpy as np
import random
from collections import deque

from model import QNetwork

import torch
import torch.nn as nn
import torch.optim as optim

# from old.replay_buffer import PrioritizedReplayBuffer
from utils.PrioReplayBuffer import PrioReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-3  # learning rate
UPDATE_EVERY = 16  # how often to update the network
# UPDATE_TARGET_EVERY = 100 * UPDATE_EVERY #20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# noinspection SpellCheckingInspection
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioReplayBuffer(buf_size=BUFFER_SIZE)  # ReplayBuffer(BUFFER_SIZE)
        self.local_memory = []
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # self.t_update_target_step = 0

    def step(self, state, action, reward, next_state, done, beta):
        # Save experience in replay memory
        self.local_memory.append((state, action, reward, next_state, done))
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # self.t_update_target_step = (self.t_update_target_step + 1) % UPDATE_TARGET_EVERY

        if self.t_step == 0:

            with torch.no_grad():
                states, actions, rewards, next_states, dones = [list(tup) for tup in
                                                                zip(*self.local_memory)]
                states = torch.FloatTensor(states).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                actions = torch.LongTensor(actions).to(device)

                Qa_s = self.qnetwork_local(states).gather(1, actions.unsqueeze(-1)).squeeze(
                    -1)  # state_action_values #TODO a mistake is about here
                Qa_next_s = self.qnetwork_target(next_states).max(1)[0]  # the maximum Q at the next state
                target_Q = rewards + GAMMA * Qa_next_s * (1 - dones)
                # calculate td-errors
                td_errors = torch.abs(target_Q - Qa_s)
                # store the local memory in the PER
                for memory, error in zip(self.local_memory, td_errors):
                    self.memory.populate(memory)  # td_errors
            # empty the memory
            self.local_memory = []
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, indexes, is_values = self.memory.sample(BATCH_SIZE, beta=beta)
                self.learn(experiences, indexes, is_values)
                # ------------------- update target network ------------------- #
                self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, indexes, is_values):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        actions = torch.LongTensor(actions).to(device)
        is_values = torch.FloatTensor(is_values).to(device)

        self.optimizer.zero_grad()  # resets the gradient

        Qa_s = self.qnetwork_local(states).gather(1, actions.unsqueeze(-1)).squeeze(
            -1)  # state_action_values #TODO a mistake is about here
        Qa_next_s = self.qnetwork_target(next_states).max(1)[0]  # the maximum Q at the next state
        target_Q = rewards + GAMMA * Qa_next_s * (1 - dones)
        td_errors = torch.abs(target_Q - Qa_s)
        # updates the priority by using the newly computed td_errors
        self.memory.update_priorities(indexes, td_errors)
        # Notice that detach for the target_Q
        error = 0.5 * (target_Q.detach() - Qa_s).pow(2) * is_values# * td_errors
        error = error.mean()
        error.backward()
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1, norm_type=float("inf"))
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)

        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def push(self, memory, td_error):
        """Add a new experience to memory."""
        # e = self.experience(state, action, reward, next_state, done)
        self.memory.append(memory)

    def sample(self, batch_size, beta):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(
        #     np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        # dones = torch.from_numpy(
        #     np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return experiences, 1.0, 1.0

    def update_priorities(self, a, b):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
