from os import times
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import backward, Variable
import random
from random import sample


class Q_estimator(nn.Module):
  def __init__(self, scent_dim=3, action_embedding_vocab=3, action_embedding_size=3):
    super(Q_estimator, self).__init__()
    self.bin_feature_conv = nn.Sequential(*[nn.Conv2d(in_channels=4, out_channels= 8, kernel_size=(3,3), stride=3, padding='valid'),
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(3,3), stride=1, padding='valid'), 
                                            nn.Flatten()])
    self.action_embedding = nn.Embedding(action_embedding_vocab, action_embedding_size)
    self.linear_1 = nn.Linear(scent_dim + action_embedding_size + 9, 4)
    self.linear_2 = nn.Linear(4, 1)
    
  def forward(self, scent, bin_feat, action):
    bin_feat = self.bin_feature_conv(bin_feat)
    action = self.action_embedding(action)
    x = torch.cat((scent, bin_feat, action), dim=-1)
    x = F.relu(self.linear_1(x))
    return self.linear_2(x)


class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
     n-step semi-gradient Sarsa
  '''
  def __init__(self, env_specs, n=30, discount=0.85, learning_rate=0.05, epsilon=0.1, state_spaces_considered=['scent_space', 'feature_space'], actions_considered=[0,1,2]):
    self.env_specs = env_specs
    self.n = n
    self.discount = discount
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.actions_considered = actions_considered
    
    # get the q estimator input size (i.e. S and A)
    q_estimator_state_input_dim = 0
    self.state_space_indices = []
    idx = 0
    for k, v in env_specs.items():
      if k in state_spaces_considered:
        q_estimator_state_input_dim += v.shape[0]
        self.state_space_indices.append(idx)
      idx += 1

    # self.q_estimator = Q_estimator(q_estimator_state_input_dim)
    self.q_estimator = Q_estimator()
    self.optimizer = optim.SGD(self.q_estimator.parameters(), lr=self.learning_rate)

    self.episode_start = 0

    # define the components of G_t
    self.past_n_rewards = np.zeros(n)
    self.past_n_states = np.zeros(n, dtype=object)
    self.past_n_actions = np.zeros(n)
    self.past_n_G_t = np.zeros(n)
    powers = [i for i in range(n - 1, -1, -1)]
    bases = np.ones(n) * discount
    self.past_n_discounts = np.power(bases, powers)

  def load_weights(self):
    pass

  def act(self, curr_obs, mode='eval'):
    if curr_obs is None:
      return sample(self.actions_considered, 1)[0]
      # return self.env_specs['action_space'].sample()

    elif mode=='eval':
      # extract the current state from observations
      curr_state = self.extract_state(curr_obs)
      q_values = []
      for action in self.actions_considered:
        S_t_scent, S_t_bin_feat, A_t = self.prepare_q_estimator_input(curr_state, action)
        q_values.append(self.q_estimator(S_t_scent, S_t_bin_feat, A_t).item())
      # print(np.argmax(q_values))
      return np.argmax(q_values)

    else:
      uniform_draw = random.uniform(0,1)
      if uniform_draw < self.epsilon:
        return sample(self.actions_considered, 1)[0]
      else:
        curr_state = self.extract_state(curr_obs)
        q_values = []
        for action in self.actions_considered:
          S_t_scent, S_t_bin_feat, A_t = self.prepare_q_estimator_input(curr_state, action)
          q_values.append(self.q_estimator(S_t_scent, S_t_bin_feat, A_t).item())
        # print(np.argmax(q_values))
        return np.argmax(q_values)


  def extract_state(self, curr_obs):
    state = []
    for idx in self.state_space_indices:
      if idx == 2:
        state.append(np.array([[np.reshape(arr, (15,15)) for arr in np.split(curr_obs[idx], 4)]]))
      else:
        state.append(np.array([curr_obs[idx]]))
      # state.append(curr_obs[idx].flatten())
    return state

  def prepare_q_estimator_input(self, state, action):
    inputs_ = [torch.tensor(input_, dtype=torch.float32) for input_ in state]
    inputs_.append(torch.tensor([action], dtype=torch.int))
    # print(inputs_)
    # return torch.tensor(np.concatenate([state, [action]]), dtype=torch.float32)
    return inputs_

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    # identify when new episode begins
    if curr_obs is None:
      self.episode_start = timestep

    # since initial curr_obs are None, skip initial step
    if curr_obs is not None:
      # extract the current state from observations
      curr_state = self.extract_state(curr_obs)

      # update the n_step memory buffer
      self.past_n_states = np.roll(self.past_n_states, 1)
      self.past_n_states[0] = curr_state

      self.past_n_actions = np.roll(self.past_n_actions, 1)
      self.past_n_actions[0] = action

      self.past_n_rewards = np.roll(self.past_n_rewards, 1)
      self.past_n_rewards[0] = reward

      # print(timestep)
    
    # avoid updating q fct approximator until n+1 timesteps are reached (since initial curr_obs are None)
    if (timestep - self.episode_start) >= self.n and curr_obs is not None:

      S_t = self.past_n_states[-1]
      A_t = self.past_n_actions[-1]
      S_t_scent, S_t_bin_feat, A_t = self.prepare_q_estimator_input(S_t, A_t)

      S_tn_scent, S_tn_bin_feat, A_tn = self.prepare_q_estimator_input(curr_state, action)
      # G_t = (self.past_n_rewards * self.past_n_discounts).sum() + (np.power(self.discount, self.n) * self.q_estimator(S_tn_A_tn).item())

      self.optimizer.zero_grad()
      with torch.no_grad():
        G_t = (self.past_n_rewards * self.past_n_discounts).sum() + (np.power(self.discount, self.n) * self.q_estimator(S_tn_scent, S_tn_bin_feat, A_tn))
      MAE = nn.L1Loss()
      loss = MAE(self.q_estimator(S_t_scent, S_t_bin_feat, A_t), G_t)
      # print(loss)
      loss.backward()
      self.optimizer.step()
      
    
