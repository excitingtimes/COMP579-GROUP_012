from os import times
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import backward, Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
from random import sample



class Q_estimator(nn.Module):
  def __init__(self, scent_dim=3, action_embedding_vocab=5, action_embedding_size=50):
    super(Q_estimator, self).__init__()
    self.bin_feature_conv = nn.Flatten(start_dim=2, end_dim=- 1)
    self.action_embedding = nn.Embedding(action_embedding_vocab, action_embedding_size, padding_idx=4)
    self.lstm_action = nn.LSTM(action_embedding_size, action_embedding_size, num_layers=1, batch_first=True, bidirectional=False)
    self.lstm_scent = nn.LSTM(scent_dim, scent_dim, num_layers=1, batch_first=True, bidirectional=False)
    self.lstm_bin_feat = nn.LSTM(900, 900, num_layers=1, batch_first=True, bidirectional=False)
    self.linear_1 = nn.Linear(scent_dim + action_embedding_size + 900, 100)
    self.linear_2 = nn.Linear(100, 10)
    self.linear_3 = nn.Linear(10, 1)
    self.linear_4 = nn.Linear(5, 1)
    
  def forward(self, scent, bin_feat, action):
    action = torch.squeeze(self.action_embedding(action), -2)
    bin_feat = self.bin_feature_conv(bin_feat)
    
    output_action, (h_n_action, c_n_action) = self.lstm_action(action)
    output_scent, (h_n_scent, c_n_scent) = self.lstm_scent(scent)
    output_bin_feat, (h_n_bin_feat, c_n_bin_feat) = self.lstm_bin_feat(bin_feat)

    x = torch.cat((h_n_scent, h_n_bin_feat, h_n_action), dim=-1)
    x = F.relu(self.linear_1(torch.squeeze(x, -2)))
    x = F.relu(self.linear_2(x))
    return self.linear_3(x)



class Q_estimator2(nn.Module):
  def __init__(self, scent_dim=3, action_embedding_vocab=4, action_embedding_size=50):
    super(Q_estimator2, self).__init__()
    self.bin_feature_conv = nn.Sequential(*[nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(1,1), stride=1, padding='same'),
                                            torch.nn.ReLU(),
                                            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1,1), stride=1, padding='same'), 
                                            nn.Flatten()])
    self.action_embedding = nn.Embedding(action_embedding_vocab, action_embedding_size, padding_index=4)
    self.batch_norm = torch.nn.BatchNorm1d(scent_dim + action_embedding_size + 15*15)
    self.linear_1 = nn.Linear(scent_dim + action_embedding_size + 15*15, 50)
    self.linear_2 = nn.Linear(50, 5)
    self.linear_3 = nn.Linear(5, 1)
    self.dropout = nn.Dropout(p=0.2)
    
  def forward(self, scent, bin_feat, action):
    bin_feat = self.bin_feature_conv(bin_feat)
    action = self.action_embedding(action)
    x = torch.cat((scent, bin_feat, action), dim=-1)
    # x = self.batch_norm(x)
    x = F.relu(self.linear_1(x))
    # x = self.dropout(x)
    x = F.relu(self.linear_2(x))
    # x = self.dropout(x)
    return self.linear_3(x)



class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
     n-step semi-gradient Sarsa with Q value approximator
  '''
  def __init__(self, env_specs, n=50, discount=0.85, learning_rate=0.0001, epsilon=1, epsilon_decay=0.9,
               markovian_window=3, buffer_size=20000, n_epochs=20,
               state_spaces_considered=['scent_space', 'feature_space'], actions_considered=[0,1,2,3]):
    self.env_specs = env_specs
    self.n = n
    self.discount = discount
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.actions_considered = actions_considered
    self.markovian_window = markovian_window
    
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
    self.optimizer = optim.Adam(self.q_estimator.parameters(), lr=self.learning_rate, weight_decay=0.01)
    self.loss = nn.MSELoss() # or L1loss

    self.episode_start = 0

    # define the components of G_t
    self.past_n_rewards = np.zeros(n)
    self.past_n_states = np.zeros(n + markovian_window, dtype=object)
    self.past_n_actions = np.zeros(n + markovian_window)
    self.past_n_G_t = np.zeros(n)
    powers = [i for i in range(n - 1, -1, -1)]
    bases = np.ones(n) * discount
    self.past_n_discounts = np.power(bases, powers)

    # Define buffer
    self.buffer_size = buffer_size
    self.scent_buffer = []
    self.bin_feat_buffer = []
    self.action_buffer = []
    self.G_t_buffer = []

    self.n_epochs = n_epochs


  def load_weights(self, root_path):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
    pass


  def act(self, curr_obs, mode='eval'):
    # return sample(self.actions_considered, 1)[0] #TODO: REMOVE
    if curr_obs is None:
      return sample(self.actions_considered, 1)[0]
      # return self.env_specs['action_space'].sample()

    # extract the current state from observations
    curr_state = self.extract_state(curr_obs)
    q_values = []

    S_tn = np.roll(self.past_n_states, 1)
    S_tn[0] = curr_state
    S_tn = S_tn[0: self.markovian_window]

    A_tn = np.roll(self.past_n_actions, 1)

    self.q_estimator.eval()
    for action in self.actions_considered:
      A_tn[0] = action
      A_tn_ = A_tn[0: self.markovian_window]
      S_t_scent, S_t_bin_feat, A_t = self.prepare_q_estimator_input(S_tn, A_tn_)
      q_values.append(self.q_estimator(S_t_scent, S_t_bin_feat, A_t).item())
    # print(np.argmax(q_values))
    # print(q_values)
    act = np.argmax(q_values)
    self.q_estimator.train()
    
    if mode=='eval':
      # update the n_step memory buffer
      self.past_n_states = np.roll(self.past_n_states, 1)
      self.past_n_states[0] = curr_state

      self.past_n_actions = np.roll(self.past_n_actions, 1)
      self.past_n_actions[0] = act

      return act

    else:
      uniform_draw = random.uniform(0,1)
      if uniform_draw < self.epsilon:
        return sample(self.actions_considered, 1)[0]
      else:
        return act


  def extract_state(self, curr_obs):
    state = []
    for idx in self.state_space_indices:
      if idx == 2:
        state.append(np.array([np.reshape(arr, (15,15)) for arr in np.split(curr_obs[idx], 4)]))
      else:
        state.append(np.array(curr_obs[idx]))
      # state.append(curr_obs[idx].flatten())
    return state


  def prepare_q_estimator_input(self, state, action):
    if isinstance(state[-1], int):
      patch_state = [obs * 0 for obs in state[0]]
      for i, obs in enumerate(state):
        if obs == 0:
          state[i] = patch_state
          action[i] = 4

    # reverse elements for RNN
    state = state[::-1]
    action = action[::-1]

    inputs_ = []
    for idx in range(len(self.state_space_indices)):
      inputs_.append(torch.tensor(np.array([[input_[idx] for input_ in state]]), dtype=torch.float32))
    inputs_.append(torch.tensor(np.array([[[act] for act in action]]), dtype=torch.int))
    # return torch.tensor(np.concatenate([state, [action]]), dtype=torch.float32)
    return inputs_


  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    # identify when new episode begins
    if curr_obs is None:
      self.episode_start = timestep
    # decay epsilon every episode
    if timestep % 5000 == 0:
      self.epsilon *= self.epsilon_decay
      print(self.epsilon)

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

      S_t = self.past_n_states[self.n - 1: self.n - 1  + self.markovian_window]
      A_t = self.past_n_actions[self.n - 1: self.n - 1  + self.markovian_window]
      S_t_scent, S_t_bin_feat, A_t = self.prepare_q_estimator_input(S_t, A_t)

      S_tn = self.past_n_states[0: self.markovian_window]
      A_tn = self.past_n_actions[0: self.markovian_window]
      S_tn_scent, S_tn_bin_feat, A_tn = self.prepare_q_estimator_input(S_tn, A_tn)
      
      self.optimizer.zero_grad()
      with torch.no_grad():
        G_t = (self.past_n_rewards * self.past_n_discounts).sum() + (np.power(self.discount, self.n) * self.q_estimator(S_tn_scent, S_tn_bin_feat, A_tn))
      loss = self.loss(self.q_estimator(S_t_scent, S_t_bin_feat, A_t), G_t.detach())
      loss.backward()
      self.optimizer.step()

      # add to buffer
      self.scent_buffer.append(S_t_scent)
      self.bin_feat_buffer.append(S_t_bin_feat)
      self.action_buffer.append(A_t)
      self.G_t_buffer.append(G_t)

      # batch train the q estimator on buffer when buffer is full
      if (timestep - self.n) % self.buffer_size == 0:
        # train the q estimator on buffer
        # avg_train_losses, avg_valid_losses = self.train_q_estimator_on_buffer(n_epochs=self.n_epochs)

        # reset buffer
        self.scent_buffer = []
        self.bin_feat_buffer = []
        self.action_buffer = []
        self.G_t_buffer = []

    # if episode done, reset memory    
    if done:
      self.past_n_rewards = np.zeros(self.n)
      self.past_n_states = np.zeros(self.n + self.markovian_window, dtype=object)
      self.past_n_actions = np.zeros(self.n + self.markovian_window)
      self.past_n_G_t = np.zeros(self.n)
      print(timestep, 'done')


  def train_q_estimator_on_buffer(self, valid_size=0.25, batch_size=200, patience=4, n_epochs=50):
    """Base code from
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb"""

    # obtain the dataloaders from buffered data
    train_loader, valid_loader = self.transform_buffered_data(valid_size=valid_size, batch_size=batch_size)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        self.q_estimator.train() # prep model for training
        for batch, (S_t_scent, S_t_bin_feat, A_t, G_t) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.q_estimator(S_t_scent, S_t_bin_feat, A_t)
            # calculate the loss
            loss = self.loss(output, G_t)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            self.optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        self.q_estimator.eval() # prep model for evaluation
        for S_t_scent, S_t_bin_feat, A_t, G_t in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.q_estimator(S_t_scent, S_t_bin_feat, A_t)
            # calculate the loss
            loss = self.loss(output, G_t)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        # print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, self.q_estimator)
        
        if early_stopping.early_stop:
            # print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    self.q_estimator.load_state_dict(torch.load('checkpoint.pt'))
    self.q_estimator.train()

    return avg_train_losses, avg_valid_losses
  

  def transform_buffered_data(self, valid_size=0.2, batch_size=200):
    """Base code from
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb"""

    # create a dataset from bufferred data
    S_t_scent = torch.cat(self.scent_buffer)
    S_t_bin_feat = torch.cat(self.bin_feat_buffer)
    A_t = torch.cat(self.action_buffer)
    G_t = torch.cat(self.G_t_buffer)
    train_data = Make_dataset(S_t_scent, S_t_bin_feat, A_t, G_t)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)
    return train_loader, valid_loader



class Make_dataset(Dataset):
  """Obtained from 
  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files"""
  def __init__(self, S_t_scent, S_t_bin_feat, A_t, G_t):
    self.S_t_scent = S_t_scent
    self.S_t_bin_feat = S_t_bin_feat
    self.A_t = A_t
    self.G_t = G_t

  def __len__(self):
    return len(self.S_t_scent)

  def __getitem__(self, idx):
    S_t_scent = self.S_t_scent[idx]
    S_t_bin_feat = self.S_t_bin_feat[idx]
    A_t = self.A_t[idx]
    G_t = self.G_t[idx]
    return S_t_scent, S_t_bin_feat, A_t, G_t



class EarlyStopping:
    """Obtained from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
