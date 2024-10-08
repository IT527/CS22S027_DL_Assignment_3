import numpy as np
import torch
import torch.nn as nn
import heapq #For Beam Search
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import wandb
from tqdm import tqdm #To keep track of progress loops
from arguments import *
args = parsArg()
device = args.device
"""#Sequence Encoder"""

class SequenceEncoder(nn.Module):
    '''
    This class defines a sequence encoder with customizable parameters such as input size, embedding size,
    and the type of recurrent layer to use (e.g., GRU, LSTM, or RNN).
    '''
    def __init__(self, input_size=29, embedding_size=256, hidden_size=256, rnn_type="lstm", num_layers=2, use_bidirectional=True, dropout_rate=0, device = device):
        super(SequenceEncoder, self).__init__()

        self.config = {
            'input_size': input_size,
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'rnn_type': rnn_type,
            'num_layers': num_layers,
            'is_bidirectional': use_bidirectional,
            'dropout_rate': dropout_rate,
            # 'device': device.type,
            'device': device,
        }

        # Define RNN parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.direction_factor = 2 if use_bidirectional else 1
        self.device = self.device = torch.device(device)

        # Define the embedding layer
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(self.input_size, self.embedding_size)

        # Map the rnn_type string to the corresponding PyTorch class
        rnn_cell = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }[rnn_type]
        #Handling case when unxpected cell type input recieved
        if rnn_cell is None:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Choose from 'rnn', 'gru', or 'lstm'.")

        # Define the RNN layer
        self.rnn_layer = rnn_cell(input_size=self.embedding_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  dropout=(dropout_rate if num_layers > 1 else 0),  # PyTorch applies dropout only if num_layers > 1
                                  bidirectional=use_bidirectional)

    def forward(self, sequence_input, hidden_state, cell_state=None):
        '''
          Forward pass through the network.

          Parameters:
          - sequence_input: Tensor of shape (seq_len, batch_size)
          - hidden_state: Tensor of shape (num_layers * num_directions, batch_size, hidden_size)
          - cell_state: Only for LSTM, Tensor of shape (num_layers * num_directions, batch_size, hidden_size)

          Returns:
          - outputs: Tensor of shape (seq_len, batch_size, num_directions * hidden_size)
          - hidden_state: Tensor of shape (num_layers * num_directions, batch_size, hidden_size)
          - cell_state: Only for LSTM, Tensor of shape (num_layers * num_directions, batch_size, hidden_size)
        '''
        embedding_output = self.embedding_layer(sequence_input)
        embedding_output = self.dropout_layer(embedding_output)

        if self.config['rnn_type'] == 'lstm':
            outputs, (hidden_state, cell_state) = self.rnn_layer(embedding_output, (hidden_state, cell_state))
        else:
            outputs, hidden_state = self.rnn_layer(embedding_output, hidden_state)

        return outputs, hidden_state, cell_state

    def get_configuration(self):
        '''Returns the configuration of the model.'''
        return self.config

    def initialize_hidden_state(self, batch_size):
        '''Initializes the hidden state for the RNN.'''
        return torch.zeros(self.direction_factor * self.num_layers, batch_size, self.hidden_size, device=device)
