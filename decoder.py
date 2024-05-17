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
from accuracy import *
from encoder import *
from attention_mechanism import *
from beam_search import *
args = parsArg()

"""#Sequence Decoder"""

class SequenceDecoder(nn.Module):
    '''
      A flexible sequence decoder supporting various RNN architectures (GRU, LSTM, RNN) and optional attention mechanism.

      Attributes:
        input_size (int): Size of the input vocabulary.
        embedding_size (int): Dimension of the embedding vectors.
        hidden_size (int): Number of features in the hidden state of the RNN.
        rnn_type (str): Type of RNN cell to use ('rnn', 'gru', or 'lstm').
        num_layers (int): Number of recurrent layers.
        use_attention (bool): Flag to use attention mechanism.
        attention_dimension (int): Dimension of the attention mechanism.
        dropout_rate (float): Dropout rate for regularization.
        use_bidirectional (bool): Flag to use bidirectional RNN.
        device (torch.device): Device to run the model on.
    '''
    def __init__(self, input_size=67, embedding_size=256, hidden_size=256, rnn_type="lstm", num_layers=3, use_attention=False, attention_dimension=None, dropout_rate=0, use_bidirectional=False, device=args.device):
        super(SequenceDecoder, self).__init__()
        # Define RNN Parameter
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.dropout_layer = nn.Dropout(dropout_rate)
        # Define the embedding layer
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        self.input_size = embedding_size
        self.attention_dimension = 0
        self.attention_out_dimension = 1


        self.direction_factor = 2 if use_bidirectional else 1
        self.device = device

        self.rnn = self.init_rnn(rnn_type, embedding_size + (attention_dimension if use_attention else 0),
                                 hidden_size, num_layers, dropout_rate, use_bidirectional)
        # Define the linear layer to map RNN outputs to the output vocabulary size
        self.W1 = nn.Linear(hidden_size * self.direction_factor, input_size)
        self.softmax = F.softmax
        #Add attention mechanish if required
        if use_attention:
            self.attention = Attention(hidden_size, self.attention_out_dimension)

    def init_rnn(self, cell_type, input_size, hidden_size, num_layers, dropout, bidirectional): #Defining RNN Cell type
        rnn_cell = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }.get(cell_type)

        if rnn_cell is None:
            raise ValueError(f"Unsupported rnn_type '{cell_type}'. Choose from 'rnn', 'gru', or 'lstm'.")

        return rnn_cell(input_size, hidden_size, num_layers, dropout=dropout,  bidirectional=bidirectional)

    def forward(self, input, hidden, cell=None, encoder_outputs=None):
        '''
          Forward pass through the sequence decoder.

          Args:
              input (torch.Tensor): Input tensor.
              hidden (torch.Tensor): Hidden state tensor.
              cell (torch.Tensor, optional): Cell state tensor (for LSTM).
              encoder_outputs (torch.Tensor, optional): Encoder outputs for attention mechanism.

          Returns:
              Output tensor, hidden state tensor, cell state tensor (for LSTM), attention weights (if used attention mechanism).
        '''
        embedded = self.dropout_layer(self.embedding(input))
        attention_weights = None
        if self.use_attention:
            context, attention_weights = self.attention.forward(hidden, encoder_outputs)
            # print("Embedded shape:", embedded.shape)
            # print("Context shape:", context.shape)
            embedded = torch.cat((embedded, context), 2)

        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            output, hidden = self.rnn(embedded, hidden)
        output = self.W1(output)
        return output, hidden, cell, attention_weights

    def getParams(self):
        '''Returns the configuration of the model.'''
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
