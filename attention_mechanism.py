import numpy as np
import torch
import torch.nn as nn
import heapq #For Beam Search
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import wandb
from tqdm import tqdm #To keep track of progress loops

"""#Attention Mechanism"""

class Attention(nn.Module):
    '''
     Attention mechanism for sequence decoding.
    '''
    def __init__(self, hidden_dim, attention_out_dim):
        super(Attention, self).__init__()
        self.W = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.V = nn.Sequential(nn.Linear(hidden_dim, attention_out_dim), nn.LeakyReLU())

    def forward(self, hidden, encoder_outputs):
        # Apply linear transformation and non-linearity
        encoder_transform = self.W(encoder_outputs)
        hidden_transform = self.U(hidden)

        # Calculate attention scores
        concat_transform = encoder_transform + hidden_transform
        score = torch.tanh(concat_transform)
        scores = self.V(torch.tanh(concat_transform))  # [batch_size, seq_len]

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=0)  # [batch_size, seq_len, 1]

        # Create context vector
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=0)  # [batch_size, hidden_dim]
        context_vector = torch.sum(context_vector,dim=0).unsqueeze(0) # Unsqueezed to match `embedded` dimensions
        return context_vector, attention_weights
