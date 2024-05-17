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
from attention_mechanism import *
from beam_search import *
from data_preparation import *
from decoder import *
from encoder import *
"""#Seq2Seq"""

class Seq2Seq(nn.Module):

    '''
    Modular Seq2Seq model for transliteration tasks with optional attention mechanism.
    '''
    def __init__(self, **kwargs):
        super(Seq2Seq, self).__init__()
        self.kwargs = kwargs
        self.configure_parameters(kwargs)
        self.initialize_components()

    def configure_parameters(self, params):
        # Unpack parameters or set default values
        self.input_seq_length = params.get('input_seq_length', 29)
        self.output_seq_length = params.get('output_seq_length', 32)
        self.encoder_input_dimension = params.get('encoder_input_dimension', 32)
        self.decoder_input_dimension = params.get('decoder_input_dimension', 67)
        self.encoder_hidden_dimension = params.get('encoder_hidden_dimension', 256)
        self.decoder_hidden_dimension = params.get('decoder_hidden_dimension', 256)
        self.encoder_embed_dimension = params.get('encoder_embed_dimension', 256)
        self.decoder_embed_dimension = params.get('decoder_embed_dimension', 256)
        self.bidirectional = params.get('bidirectional', True)
        self.encoder_num_layers = params.get('encoder_num_layers', 2)
        self.decoder_num_layers = params.get('decoder_num_layers', 3)
        self.cell_type = params.get('cell_type', 'lstm')
        self.dropout = params.get('dropout', 0.0)
        self.beam_width = params.get('beam_width', 1)
        self.device = params.get('device')
        if isinstance(self.beam_width, tuple):
            self.beam_width = self.beam_width[0]  
        else:
            self.beam_width = int(self.beam_width)

        #We haven't explored attention with beam search
        self.use_attention = params.get('attention', True)
        if self.beam_width > 1:
            self.use_attention = False
        self.direction_value = 2 if self.bidirectional else 1

    def initialize_components(self):
        # Initialize encoder and decoder
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        # Initialize linear transformation layers for matching dimensions
        self.init_linear_transformations()

    def create_encoder(self):
        # Create and return an encoder component
        return SequenceEncoder(
            input_size=self.encoder_input_dimension,
            embedding_size=self.encoder_embed_dimension,
            hidden_size=self.encoder_hidden_dimension,
            rnn_type=self.cell_type,
            num_layers=self.encoder_num_layers,
            use_bidirectional=self.bidirectional,
            dropout_rate=self.dropout,
            device=self.device
        )

    def create_decoder(self):
        # Create and return a decoder component
        return SequenceDecoder(
            input_size=self.decoder_input_dimension,
            embedding_size=self.decoder_embed_dimension,
            hidden_size=self.decoder_hidden_dimension,
            attention_dimension=self.decoder_hidden_dimension,
            rnn_type=self.cell_type,
            num_layers=self.decoder_num_layers,
            dropout_rate=self.dropout,
            device=self.device,
            use_attention=self.use_attention,
            use_bidirectional=self.bidirectional
        )

    def init_linear_transformations(self):
        # Initialize linear transformations for hidden state, cell state, and attention
        self.enc_dec_linear1 = nn.Linear(self.encoder_hidden_dimension, self.decoder_hidden_dimension)
        self.enc_dec_linear2 = nn.Linear(self.encoder_num_layers*self.direction_value, self.decoder_num_layers*self.direction_value)
        self.enc_dec_cell_linear1 = nn.Linear(self.encoder_hidden_dimension, self.decoder_hidden_dimension)
        self.enc_dec_cell_linear2 = nn.Linear(self.encoder_num_layers*self.direction_value, self.decoder_num_layers*self.direction_value)
        self.enc_dec_att_linear1 = nn.Linear(self.encoder_hidden_dimension, self.decoder_hidden_dimension)
        self.enc_dec_att_linear2 = nn.Linear(self.encoder_num_layers*self.direction_value, self.decoder_num_layers*self.direction_value)
        self.softmax = F.softmax



    def forward(self, input, target, teacher_force, beam=False):
        """
          Forward pass for the sequence-to-sequence model.

          Parameters:
          input (torch.Tensor): The input sequence tensor of shape (batch_size, input_seq_length).
          target (torch.Tensor): The target sequence tensor of shape (batch_size, output_seq_length).
          teacher_force (bool): Flag indicating whether to use teacher forcing during training.
          beam (bool): Flag indicating whether to use beam search decoding. Default is False.

          Returns:
          predicted (torch.Tensor): The predicted output sequence tensor of shape (output_seq_length, batch_size, decoder_input_dimension).
          att_weights (torch.Tensor): The attention weights tensor of shape (output_seq_length, input_seq_length, direction_value * decoder_num_layers, batch_size).
        """
        batch_size = input.shape[0]
        enc_hidden = self.encoder.initialize_hidden_state(batch_size) # Initialize hidden state for the encoder
        cell = self.encoder.initialize_hidden_state(batch_size) if self.cell_type == 'lstm' else None  # Initialize cell state if using LSTM

        # Initialize encoder outputs if using attention
        encoder_outputs = torch.zeros(
            self.input_seq_length,
            self.direction_value * self.decoder_num_layers,
            batch_size,
            self.decoder_hidden_dimension,
            device=self.device
        ) if self.use_attention else None

        for t in range(self.input_seq_length): #Batch input to encoder
            enc_output, enc_hidden, cell = self.encoder(input[:, t].unsqueeze(0), enc_hidden, cell)

            if self.use_attention:
                enc_hidden_new = self.enc_dec_att_linear1(enc_hidden)
                enc_hidden_new = enc_hidden_new.permute(2, 1, 0).contiguous()
                enc_hidden_new = self.enc_dec_att_linear2(enc_hidden_new)
                enc_hidden_new = enc_hidden_new.permute(2, 1, 0).contiguous()
                encoder_outputs[t] = enc_hidden_new

        enc_last_state = enc_hidden #s_o for decoder
        predicted = torch.zeros(self.output_seq_length, batch_size, self.decoder_input_dimension, device=self.device)

        #Store attention weights for plotting attention heatmaps
        att_weights = torch.zeros(
            self.output_seq_length,
            self.input_seq_length,
            self.direction_value * self.decoder_num_layers,
            batch_size,
            device=self.device
        )

        dec_hidden = self.enc_dec_linear2(self.enc_dec_linear1(enc_last_state).permute(2, 1, 0).contiguous()).permute(2, 1, 0).contiguous() #Configuring dimensions
        if self.cell_type == 'lstm': #If LSTM, last state to decoder
            cell = self.enc_dec_cell_linear2(self.enc_dec_cell_linear1(cell).permute(2, 1, 0).contiguous()).permute(2, 1, 0).contiguous()

        # Initial output (1's, representing <SOS>)
        output = torch.ones(1, batch_size, dtype=torch.long, device=self.device)
        predicted[0, :, 1] = torch.ones(batch_size)
        attention_weights = None

        for t in range(1, self.output_seq_length): #Decoding characters
            if teacher_force: #Teacher forcing in initial epochs for better learning
                output, dec_hidden, cell, attention_weights = self.decoder(target[:, t-1].unsqueeze(0), dec_hidden, cell, encoder_outputs)
                predicted[t] = output.squeeze(0)
            else:
                if self.beam_width > 1 and beam: #Beam search during inference onlyy.
                    beam = BeamSearchDecoder(beam_size=self.beam_width)
                    beam.search(self, output, dec_hidden, cell, predicted)
                    break

                output, dec_hidden, cell, attention_weights = self.decoder(output, dec_hidden, cell, encoder_outputs) #Greedy search
                predicted[t] = output.squeeze(0)

                if self.use_attention:
                    att_weights[t] = attention_weights.squeeze(3)

                # Prepare output for the next input
                output = self.softmax(output, dim=2)
                output = torch.argmax(output, dim=2)

        return predicted, att_weights
