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
from seq2seq_model import *
from train_execution import *

args = parsArg()
key = '90512e34eff1bdddba0f301797228f7b64f546fc'
batch_size = args.batch_size
encoder_hidden_dimension = args.encoder_hidden_dimension
decoder_hidden_dimension = args.decoder_hidden_dimension
encoder_embed_dimension = args.encoder_embed_dimension
decoder_embed_dimension = args.decoder_embed_dimension
bidirectional = args.bidirectional
encoder_num_layers = args.encoder_num_layers
decoder_num_layers = args.decoder_num_layers
cell_type = args.cell_type
dropout = args.dropout
beam = args.beam
beam_width = args.beam_width
device = args.device
attention = args.attention
epochs = args.epochs
wandb.login(key = key)


"""#Train and Evaluate"""
import warnings
warnings.filterwarnings("ignore")
file_path = "aksharantar_sampled/hin/hin_train.csv"
english_dict, hindi_dict, english_index_dict, hindi_index_dict = data_dict(file_path)

X_train, y_train = load_and_process_data("aksharantar_sampled/hin/hin_train.csv", english_dict, hindi_dict)
X_val, y_val = load_and_process_data("aksharantar_sampled/hin/hin_valid.csv", english_dict, hindi_dict)
X_test, y_test = load_and_process_data("aksharantar_sampled/hin/hin_test.csv", english_dict, hindi_dict)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset,shuffle=False,batch_size=batch_size)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

config = {
    'input_embedding': args.encoder_embed_dimension,
        'batch_size' : args.batch_size,
        'number_of_enc_layer': args.encoder_num_layers,
        'number_of_dec_layer': args.decoder_num_layers,
        'hidden_size': args.encoder_hidden_dimension,
        'cell_type': args.cell_type,
        'bidirectional' : args.bidirectional,
        'dropout': args.dropout,
        'beam_width' : args.beam_width,
        'epochs' : args.epochs
}
run = wandb.init(config=config, project=args.wandb_project, entity=args.wandb_entity)

model = Seq2Seq(
    encoder_hidden_dimension = encoder_hidden_dimension,
    decoder_hidden_dimension =decoder_hidden_dimension,
    encoder_embed_dimension = encoder_embed_dimension,
    decoder_embed_dimension = decoder_embed_dimension,
    bidirectional = bidirectional,
    encoder_num_layers = encoder_num_layers,
    decoder_num_layers = decoder_num_layers,
    cell_type = cell_type,
    dropout = dropout,
    beam_width = beam_width,
    device = device,
    attention = attention
)
beam = beam
model.to(device)
epochs = epochs
losses_train, losses_val, accuracies_train, accuracies_val = train_and_evaluate_model(model, train_loader, val_loader, epochs, beam, device=device)
wandb.run.name = f'in_emb_{wandb.config.input_embedding}_bs_{wandb.config.batch_size}_enc_layers_{wandb.config.number_of_enc_layer}_dec_layers_{wandb.config.number_of_dec_layer}_hls_{wandb.config.hidden_size}_cell_{wandb.config.cell_type}_bidir_{wandb.config.bidirectional}_dp_{wandb.config.dropout}_bms_{wandb.config.beam_width}_epoch_{wandb.config.epochs}'
test_loss, test_accuracy = evaluate(model, test_loader, criterion=nn.CrossEntropyLoss(), device=device, beam=False)
print(f'Test loss: {test_loss} \nTest Accuracy: {test_accuracy}')