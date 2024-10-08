import argparse
import torch

def parsArg():
    parser = argparse.ArgumentParser(description="Train Seq2Seq model with specified parameters.")
    parser.add_argument('-wp', '--wandb_project',default='CS6910_Assignment_3', help='Project name used to track experiments in Weights & Biases dashboard.')
    parser.add_argument('-we', '--wandb_entity',default='cs22s027', help='WandB Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-ehd', '--encoder_hidden_dimension', type=int, default=256, help='Encoder hidden dimension size')
    parser.add_argument('-dhd', '--decoder_hidden_dimension', type=int, default=256, help='Decoder hidden dimension size')
    parser.add_argument('-eed', '--encoder_embed_dimension', type=int, default=256, help='Encoder embedding dimension size')
    parser.add_argument('-ded', '--decoder_embed_dimension', type=int, default=256, help='Decoder embedding dimension size')
    parser.add_argument('-b','--batch_size', type=int, default=128, help='Batch size used to train the model')
    parser.add_argument('-bd', '--bidirectional', type=str, default=True, choices=['True','False'], help='Whether to use bidirectional pass or not')
    parser.add_argument('-enl', '--encoder_num_layers', type=int, default=2, help='Number of layers in the encoder')
    parser.add_argument('-dnl', '--decoder_num_layers', type=int, default=3, help='Number of layers in the decoder')
    parser.add_argument('-ct', '--cell_type', type=str, default='lstm', choices=['lstm', 'gru', 'rnn'], help='Type of RNN cell to use')
    parser.add_argument('-dp', '--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('-bw', '--beam_width', type=int, default=1, help='Beam width for beam search')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('-bm', '--beam', type=str, default=True, choices=['True','False'], help='Whether to use beam search decoding')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('-a','--attention', type=str, default=False, choices=['True','False'], help='Need Attention or not')

    #Parse the arguments
    args = parser.parse_args()

    return args
