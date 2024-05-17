import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from arguments import *

args = parsArg()

"""# Data Preprocessing

## Data Vocab Disctionaries
"""

def create_char_index_dict(data):
    """
    Creates dictionaries mapping characters to indices and vice versa from the given data.

    Parameters:
        data (numpy.ndarray): The input data array containing English or Hindi texts.

    Returns:
        tuple: Two dictionaries mapping characters to indices and indices to characters.
    """
    char_index_dict = {}
    index_char_dict = {}
    index = 0
    # Adding special symbols
    special_symbols = ['<Pad>', '<Go>', '<Stop>'] #Add padding to handle unexpected large sequence in test data
    for symbol in special_symbols:
        char_index_dict[symbol] = index
        index_char_dict[index] = symbol
        index += 1

    for text in data.flatten():
        for char in text:
            if char not in char_index_dict:
                char_index_dict[char] = index
                index_char_dict[index] = char
                index += 1


    return char_index_dict, index_char_dict

def data_dict(file_path):
    """
    Loads data from a CSV file and creates dictionaries for character indices.

    Parameters:
        file_path (str): The path to the CSV file containing English or Hindi texts.

    Returns:
        Respective dictionaries mappings for character to indices and indices to characters.
    """
    try:
        arr = np.loadtxt(file_path, delimiter=",", encoding='utf-8', dtype=str)
    except Exception as e:
        raise FileNotFoundError(f"Unable to load data from {file_path}: {e}")

    english_dict, english_index_dict = create_char_index_dict(arr[:, 0]) #Dictionary mappings for English
    hindi_dict, hindi_index_dict = create_char_index_dict(arr[:, 1]) #Dictionary mappings for Hindi
    # max_len_english = max(len(row) for row in arr[:, 0]) #Computing max length of sequence in English
    # max_len_hindi = max(len(row) for row in arr[:, 1]) #Computing max length of sequence in Hindi

    return english_dict, hindi_dict, english_index_dict, hindi_index_dict

"""##Creating Tensor Dataset"""

def encode_text(text, dictionary, seq_len):
      encoded = [1] + [dictionary.get(char, 0) for char in text[:seq_len-2]] + [2]
      return np.pad(encoded, (0, max(0, seq_len - len(encoded))), mode='constant')


def load_and_process_data(file_path, eng_dict, hin_dict, eng_seq_len=29, hin_seq_len=32, dtype=torch.int64, device=args.device):
    """
    Loads and preprocesses text data from a CSV file, and converts it into PyTorch tensors.

    Parameters:
        file_path (str): Path to the CSV data file.
        eng_dict (dict): Dictionary mapping English characters to numerical indices.
        hin_dict (dict): Dictionary mapping Hindi characters to numerical indices.
        eng_seq_len (int): The maximum length of English sequences.
        hin_seq_len (int): The maximum length of Hindi sequences.
        dtype (torch.dtype, optional): Data type of the tensors.
        device (str, optional): The device to transfer tensors ('cpu' or 'cuda').

    Returns:
       Two tensors representing input data and target data.
    """

    seq = np.loadtxt(file_path, delimiter=",", encoding='utf-8', dtype=str)

    inputs = []
    targets = []
    for eng_text, hin_text in seq:
        inputs.append(encode_text(eng_text, eng_dict, eng_seq_len))
        targets.append(encode_text(hin_text, hin_dict, hin_seq_len))

    input_tensor = torch.tensor(inputs, dtype=dtype).to(device)
    target_tensor = torch.tensor(targets, dtype=dtype).to(device)


    return input_tensor, target_tensor
