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
from decoder import *
from encoder import *
"""#Beam Search"""

class SearchNode(object):
    '''
    Represents a node in the beam search, storing relevant information.
    '''
    def __init__(self, probability=1, path_probability=0, index=1, hidden=None, cell=None, parent=None):
        self.probability = probability  # Probability of the current character
        self.path_probability = path_probability  # Cumulative probability of the path
        self.index = index  # Index of the current element
        self.hidden = hidden  # Hidden state from the model
        self.cell = cell  # Cell state for LSTM models
        self.parent = parent  # Parent node
        self.length = 0  # Length of the path

    def __lt__(self, other):
        # Override less than operator to prioritize nodes with higher path probability in the heap
        return self.path_probability > other.path_probability


class BeamSearchDecoder():
    '''
    Implements beam search decoding using a beam width to limit the number of nodes explored.
    '''
    def __init__(self, beam_size=3):
        self.beam_width = beam_size  # Width of the beam
        self.open_list = []  # Nodes to be explored
        heapq.heapify(self.open_list)
        self.paths = []  # Probable paths

    def search(self, model, outputs, dec_hiddens, cells, predicted):
        '''
          Performs beam search to decode the sequences.

          Parameters:
              model: The trained model for decoding.
              outputs: The initial outputs from the model.
              dec_hiddens: Initial hidden states from the model.
              cells: Initial cell states from the model (if using LSTM).
              predicted: Tensor to store the final predicted sequences.
        '''
        batch_size = outputs.shape[1]

        for i in range(batch_size):
            with torch.no_grad():
                model.eval()
                output = outputs[:, i:i+1].contiguous()
                dec_hidden = dec_hiddens[:, i:i+1, :].contiguous()
                cell = cells[:, i:i+1, :].contiguous() if cells is not None else None

                # Initialize the root node
                root_node = self._create_root_node(output, dec_hidden, cell)
                heapq.heappush(self.open_list, root_node)

                # Perform beam search
                while self.open_list:
                    curr_node = heapq.heappop(self.open_list)
                    if curr_node.length == model.output_seq_length - 1:
                        self.paths.append(curr_node)
                        continue

                    self._expand_node(curr_node, model)

                self._finalize_path(i, model, outputs, dec_hiddens, cells, predicted)

    def _create_root_node(self, output, dec_hidden, cell):
        index = output.contiguous()
        root_node = SearchNode(1, 1, index, dec_hidden, cell, None)
        return root_node

    def _expand_node(self, curr_node, model):
        output, dec_hidden, cell, _ = model.decoder.forward(
            curr_node.index, curr_node.hidden, curr_node.cell, None)
        output = model.softmax(output, dim=2)

        # Get top k predictions
        topk_probs, topk_indices = torch.topk(output, self.beam_width, dim=2)

        for j in range(self.beam_width):
            prob = topk_probs[:, :, j]
            idx = topk_indices[:, :, j]

            # Skip nodes with very low probabilities
            if curr_node.path_probability * prob.item() < 0.001:
                continue

            new_node = self._create_new_node(curr_node, prob, idx, dec_hidden, cell)
            heapq.heappush(self.open_list, new_node)

        # Retain only the top k nodes
        self.open_list = heapq.nsmallest(self.beam_width, self.open_list)

    def _create_new_node(self, curr_node, prob, idx, dec_hidden, cell):
        new_node = SearchNode(prob.item(), curr_node.path_probability * prob.item(), idx, dec_hidden, cell, curr_node)
        new_node.length = curr_node.length + 1
        return new_node

    def _finalize_path(self, batch_index, model, outputs, dec_hiddens, cells, predicted):
        if self.paths:
            best_path = min(self.paths)
            self.paths = []

            # Reverse the path to correct the order
            best_path = self._reverse_path(best_path)

            # Traverse the best path and update predictions
            self._traverse_and_update_path(best_path, model, predicted, batch_index)
        else:
            # Fallback to greedy decoding if no paths are found
            self._fallback_greedy_decode(model, outputs, dec_hiddens, cells, predicted, batch_index)

    def _reverse_path(self, path):
        prev = None
        current = path
        while current is not None:
            next_node = current.parent
            current.parent = prev
            prev = current
            current = next_node
        return prev

    def _traverse_and_update_path(self, path, model, predicted, batch_index):
        for t in range(1, model.output_seq_length):
            output, dec_hidden, cell, _ = model.decoder.forward(path.index, path.hidden, path.cell, None)
            predicted[t, batch_index:batch_index+1] = output
            path = path.parent

    def _fallback_greedy_decode(self, model, outputs, dec_hiddens, cells, predicted, batch_index):
        output = outputs[:, batch_index:batch_index+1].contiguous()
        index = output.contiguous()
        dec_hidden = dec_hiddens[:, batch_index:batch_index+1, :].contiguous()
        cell = cells[:, batch_index:batch_index+1, :].contiguous() if cells is not None else None
        for t in range(1, model.output_seq_length):
            output, dec_hidden, cell, _ = model.decoder.forward(index, dec_hidden, cell, None)
            predicted[t, batch_index:batch_index+1] = output
            output = model.softmax(output, dim=2)
            index = torch.argmax(output, dim=2)
