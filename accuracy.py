import torch

"""#Accuracy"""

def accuracy_calc(pred_seq , true_seq):
    """
      Calculate the accuracy of predicted sequences against the true sequences.

      This function compares the predicted sequences with the true sequences and
      calculates how many sequences are completely correct.

      Args:
          pred_seq (torch.Tensor): A tensor containing the predicted sequences.
          true_seq (torch.Tensor): A tensor containing the true sequences.

      Returns:
          int: The number of completely correct sequences.
    """
    num_sample,seq_len = true_seq.shape #Get the number of samples and sequence length from the true sequences
    score = torch.sum(torch.sum(pred_seq == true_seq,axis = 1) == seq_len) #Check if the number of correct elements in each sequence equals the sequence length and then add.
    return score
