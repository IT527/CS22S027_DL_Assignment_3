import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from tqdm import tqdm #To keep track of progress loops
from accuracy import *

"""#Execute Training"""

def train_one_epoch(model, data_loader, optimizer, criterion, epoch, epochs, device):
    '''
      Train the model for one epoch.
    '''
    model.train()
    total_loss = 0

    for source, target in data_loader:  #Iterate over the data loader
        source, target = source.to(device), target.to(device)
        optimizer.zero_grad()
        #Forward pass
        output, _ = model(source, target, epoch < epochs * 0.3, False)
        output = output.permute(1, 0, 2).reshape(-1, 67)
        expected = F.one_hot(target, num_classes=67).float().reshape(-1, 67)

        loss = criterion(output, expected)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1) #Clip gradients to prevent exploding gradients
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, criterion, device, beam=False):
    '''
      Evaluate the model.
    '''
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs, None, False, beam)

            outputs = outputs.permute(1, 0, 2).reshape(-1, 67)
            expected = F.one_hot(targets, num_classes=67).float().reshape(-1, 67)

            loss = criterion(outputs, expected) #Compute loss
            total_loss += loss.item()

            acc_outputs = torch.argmax(F.softmax(outputs, dim=1), dim=1).view_as(targets) #Compute accuracy
            total_accuracy += accuracy_calc(acc_outputs, targets)

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader.dataset)

    return avg_loss, avg_accuracy

def train_and_evaluate_model(model, train_loader, val_loader, epochs, beam, device):
    """
      Train and evaluate the model.

      Args:
          model: The model to be trained and evaluated.
          train_loader: DataLoader for the training data.
          val_loader: DataLoader for the validation data.
          epochs: Total number of epochs.
          beam: Boolean flag to indicate whether to use beam search or not.
          device: Device to run the training and evaluation on (CPU or GPU).

      Returns:
          train_loss_list: List of training losses for each epoch.
          val_loss_list: List of validation losses for each epoch.
          train_accuracy_list: List of training accuracies for each epoch.
          val_accuracy_list: List of validation accuracies for each epoch.
    """
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    train_loss_list, val_loss_list = [], []
    train_accuracy_list, val_accuracy_list = [], []

    for epoch in tqdm(range(epochs)):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, epochs, device)

        train_loss_eval, train_accuracy = evaluate(model, train_loader, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, beam)
        #Print the metrics
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss_eval}, Train Accuracy: {train_accuracy}')
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
        #Store the metrics
        train_loss_list.append(train_loss_eval)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        #Log into wandb
        wandb.log({'val_accuracy': val_accuracy,
                  'val_loss': val_loss,
                  'train_accuracy': train_accuracy,
                  'train_loss': train_loss_eval
                  })

    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list
