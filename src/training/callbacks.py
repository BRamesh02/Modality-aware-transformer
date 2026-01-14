import torch
import numpy as np
import os

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state to a file.
    """
    def __init__(self, patience=5, min_delta=0.0, path='checkpoint.pt', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
            '''Saves model when validation loss decreases.'''
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
                
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
