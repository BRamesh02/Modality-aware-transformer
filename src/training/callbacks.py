import torch
import numpy as np
import os

class EarlyStopping:
    """
    Early stops the training if validation IC (Information Coefficient) doesn't improve after a given patience.
    Saves the best model state (Highest IC) to a file.
    """
    def __init__(self, patience: int, min_delta=0.0, path='checkpoint.pt', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation IC improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_ic = -np.inf
        self.early_stop = False

    def __call__(self, val_ic, model):
        """
        Args:
            val_ic (float): The current epoch's validation Information Coefficient.
            model (torch.nn.Module): The model instance to save.
        """
        score = val_ic

        if self.best_ic == -np.inf:
            self.best_ic = score
            self.save_checkpoint(val_ic, model)
        
        elif score < self.best_ic + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.save_checkpoint(val_ic, model)
            self.best_ic = score
            self.counter = 0

    def save_checkpoint(self, val_ic, model):
        '''Saves model when validation IC increases.'''
        if self.verbose:
            print(f'   Validation IC increased ({self.best_ic:.4f} --> {val_ic:.4f}). Saving model...')
        
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        torch.save(model.state_dict(), self.path)