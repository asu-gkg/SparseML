import torch.nn as nn
from torch.nn.utils import prune
import torch

class Pruner():
    def __init__(self, amount):
        self.amount = amount
        self.pruned = False
    
    def prune_model(self, model):
        self.pruned = True
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=self.amount)
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.amount)
    
    def recover(self, model):
        if self.pruned==False:
            return 
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.remove(module, name='weight')
            if isinstance(module, nn.Linear):
                prune.remove(module, name='weight')
            self.pruned = False