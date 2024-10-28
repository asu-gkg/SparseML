import torch.nn as nn
from torch.nn.utils import prune
import torch

class Pruner():
    def __init__(self, amount):
        self.amount = amount
    
    def prune_model(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=self.amount)
                # prune.remove(module, 'weight')
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.amount)
                # prune.remove(module, 'weight')
            # if isinstance(module, nn.BatchNorm2d):
            #     prune.l1_unstructured(module, name='weight', amount=self.amount)
            #     prune.remove(module, 'weight')
            
    