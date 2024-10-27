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
                prune.remove(module, 'weight')
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.amount)
                prune.remove(module, 'weight')
            # if isinstance(module, nn.BatchNorm2d):
            #     prune.l1_unstructured(module, name='weight', amount=self.amount)
            #     prune.remove(module, 'weight')
            
    def model_sparsity(self, model):
        total_params = 0     
        zero_params = 0    
        for param in model.parameters():
            total_params += param.numel()   
            zero_params += torch.sum(param == 0).item()

        sparsity = 100.0 * zero_params / total_params 
        return sparsity