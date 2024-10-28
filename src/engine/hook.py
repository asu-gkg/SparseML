
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch

def sparsity(tensor):
    total_nums = tensor.numel()
    zero_nums = torch.sum(tensor == 0).item()
    return 100 * zero_nums / total_nums
    
def allreduce_hook(state, bucket):
    return default_hooks.allreduce_hook(process_group=state, bucket=bucket)