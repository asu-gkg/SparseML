
import torch.distributed
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.distributed import all_reduce, all_gather, ReduceOp
import torch
from engine.terngrad import TernGradCompressor, PruningAwareTernGradCompressor
from engine.topk import TopKCompressor
from engine.threshold import ThresholdCompressor
import torch.distributed as dist

terngrad_compressor = TernGradCompressor()
topk_compressor = None
threshold_compressor = None
pruning_aware_terngrad_compressor = PruningAwareTernGradCompressor()

def sparsity(tensor):
    total_nums = tensor.numel()
    zero_nums = torch.sum(tensor == 0).item()
    return 100 * zero_nums / total_nums
    
def allreduce_hook(state, bucket):
    return default_hooks.allreduce_hook(process_group=state, bucket=bucket)

def gather(values, indices):
        gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
        gathered_indices = [torch.zeros_like(indices) for _ in range(dist.get_world_size())]

        all_gather(gathered_values, values)
        all_gather(gathered_indices, indices)

        combined_values = torch.cat(gathered_values)
        combined_indices = torch.cat(gathered_indices)
        return combined_values, combined_indices

def terngrad_hook(state, bucket):
    tensor = bucket.buffer()
    tensor_compressed, numel = terngrad_compressor.compress(tensor)
    
    compressed_data, scalar = tensor_compressed
    all_reduce(compressed_data, op=ReduceOp.SUM)
    all_reduce(scalar, op=ReduceOp.AVG)
    
    decompressed_tensor = terngrad_compressor.decompress((compressed_data, scalar), numel)
        
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut


def pruning_aware_terngrad_hook(state, bucket):
    tensor_compressed, numel = pruning_aware_terngrad_compressor.compress(bucket)
    
    compressed_data, scalar = tensor_compressed
    all_reduce(compressed_data, op=ReduceOp.SUM)
    all_reduce(scalar, op=ReduceOp.AVG)
    
    decompressed_tensor = pruning_aware_terngrad_compressor.decompress((compressed_data, scalar), numel)
    pruning_aware_terngrad_compressor.record_zeros_positions(bucket, decompressed_tensor)
    fut = torch.futures.Future()
    fut.set_result(decompressed_tensor)
    return fut

def topk_hook(state, bucket):
    tensor = bucket.buffer()
    values, indices, ctx = topk_compressor.compress(tensor=tensor)
    gathered_data = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_data, [values, indices])
    
    decompressed_tensors = []
    for gv, gi in gathered_data:
        decompressed_tensor = topk_compressor.decompress(gv, gi, ctx)
        decompressed_tensors.append(decompressed_tensor)
        
    aggregated_tensor = sum(decompressed_tensors) / dist.get_world_size()
    fut = torch.futures.Future()
    fut.set_result(aggregated_tensor)
    return fut

def topk_hook_wrapper(compression_ratio):
    global topk_compressor
    topk_compressor = TopKCompressor(compress_ratio=compression_ratio)
    return topk_hook

def threshold_hook(state, bucket):
    tensor = bucket.buffer()
    compressed_tensor, ctx = threshold_compressor.compress(tensor)
    values, indices = compressed_tensor
    gathered_data = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_data, [values, indices])
    
    decompressed_tensors = []
    for gv, gi in gathered_data:
        decompressed_tensor = threshold_compressor.decompress([gv, gi], ctx)
        decompressed_tensors.append(decompressed_tensor)
        
    aggregated_tensor = sum(decompressed_tensors) / dist.get_world_size()
    fut = torch.futures.Future()
    fut.set_result(aggregated_tensor)
    return fut

def threshold_hook_wrapper(threshold):
    global threshold_compressor
    threshold_compressor = ThresholdCompressor(threshold=threshold)
    return threshold_hook