

import torch
import torch.distributed as dist

def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices


def desparsify(values, indices, numel):
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor():
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio

    def compress(self, tensor):
        values, indices = sparsify(tensor, self.compress_ratio)
        
        ctx = tensor.numel(), tensor.size()
        return values, indices, ctx

    def decompress(self, values, indices, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(values, indices, numel)
        return tensor_decompressed.view(shape)
