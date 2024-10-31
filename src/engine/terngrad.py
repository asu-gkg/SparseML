
import torch

class TernGradCompressor():
    def compress(self, tensor):
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()
        
        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)
    

class PruningAwareTernGradCompressor():
    def __init__(self):
        self.mask_map = {}
        
    def compress(self, bucket):
        tensor = bucket.buffer()
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()
        
        # 直接使用梯度的符号，不进行额外的随机置零
        sign_gradient = gradient.sign() * scalar
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def record_zeros_positions(self, bucket, decompressed_tensor):
        index = bucket.index()
        if index not in self.mask_map:
            self.mask_map[index] = (decompressed_tensor == 0).type(torch.bool)
        else:
            previous_mask = self.mask_map[index]
            current_mask = (decompressed_tensor == 0).type(torch.bool)
            if previous_mask.size() != current_mask.size():
                print(f'previous_mask.size: {previous_mask.size()}, current_mask.size: {current_mask.size()}')
                self.mask_map[index] = current_mask
                return 
            self.mask_map[index] = previous_mask & current_mask
            
            updates = (previous_mask != current_mask) & (current_mask == False)
            num_updates = updates.sum().item()
            sparsity = (self.mask_map[index] == True).sum().item()
            print(bucket.index(), bucket.buffer().size(), num_updates, sparsity)

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)