import torch
from tqdm import tqdm
from engine import pruner
from engine import hook

class Trainer:
    def __init__(self, model, optimizer, criterion, lr_scheduler, device, train_dataloader, val_dataloader, 
                pruner:pruner.Pruner, comm_hook):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.pruner = pruner
        self.hook = comm_hook
        
        if self.hook=='default':
            self.model.register_comm_hook(None, hook.allreduce_hook)

    def train_step(self, inputs, labels):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_sparsity = self.grad_sparsity()
        self.optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        correct_predictions = torch.sum(preds == labels).item()
        return loss.item(), correct_predictions, labels.size(0), grad_sparsity

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        self.train_dataloader.sampler.set_epoch(epoch)
        
        if self.pruner != None and epoch==1:
            self.pruner.prune_model(self.model)
                    
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
        for step, (inputs, labels) in enumerate(progress_bar):
            loss, correct_preds, batch_size, grad_sparsity = self.train_step(inputs, labels)
            total_loss += loss
            correct_predictions += correct_preds
            total_samples += batch_size

            avg_loss = total_loss / (step + 1)
            accuracy = correct_predictions / total_samples
            weights_sparsity = self.model_sparsity()
            
            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy, weights_sparsity=weights_sparsity, grad_sparsity=grad_sparsity)
        
        self.lr_scheduler.step()
        
        return self.evaluate()
    
    def evaluate(self):
        val_dataloader = self.val_dataloader
        
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc="Validating")
            for step, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels).item()
                total_samples += labels.size(0)
                
                avg_loss = total_loss / (step + 1)
                accuracy = correct_predictions / total_samples
                progress_bar.set_postfix(val_loss=avg_loss, val_accuracy=accuracy)
        
        return avg_loss, accuracy
    
    def model_sparsity(self):
        total_params = 0     
        zero_params = 0    
        for param in self.model.parameters():
            total_params += param.numel()   
            zero_params += torch.sum(param == 0).item()

        sparsity = 100.0 * zero_params / total_params 
        return sparsity
    
    def grad_sparsity(self):
        total_grads = 0
        zero_grads = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                total_grads += grad_data.numel()
                zero_grads += torch.sum(grad_data == 0).item()
        return 100.0 * zero_grads / total_grads
