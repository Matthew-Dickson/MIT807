import torch
from torch import nn
from Functions.ActivationFunctions.activation_functions import batch_softmax_with_temperature

class BaseModel(nn.Module):

    def predict(self,dataloader,device="cpu"):
        self.eval()
        with torch.no_grad():
            correct = 0
            for images, labels in dataloader:
                images,labels = images.to(device),labels.to(device)
                logits = self(images)
                _,predictions = torch.max(logits,1)
                correct+=(predictions == labels).sum().item()

        return predictions, correct
    
    def generate_soft_targets(self,images,temperature = 40):
        self.eval()
        with torch.no_grad():
            logits = self(images)
            probs_with_temperature = batch_softmax_with_temperature(logits, temperature)
        return probs_with_temperature
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    