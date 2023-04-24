import torch
from torch import nn
import torch.nn.functional as F

from Functions.ActivationFunctions.activation_functions import batch_softmax_with_temperature
from Models.BaseModel import BaseModel


class Model(BaseModel):

    def train_epoch(self,dataloader,loss_fn,optimizer,device="cpu"):
        train_loss,train_correct=0.0,0
        self.train()
        for images, labels in dataloader:

            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = self(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        return train_loss,train_correct


    def valid_epoch(self,dataloader,loss_fn,device="cpu"):
        valid_loss, val_correct = 0.0, 0
        self.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images,labels = images.to(device),labels.to(device)
                output = self(images)
                loss=loss_fn(output,labels)
                valid_loss+=loss.item()*images.size(0)
                scores, predictions = torch.max(output.data,1)
                val_correct+=(predictions == labels).sum().item()

        return valid_loss,val_correct
    