import torch
from torch import nn
import torch.nn.functional as F


def accuracy(predicted, actual):
    _, predictions = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predictions == actual).item() / len(predictions))

def accuracy_vanilla_kd(predicted, actual):
    _, predictions = torch.max(predicted, dim=1)
    _, actual_grouped = torch.max(actual, dim=1)
    return torch.tensor(torch.sum(predictions == actual_grouped).item() / len(predictions))

def rescale(t, bottom=0, top=1):
    t_min, t_max = t.min(), t.max()
    new_min, new_max = bottom, top
    t = (t - t_min) / (t_max - t_min) * (new_max - new_min) + new_min
    return t

class BaseModel(nn.Module):

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
    
    def save(self,path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(self.load(path))
    

    # def training_step(self, batch):
    #     images, labels = batch
    #     out = self(images)
    #     loss = F.cross_entropy(out, labels)
    #     acc = accuracy(out, labels)
    #     return loss, acc

    # def training_step_vanilla_kd(self, batch, teacher_model):

    #     images, labels = batch

    #     teacher_out = teacher_model(images)
    #     teacher_out = rescale(teacher_out)

    #     out = self(images)

    #     loss = F.cross_entropy(out, teacher_out)
    #     acc = accuracy_vanilla_kd(out, teacher_out)
    #     return loss, acc

    # def validation_step(self, batch):
    #     images, labels = batch
    #     out = self(images)
    #     loss = F.cross_entropy(out, labels)
    #     acc = accuracy(out, labels)
    #     return {"val_loss": loss.detach(), "val_acc": acc}

    # def validation_epoch_end(self, outputs):
    #     batch_losses = [loss["val_loss"] for loss in outputs]
    #     loss = torch.stack(batch_losses).mean()
    #     batch_accuracy = [accuracy["val_acc"] for accuracy in outputs]
    #     acc = torch.stack(batch_accuracy).mean()
    #     return {"val_loss": loss.item(), "val_acc": acc.item()}

    # def epoch_end(self, epoch, result):
    #     print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, train_acc: {:.4f} , val_loss: {:.4f}, val_acc: {:.4f}".format(
    #         epoch, result['lrs'][-1], result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))