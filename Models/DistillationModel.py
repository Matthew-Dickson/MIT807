import torch
from Functions.ActivationFunctions.activation_functions import batch_softmax_with_temperature
from Functions.LossFunctions.loss_functions import vanillia_knowledge_distillation_loss
from Models.BaseModel import BaseModel


class Distillation(BaseModel):
    
    def train_epoch(self,
                    dataloader,
                    loss_fn,
                    optimizer,
                    teacher,
                    options,
                    distill_loss_function,
                    student_loss_function,
                    temperature,
                    loss_implementation=vanillia_knowledge_distillation_loss,
                    device="cpu"):
        train_loss,train_correct=0.0,0
        self.train()
        for images, labels in dataloader:

            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            logits = self(images)

            soft_probabilities = batch_softmax_with_temperature(logits, temperature)
            soft_targets = teacher.generate_soft_targets(images = images,
                                                         temperature = temperature)
        
            loss=loss_fn(soft_targets = soft_targets,
                         soft_probabilities = soft_probabilities,
                         logits = logits,
                         labels = labels,
                         options = options,
                         distill_loss_function=distill_loss_function,
                         student_loss_function=student_loss_function,
                         loss_implementation=loss_implementation)
            
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predictions = torch.max(logits.data, 1)
            train_correct += (predictions == labels).sum().item()

        return train_loss,train_correct
  
    def valid_epoch(self,
                    dataloader,
                    loss_fn,
                    teacher,
                    options,
                    distill_loss_function,
                    student_loss_function,
                    temperature,
                    loss_implementation=vanillia_knowledge_distillation_loss,
                    device="cpu"):
        valid_loss, val_correct = 0.0, 0
        self.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images,labels = images.to(device),labels.to(device)
                logits = self(images)
            
                student_probs = batch_softmax_with_temperature(logits, 1)
                soft_probabilities = batch_softmax_with_temperature(logits, temperature)
                soft_targets = teacher.generate_soft_targets(images = images,
                                                         temperature = temperature)

                loss=loss_fn(soft_targets = soft_targets,
                         soft_probabilities = soft_probabilities,
                         logits = student_probs,
                         labels = labels,
                         options = options,
                         distill_loss_function=distill_loss_function,
                         student_loss_function=student_loss_function,
                         loss_implementation=loss_implementation)
                
                valid_loss+=loss.item()*images.size(0)
                _, predictions = torch.max(logits.data,1)
                val_correct+=(predictions == labels).sum().item()

        return valid_loss,val_correct
    

    
    