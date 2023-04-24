import torch
from torch import nn
from Functions.ActivationFunctions.activation_functions import batch_softmax_with_temperature

class BaseModel(nn.Module):

    def predict(self,dataloader,device="cpu"):
        self.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images,labels = images.to(device),labels.to(device)
                logits = self(images)
                _,predictions = torch.max(logits,1)

        return predictions
    
    def generate_soft_targets(self,images,temperature = 40):
        self.eval()
        with torch.no_grad():
            logits = self(images)
            probs_with_temperature = batch_softmax_with_temperature(logits, temperature)
            _ , predictions = torch.max(probs_with_temperature,1)
        return probs_with_temperature
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    


# import torch
# from torch import nn
# import torch.nn.functional as F

# from Functions.ActivationFunctions.activation_functions import batch_softmax_with_temperature



# def vanillia_knowledge_distillation_loss(distillation_loss,student_loss, options) -> torch.float32:
#      alpha =  options.get("alpha") if options.get("alpha") != None else 0.1         
#      return alpha * distillation_loss  + (1-alpha) * student_loss

# def distillation_loss(soft_predictions, soft_targets, distill_loss_function = nn.CrossEntropyLoss()) -> torch.float32: 
#     return distill_loss_function(input=soft_predictions, target=soft_targets)

# def student_loss(predictions, hard_target, student_loss_fn = nn.CrossEntropyLoss()) -> torch.float32: 
#     return student_loss_fn(input=predictions, target=hard_target)

# def knowledge_distillation_loss(student_loss, distillation_loss, options ={"alpha" : 0.1},
#                                  loss_function=vanillia_knowledge_distillation_loss) -> torch.float32:
#     return loss_function(distillation_loss,student_loss, options)

# class BaseModel(nn.Module):

#     def train_epoch(self,dataloader,loss_fn,optimizer,device="cpu"):
#         train_loss,train_correct=0.0,0
#         self.train()
#         for images, labels in dataloader:

#             images,labels = images.to(device),labels.to(device)
#             optimizer.zero_grad()
#             output = self(images)
#             loss = loss_fn(output,labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * images.size(0)
#             scores, predictions = torch.max(output.data, 1)
#             train_correct += (predictions == labels).sum().item()

#         return train_loss,train_correct
    
#     def train_epoch_student(self,
#                             dataloader,
#                             loss_fn,
#                             optimizer,
#                             teacher,
#                             temperature = 40,
#                             device="cpu"):
        
#         train_loss,train_correct=0.0,0
#         self.train()
#         for images, hard_targets in dataloader:

#             images,labels = images.to(device),hard_targets.to(device)
#             optimizer.zero_grad()
#             logits = self(images)
#             probs_with_temperature = batch_softmax_with_temperature(logits, temperature)
#             probs = batch_softmax_with_temperature(logits, 1)

#             soft_targets = teacher.generate_soft_targets(images = images,
#                                                          temperature = temperature)

#             first_loss_term = distillation_loss(probs_with_temperature, soft_targets)
#             second_loss_term = student_loss(probs, labels)

#             loss=knowledge_distillation_loss(first_loss_term,second_loss_term,options ={"alpha" : 0.4})
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * images.size(0)
#             #Incorrect
#             scores, predictions = torch.max(probs, 1)
#             train_correct += (predictions == labels).sum().item()

#         return train_loss,train_correct
  
#     def valid_epoch_student(self,
#                     dataloader,
#                     loss_fn,
#                     teacher,
#                     temperature = 40,
#                     device="cpu"):
#         valid_loss, val_correct = 0.0, 0
#         self.eval()
#         with torch.no_grad():
#             for images, labels in dataloader:
#                 images,labels = images.to(device),labels.to(device)
#                 logits = self(images)
#                 probs_with_temperature = batch_softmax_with_temperature(logits, temperature)
#                 probs = batch_softmax_with_temperature(logits, 1)

#                 soft_targets = teacher.generate_soft_targets(images = images,
#                                                          temperature = temperature)

#                 first_loss_term = distillation_loss(probs_with_temperature, soft_targets)
#                 second_loss_term = student_loss(probs, labels)

#                 loss=knowledge_distillation_loss(first_loss_term,second_loss_term,options ={"alpha" : 0.4})
#                 valid_loss+=loss.item()*images.size(0)
#                 scores, predictions = torch.max(probs,1)
#                 val_correct+=(predictions == labels).sum().item()

#         return valid_loss,val_correct
    

#     def valid_epoch(self,dataloader,loss_fn,device="cpu"):
#         valid_loss, val_correct = 0.0, 0
#         self.eval()
#         with torch.no_grad():
#             for images, labels in dataloader:
#                 images,labels = images.to(device),labels.to(device)
#                 output = self(images)
#                 loss=loss_fn(output,labels)
#                 valid_loss+=loss.item()*images.size(0)
#                 scores, predictions = torch.max(output.data,1)
#                 val_correct+=(predictions == labels).sum().item()

#         return valid_loss,val_correct
    
#     def predict(self,dataloader,device="cpu"):
#         self.eval()
#         with torch.no_grad():
#             for images, labels in dataloader:
#                 images,labels = images.to(device),labels.to(device)
#                 logits = self(images)
#                 _,predictions = torch.max(logits,1)

#         return predictions
    

#     def generate_soft_targets(self,images,temperature = 40):
#         self.eval()
#         with torch.no_grad():
#             logits = self(images)
#             probs_with_temperature = batch_softmax_with_temperature(logits, temperature)
#             _ , predictions = torch.max(probs_with_temperature,1)
#         print(len(probs_with_temperature[0]))
#         print(len(probs_with_temperature))
#         print(probs_with_temperature)
#         return probs_with_temperature
    
#     def save(self, path):
#         torch.save(self.state_dict(), path)

#     def load(self, path):
#         self.load_state_dict(torch.load(path))
    