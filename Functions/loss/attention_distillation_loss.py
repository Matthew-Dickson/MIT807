import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax

# Define the knowledge distillation loss
class AttentionDistillationLoss(nn.Module):
    def __init__(self,
                 options,
                 distillation_criterion = nn.KLDivLoss(reduction='batchmean'),
                 student_criterion = nn.CrossEntropyLoss()):
        super(AttentionDistillationLoss, self).__init__()
        self.alpha =  options.get("alpha") if options.get("alpha") != None else 0.1    
        self.beta =  options.get("beta") if options.get("beta") != None else 1     
        self.temperature =  options.get("temperature") if options.get("temperature") != None else 20 

        self.distillation_criterion = distillation_criterion
        self.student_criterion= student_criterion

    def _get_kernels(self,model):
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                kernels = layer.weight.data.clone()
        return kernels
    
    def _batch_softmax_with_temperature(self, batch_logits, temperature) -> torch.tensor:
        return softmax(batch_logits/temperature,dim=1)

    def _calculate_channel_attention(self, filters):
        attention = F.adaptive_avg_pool2d(filters, (1, 1))
        attention = attention.view(attention.size(0), -1)
        attention = F.softmax(attention, dim=1)
        return attention

    def _calculate_filter_loss(self, student_model, teacher_model):
        student_kernels = self._get_kernels(student_model)
        teacher_kernels = self._get_kernels(teacher_model)

        flatten_student_kernels = torch.flatten(student_kernels)
        flatten_teacher_kernels = torch.flatten(teacher_kernels)
    
        student_attention = self.calculate_channel_attention(flatten_student_kernels)
        teacher_attention = self.calculate_channel_attention(flatten_teacher_kernels)
        filter_loss = F.mse_loss(student_attention, teacher_attention)

        return filter_loss

    def forward(self, student_logits, features, labels, student_model, teacher_model):
        soft_probabilities = self._batch_softmax_with_temperature(batch_logits=student_logits, temperature=self.temperature)
        soft_targets = teacher_model.generate_soft_targets(images = features,
                                                                temperature = self.temperature)
        distillation_loss = self.distillation_criterion(soft_probabilities, soft_targets) 
        student_loss = self.student_criterion(student_logits, labels) 
        filter_loss = self._calculate_filter_loss(student_model, teacher_model) 
        loss = ((1-self.beta)*(self.alpha * distillation_loss  + (1-self.alpha) * student_loss)) + (self.beta* filter_loss)
        return loss
