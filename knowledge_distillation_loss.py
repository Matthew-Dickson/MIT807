import torch
import torch.nn as nn

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        self.classification_criterion = nn.CrossEntropyLoss()

    def calculate_filter_loss(self, student_model, teacher_model):
        student_params = student_model.state_dict()
        teacher_params = teacher_model.state_dict()

        filter_loss = 0.0
        count = 0

        for s_key, t_key in zip(student_params.keys(), teacher_params.keys()):
            if "features" in s_key:  # Filter difference in convolutional layers
                student_weights = student_params[s_key]
                teacher_weights = teacher_params[t_key]
                filter_loss += torch.norm(student_weights - teacher_weights, p=2)
                count += 1

        if count > 0:
            filter_loss /= count

        return filter_loss

    def forward(self, student_outputs, teacher_outputs, labels, student_model, teacher_model):
        distillation_loss = self.distillation_criterion(student_outputs, teacher_outputs) * (1 - self.alpha)
        classification_loss = self.classification_criterion(student_outputs, labels) * self.alpha
        filter_loss = self.calculate_filter_loss(student_model, teacher_model) 
        loss = (1 - self.beta)*(distillation_loss + classification_loss)  + (filter_loss * self.beta)
        return loss