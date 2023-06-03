import torch.nn.functional as F

# Define the knowledge distillation loss
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5, beta=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def calculate_channel_attention(self, filters):
        attention = F.adaptive_avg_pool2d(filters, (1, 1))
        attention = attention.view(attention.size(0), -1)
        attention = F.softmax(attention, dim=1)
        return attention

    def filter_loss(self, student_model, teacher_model):
        student_params = student_model.parameters()
        teacher_params = teacher_model.parameters()
        filter_loss = 0.0

        for student_param, teacher_param in zip(student_params, teacher_params):
            if student_param.dim() > 1:
                student_filters = student_param.view(student_param.size(0), -1)
                teacher_filters = teacher_param.view(teacher_param.size(0), -1)
                student_attention = self.calculate_channel_attention(student_filters)
                teacher_attention = self.calculate_channel_attention(teacher_filters)
                filter_loss += F.mse_loss(student_attention, teacher_attention)

        return filter_loss

    def forward(self, student_outputs, teacher_outputs, labels, student_model, teacher_model):
        soft_student_outputs = F.softmax(student_outputs / self.temperature, dim=1)
        soft_teacher_outputs = F.softmax(teacher_outputs / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_student_outputs.log(), soft_teacher_outputs, reduction='batchmean')
        classification_loss = F.cross_entropy(student_outputs, labels)
        filter_loss = self.filter_loss(student_model, teacher_model)
        loss = (1 - self.alpha) * distillation_loss + self.alpha * ((1 - self.beta) * classification_loss + self.beta * filter_loss)
        return loss
