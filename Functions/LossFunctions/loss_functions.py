import torch
import numpy as np
 
def distillation_loss(soft_probabilities, soft_targets, distill_loss_function) -> torch.float32: 
    return distill_loss_function(input=soft_probabilities, target=soft_targets)

def student_loss(logits, hard_target, student_loss_function) -> torch.float32: 
    return student_loss_function(input=logits, target=hard_target)


def vanillia_knowledge_distillation_loss(soft_targets,
                         soft_probabilities,
                         logits,
                         labels,
                         distill_loss_function,
                         student_loss_function,
                         options) -> torch.float32:
     
     
     distill_loss_value = distillation_loss(soft_probabilities = soft_probabilities,
                                            soft_targets = soft_targets,
                                            distill_loss_function =  distill_loss_function)
     student_loss_value = student_loss(logits = logits,
                                       hard_target = labels,
                                      student_loss_function = student_loss_function)

     alpha =  options.get("alpha") if options.get("alpha") != None else 0.1         
     return alpha * distill_loss_value  + (1-alpha) * student_loss_value


def filter_loss(teacher_kernels, student_kernels):
    flatten_teacher_kernels = torch.flatten(teacher_kernels)
    flatten_student_kernels = torch.flatten(student_kernels)
     
    distance = torch.sqrt(torch.sum(torch.square(flatten_teacher_kernels)) - torch.sum(torch.square(flatten_student_kernels)))
    #try different distance metrics
    #distance = np.linalg.norm(flatten_teacher_kernels - flatten_student_kernels)
    return distance

def filter_knowledge_distillation_loss(soft_targets,
                         soft_probabilities,
                         logits,
                         labels,
                         teacher_kernels,
                         student_kernels,
                         distill_loss_function,
                         student_loss_function,
                         options) -> torch.float32:
     
     
     distill_loss_value = distillation_loss(soft_probabilities = soft_probabilities,
                                            soft_targets = soft_targets,
                                            distill_loss_function =  distill_loss_function)
     student_loss_value = student_loss(logits = logits,
                                       hard_target = labels,
                                      student_loss_function = student_loss_function)
     
     filter_loss_value = filter_loss(teacher_kernels, student_kernels)
     alpha =  options.get("alpha") if options.get("alpha") != None else 0.1    
     beta =  options.get("beta") if options.get("beta") != None else 1      
     return (alpha * distill_loss_value  + (1-alpha) * student_loss_value) + (beta)* filter_loss_value


def knowledge_distillation_loss(soft_targets,
                         soft_probabilities,
                         logits,
                         labels,
                         teacher_kernels,
                         student_kernels,
                         options,
                         distill_loss_function,
                         student_loss_function,
                         loss_implementation=vanillia_knowledge_distillation_loss) -> torch.float32:
    
    return loss_implementation(soft_targets=soft_targets,
                         soft_probabilities=soft_probabilities,
                         logits=logits,
                         labels=labels,
                         teacher_kernels=teacher_kernels,
                         student_kernels=student_kernels,
                         distill_loss_function=distill_loss_function,
                         student_loss_function=student_loss_function,
                         options=options)
    

if __name__ == '__main__':
    soft_predictions = torch.rand(3, 5)
    soft_labels = torch.empty(3, dtype = torch.long).random_(5)

    hard_predictions = torch.rand(3, 5)
    hard_labels = torch.empty(3, dtype = torch.long).random_(5)

    print('soft input: {} soft input type {} soft target: {} soft target type: {}'.format(soft_predictions, soft_predictions.dtype, soft_labels, soft_labels.dtype))
    print('hard input: {} hard input type {} hard target: {} hard target type: {}'.format(hard_predictions, hard_predictions.dtype, hard_labels, hard_labels.dtype))

    # distill_loss = distillation_loss(nn.Softmax()(soft_predictions),soft_labels, nn.KLDivLoss(reduction="batchmean"))
    # student_loss = student_loss(hard_predictions,hard_labels,nn.CrossEntropyLoss())
    # print('distill loss: {} distill loss type: {}'.format(distill_loss, distill_loss.dtype))
    # print('student loss: {} student loss type: {}'.format(student_loss, student_loss.dtype))


    filter_loss_value = filter_loss(torch.tensor([[[1,2,3,4],[1,3,4,5]]]),torch.tensor([[2,2,2],[3,3,3],[3,3,3],[4,4,4]]))
    print('student loss: {} student loss type: {}'.format(filter_loss_value, filter_loss_value.dtype))