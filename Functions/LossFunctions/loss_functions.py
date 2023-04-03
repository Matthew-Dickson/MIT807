import torch
import torch.nn as nn

def vanillia_knowledge_distillation_loss(student_loss, distillation_loss, options) -> torch.float32:
     alpha =  options.get("alpha") if options.get("alpha") != None else 0.1         
     return alpha * distillation_loss  + (1-alpha) * student_loss

def distillation_loss(soft_predictions, soft_labels, loss_function = nn.CrossEntropyLoss()) -> torch.float32: 
    return loss_function(input=soft_predictions, target=soft_labels)

def student_loss(hard_predictions, hard_labels, loss_function = nn.CrossEntropyLoss()) -> torch.float32: 
    return loss_function(input=hard_predictions, target=hard_labels)

def knowledge_distillation_loss(student_loss, distillation_loss, options ={"alpha" : 0.1},
                                 loss_function=vanillia_knowledge_distillation_loss) -> torch.float32:
    return loss_function(student_loss, distillation_loss, options)
    




if __name__ == '__main__':
    soft_predictions = torch.rand(3, 5)
    soft_labels = torch.empty(3, dtype = torch.long).random_(5)

    hard_predictions = torch.rand(3, 5)
    hard_labels = torch.empty(3, dtype = torch.long).random_(5)

    print('soft input: {} soft input type {} soft target: {} soft target type: {}'.format(soft_predictions, soft_predictions.dtype, soft_labels, soft_labels.dtype))
    print('hard input: {} hard input type {} hard target: {} hard target type: {}'.format(hard_predictions, hard_predictions.dtype, hard_labels, hard_labels.dtype))

    distill_loss = distillation_loss(soft_predictions,soft_labels)
    student_loss = distillation_loss(hard_predictions,hard_labels)
    print('distill loss: {} distill loss type: {}'.format(distill_loss, distill_loss.dtype))
    print('student loss: {} student loss type: {}'.format(student_loss, student_loss.dtype))

    loss_one = knowledge_distillation_loss(student_loss, distill_loss)
    loss_two = knowledge_distillation_loss(student_loss, distill_loss, loss_function=vanillia_knowledge_distillation_loss, options={"alpha": 1})

    print('loss_one: {} loss_one type: {}'.format(loss_one, loss_one.dtype))
    print('loss_two: {} loss_two type: {}'.format(loss_two, loss_two.dtype))
