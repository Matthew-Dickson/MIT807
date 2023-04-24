import torch
import torch.nn as nn


# def vanillia_knowledge_distillation_loss(distillation_loss,student_loss, options) -> torch.float32:
#      alpha =  options.get("alpha") if options.get("alpha") != None else 0.1         
#      return alpha * distillation_loss  + (1-alpha) * student_loss

def distillation_loss(soft_predictions, soft_targets, distill_loss_function = nn.CrossEntropyLoss()) -> torch.float32: 
    return distill_loss_function(input=soft_predictions, target=soft_targets)

def student_loss(predictions, hard_target, student_loss_function = nn.CrossEntropyLoss()) -> torch.float32: 
    return student_loss_function(input=predictions, target=hard_target)

# def knowledge_distillation_loss(student_loss, distillation_loss, options ={"alpha" : 0.1},
#                                  loss_function=vanillia_knowledge_distillation_loss) -> torch.float32:
#     return loss_function(distillation_loss,student_loss, options)


def vanillia_knowledge_distillation_loss(soft_targets,
                         probs_with_temperature,
                         student_probs,
                         labels,
                         distill_loss_function,
                         student_loss_function,
                         options) -> torch.float32:
     
     student_loss_value = student_loss(predictions = student_probs,hard_target = labels, student_loss_function = student_loss_function)
     distill_loss_value = distillation_loss(probs_with_temperature, soft_targets, distill_loss_function)

     alpha =  options.get("alpha") if options.get("alpha") != None else 0.1         
     return alpha * distill_loss_value  + (1-alpha) * student_loss_value


def knowledge_distillation_loss(soft_targets,
                         probs_with_temperature,
                         student_probs,
                         labels,
                         options,
                         distill_loss_function=nn.CrossEntropyLoss(),
                         student_loss_function=nn.CrossEntropyLoss(),
                         loss_implementation=vanillia_knowledge_distillation_loss) -> torch.float32:
    
    return loss_implementation(soft_targets=soft_targets,
                         probs_with_temperature=probs_with_temperature,
                         student_probs=student_probs,
                         labels=labels,
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

    distill_loss = distillation_loss(soft_predictions,soft_labels)
    student_loss = student_loss(hard_predictions,hard_labels)
    print('distill loss: {} distill loss type: {}'.format(distill_loss, distill_loss.dtype))
    print('student loss: {} student loss type: {}'.format(student_loss, student_loss.dtype))

    # loss_one = knowledge_distillation_loss(student_loss, distill_loss)
    # loss_two = knowledge_distillation_loss(student_loss, distill_loss, loss_function=vanillia_knowledge_distillation_loss, options={"alpha": 1})

    # print('loss_one: {} loss_one type: {}'.format(loss_one, loss_one.dtype))
    # print('loss_two: {} loss_two type: {}'.format(loss_two, loss_two.dtype))
