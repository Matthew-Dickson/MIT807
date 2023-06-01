import torch
import time
from torch.utils.data import DataLoader
from Data.Utilities.device_loader import ToDeviceLoader, to_device
import torch.nn as nn
from torch.nn.functional import softmax

from early_stopper import EarlyStopper



def get_kernels(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            kernels = layer.weight.data.clone()
    return kernels



def batch_softmax_with_temperature(batch_logits, temperature) -> torch.tensor:
    return softmax(batch_logits/temperature,dim=1)

def distillation_loss(soft_probabilities, soft_targets, distill_loss_function) -> torch.float32: 
    return distill_loss_function(input=soft_probabilities, target=soft_targets)

def student_loss(logits, hard_target, student_loss_function) -> torch.float32: 
    return student_loss_function(input=logits, target=hard_target)


def filter_loss(teacher_kernels, student_kernels):
    flatten_teacher_kernels = torch.flatten(teacher_kernels)
    flatten_student_kernels = torch.flatten(student_kernels)
     #look at attention 

    distance = torch.sqrt(torch.sum(torch.square(flatten_teacher_kernels)) - torch.sum(torch.square(flatten_student_kernels)))
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
     return (1-beta)(alpha * distill_loss_value  + (1-alpha) * student_loss_value) + (beta)* filter_loss_value


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




def train(train_dataset,
          valid_dataset,
          student_model,
          teacher_model,
          input_channels,
          output_channels,
          train_options,
          device):
        
        #Define Defaults
        distillation_type =  train_options.get("distillation_type") if train_options.get("distillation_type") != None else "none"
        learning_rate = train_options.get("learning_rate") if train_options.get("learning_rate") != None else 0.01
        batch_size = train_options.get("batch_size") if train_options.get("batch_size") != None else 128
        model_optimizer = train_options.get("optimizer") if train_options.get("optimizer") != None else torch.optim.Adam
        number_of_epochs = train_options.get("number_of_epochs") if train_options.get("number_of_epochs") != None else 9999999999999
        temperature = train_options.get("temperature") if train_options.get("temperature") != None else 20

        distill_loss_function=nn.CrossEntropyLoss()                                                    
        student_loss_function=nn.CrossEntropyLoss()
        loss_function = nn.CrossEntropyLoss()

        if(train_options.get("loss_parameters") == None):
            raise Exception("Requires loss parameters")
        
        loss_parameters = train_options.get("loss_parameters")
        
        model = to_device(student_model(input_channels=input_channels,num_classes=output_channels),device=device)
        optimizer =  model_optimizer(model.parameters(), lr=learning_rate)
       
        train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=False)
        valid_dl = DataLoader(dataset=valid_dataset, batch_size=batch_size,shuffle=False)

        train_data_on_specified_device = ToDeviceLoader(train_dl, device)
        valid_data_on_specified_device = ToDeviceLoader(valid_dl, device)

        early_stopper=EarlyStopper(patience=5,min_delta=0)

       
        if(teacher_model != None):
            #Get kernels
            student_kernels = get_kernels(model)
            teacher_kernels = get_kernels(teacher_model)


        avg_train_loss_per_epochs = []
        avg_train_acc_per_epochs = []
        avg_validation_loss_per_epochs = []
        avg_validation_acc_per_epochs = []
        avg_train_run_times_per_epochs = []
        avg_validation_run_times_per_epochs = []

        for epoch in range(number_of_epochs):
            train_loss,train_correct=0.0,0
            valid_loss, val_correct = 0.0, 0

            t0 = time.time()
            model.train()
            for images, labels in train_data_on_specified_device:
                images,labels = images.to(device),labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                _, predictions = torch.max(logits, 1)

                if(teacher_model != None):
                    soft_probabilities = batch_softmax_with_temperature(logits, temperature)
                    soft_targets = teacher_model.generate_soft_targets(images = images,
                                                                temperature = temperature)
                    

                    
                    if(distillation_type == "filter"):
                        loss = filter_knowledge_distillation_loss(soft_targets = soft_targets,
                                    soft_probabilities = soft_probabilities,
                                    logits = logits,
                                    labels = labels,
                                    teacher_kernels = teacher_kernels,
                                    student_kernels = student_kernels,
                                    options = loss_parameters,
                                    distill_loss_function=distill_loss_function,
                                    student_loss_function=student_loss_function)
                        
                    elif(distillation_type == "traditional"):
                        loss =  vanillia_knowledge_distillation_loss( soft_targets = soft_targets,
                                soft_probabilities = soft_probabilities,
                                logits = logits,
                                labels= labels,
                                distill_loss_function= distill_loss_function,
                                student_loss_function = student_loss_function,
                                options = loss_parameters)
                    else:
                        loss = loss_function(input=logits, target=labels)
                else:
                    loss = loss_function(input=logits, target=labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_correct += (predictions == labels).sum().item()

            t1 = time.time()
            model.eval()
            with torch.no_grad():
                for images, labels in valid_data_on_specified_device:
                    images,labels = images.to(device),labels.to(device)
                    logits = model(images)

                    if(teacher_model != None):
                        soft_probabilities = batch_softmax_with_temperature(logits, temperature)
                        soft_targets = teacher_model.generate_soft_targets(images = images,
                                                                    temperature = temperature)

                        if(distillation_type == "filter"):
                            loss = filter_knowledge_distillation_loss(soft_targets = soft_targets,
                                        soft_probabilities = soft_probabilities,
                                        logits = logits,
                                        labels = labels,
                                        teacher_kernels = teacher_kernels,
                                        student_kernels = student_kernels,
                                        options = loss_parameters,
                                        distill_loss_function=distill_loss_function,
                                        student_loss_function=student_loss_function)
                        
                        elif(distillation_type == "traditional"):
                            loss =  vanillia_knowledge_distillation_loss( soft_targets = soft_targets,
                                    soft_probabilities = soft_probabilities,
                                    logits = logits,
                                    labels= labels,
                                    distill_loss_function= distill_loss_function,
                                    student_loss_function = student_loss_function,
                                    options = loss_parameters)
                        else:
                            loss = loss_function(input=logits, target=labels)
                    else:
                        loss = loss_function(input=logits, target=labels)
                    
                    valid_loss+=loss.item()*images.size(0)
                    _, predictions = torch.max(logits.data,1)
                    val_correct+=(predictions == labels).sum().item()

            t2 = time.time()
                
            avg_train_loss_per_epoch = (train_loss / len(train_dataset))
            avg_train_acc_per_epoch = (train_correct / len(train_dataset)) * 100
            avg_validation_loss_per_epoch = valid_loss / len(valid_dataset)
            avg_validation_acc_per_epoch = val_correct / len(valid_dataset) * 100
            avg_train_run_time = t1-t0
            avg_validation_run_time = t2-t1

            avg_train_loss_per_epochs.append(avg_train_loss_per_epoch)
            avg_train_acc_per_epochs.append(avg_train_acc_per_epoch)
            avg_validation_loss_per_epochs.append(avg_validation_loss_per_epoch)
            avg_validation_acc_per_epochs.append(avg_validation_acc_per_epoch)
            avg_train_run_times_per_epochs.append(avg_train_run_time)
            avg_validation_run_times_per_epochs.append(avg_validation_run_time)




            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} % Training Time Taken: {:.2f} Seconds Validation Time Taken {:.2f} Seconds"
                    .format(epoch + 1,
                    number_of_epochs,
                    avg_train_loss_per_epoch,
                    avg_validation_loss_per_epoch,
                    avg_train_acc_per_epoch,
                    avg_validation_acc_per_epoch,
                    avg_train_run_time,
                    avg_validation_run_time
                    ))
            
            iteration = epoch+1
            
            if(early_stopper.early_stop(validation_loss=avg_validation_loss_per_epoch)):
                break
            
    

            avg_train_loss_per_epochs.append(avg_train_loss_per_epoch)
            avg_train_acc_per_epochs.append(avg_train_acc_per_epoch)
            avg_validation_loss_per_epochs.append(avg_validation_loss_per_epoch)
            avg_validation_acc_per_epochs.append(avg_validation_acc_per_epoch)
            avg_train_run_times_per_epochs.append(avg_train_run_time)
            avg_validation_run_times_per_epochs.append(avg_validation_run_time)

        return {"results": {"avg_train_loss_per_epochs": avg_train_loss_per_epochs,
                                "avg_validation_loss_per_epochs":avg_validation_loss_per_epochs,
                                "avg_train_acc_per_epochs": avg_train_acc_per_epochs,
                                "avg_validation_acc_per_epochs": avg_validation_acc_per_epochs,
                                "avg_train_run_times": avg_train_run_times_per_epochs,
                                "avg_validation_run_times": avg_validation_run_times_per_epochs},
                    "train_info":{
                        "convergence_iteration": iteration,
                    },
                    "model": model
                }

