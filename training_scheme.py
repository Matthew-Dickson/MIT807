import torch
import time
from torch.utils.data import DataLoader
from data.utilities.device_loader import ToDeviceLoader, to_device
from functions.loss_type import LossType
from utils.early_stopper import EarlyStopper

SCHEDULE = [81,122]

def adjust_learning_rate(optimizer, epoch, learning_rate, gamma):
    global state
    if epoch in SCHEDULE:
        learning_rate *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    return learning_rate


def train(train_dataset,
          valid_dataset,
          student_model,
          teacher_model,
          train_options,
          loss_function,
          device):
        
        #Define Defaults
        learning_rate = train_options.get("learning_rate") if train_options.get("learning_rate") != None else 0.01
        batch_size = train_options.get("batch_size") if train_options.get("batch_size") != None else 128
        model_optimizer = train_options.get("optimizer") if train_options.get("optimizer") != None else torch.optim.Adam
        number_of_epochs = train_options.get("number_of_epochs") if train_options.get("number_of_epochs") != None else 9999999999999


        early_stopping_options = train_options.get("early_stopping") 
        patience =  early_stopping_options.get("patience") if early_stopping_options.get("patience") != None else 5
        min_delta =  early_stopping_options.get("min_delta") if early_stopping_options.get("min_delta") != None else 0
        
   
        loss_parameters = train_options.get("loss_parameters")
        distillation_type =  loss_parameters.get("distillation_type") if loss_parameters.get("distillation_type") != None else "none"

        if(distillation_type == "traditional" or distillation_type == filter):
            if(train_options.get("loss_parameters") == None):
                raise Exception("Requires loss parameters")
        
        model = to_device(student_model,device=device)
        GAMMA = 0.1

        #optimizer =  model_optimizer(model.parameters(), lr=learning_rate)

        optimizer =  model_optimizer(model.parameters(), lr=learning_rate, momentum=0.9,
                                weight_decay=1e-4)
        
       
        train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=False)
        valid_dl = DataLoader(dataset=valid_dataset, batch_size=batch_size,shuffle=False)

        train_data_on_specified_device = ToDeviceLoader(train_dl, device)
        valid_data_on_specified_device = ToDeviceLoader(valid_dl, device)

        early_stopper=EarlyStopper(patience=patience,min_delta=min_delta)

        avg_train_loss_per_epochs = []
        avg_train_acc_per_epochs = []
        avg_validation_loss_per_epochs = []
        avg_validation_acc_per_epochs = []
        avg_train_run_times_per_epochs = []
        avg_validation_run_times_per_epochs = []


        for epoch in range(number_of_epochs):
            learning_rate = adjust_learning_rate(optimizer, epoch,learning_rate,GAMMA)
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

                    if(distillation_type == LossType.FILTER.name):
                        loss = loss_function(student_logits = logits,
                                                                  labels = labels,
                                                                  features = images,
                                                                  teacher_model = teacher_model,
                                                                  student_model = student_model)
                        
                    elif(distillation_type == LossType.TRADITIONAL.name):
                        loss = loss_function(student_logits = logits,
                                                                  labels = labels,
                                                                  features = images,
                                                                  teacher_model = teacher_model)
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

                        if(distillation_type == LossType.FILTER.name):
                            loss = loss_function(student_logits = logits,
                                                                  labels = labels,
                                                                  features = images,
                                                                  teacher_model = teacher_model,
                                                                  student_model = student_model)
                        
                        elif(distillation_type == LossType.TRADITIONAL.name):
                            loss = loss_function(student_logits = logits,
                                                                    labels = labels,
                                                                    features = images,
                                                                    teacher_model = teacher_model)
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
            avg_validation_loss_per_epoch = (valid_loss / len(valid_dataset))
            avg_validation_acc_per_epoch = (val_correct / len(valid_dataset)) * 100
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
            
            # if(early_stopper.early_stop(validation_loss=avg_validation_loss_per_epoch)):
            #     break

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

