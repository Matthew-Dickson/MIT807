import torch
import time
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from Data.Utilities.device_loader import ToDeviceLoader, to_device
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

import random
import numpy as np

from Functions.LossFunctions.loss_functions import cross_entropy

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, w, h)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))    
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(torch.Tensor.cpu(grid).numpy().transpose((1, 2, 0)))

def loop_through_layers(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            kernels = layer.weight.data.clone()
            visTensor(kernels, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    

g = torch.Generator()
g.manual_seed(0)

def train(train_dataset,
          specific_model,
          name_of_model,
          model_optimizer,
          input_channels,
          output_channels,
          learning_rate,
          device,
          num_of_epochs,
          criterion,
          batch_size):
        
        model = to_device(specific_model(input_channels=input_channels,num_classes=output_channels),device=device)
        optimizer =  model_optimizer(model.parameters(), lr=learning_rate)
        #History for current fold
        history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[], 'train_time':[], 'valid_time': []}
        train_dl = DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               worker_init_fn=seed_worker,
                                generator=g)
        train_data_on_specified_device = ToDeviceLoader(train_dl, device)

        model_histories={}

        for epoch in range(num_of_epochs):
            t0 = time.time()
            train_loss, train_correct=model.train_epoch(dataloader = train_data_on_specified_device,
                                                        loss_fn = criterion,
                                                        optimizer = optimizer,
                                                        device = device)
            t1 = time.time()
         
            avg_train_loss_per_epoch = train_loss / len(train_dataset)
            avg_train_acc_per_epoch = train_correct / len(train_dataset) * 100


            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Training Acc {:.2f} % Training Time Taken: {:.2f} "
                .format(epoch + 1,
                num_of_epochs,
                avg_train_loss_per_epoch,
                avg_train_acc_per_epoch,
                t1-t0,
                ))
            
            history['train_loss'].append(avg_train_loss_per_epoch)    
            history['train_acc'].append(avg_train_acc_per_epoch)  
            history['train_time'].append(t1-t0)

            model_histories['history'] = history 

        model.save('./Data/Models/{}.pt'.format(name_of_model))
        return model_histories


def train_knowledge_distilation_no_k(train_dataset,
          model_optimizer,
          student_model,
          teacher_model,
          name_of_teacher_model,
          input_channels,
          output_channels,
          learning_rate,
          criterion,
          temperature,
          options,
          device,
          batch_size,
          num_of_epochs = 100):
        

        try:
            teacher = to_device(teacher_model(input_channels=input_channels,num_classes=output_channels),device=device)
            teacher.load('./Data/Models/{}.pt'.format(name_of_teacher_model))
        except:
            raise Exception("Could not load teacher model")
        
        model = to_device(student_model(input_channels=input_channels,num_classes=output_channels),device=device)
        optimizer =  model_optimizer(model.parameters(), lr=learning_rate)
        #History for current fold
        history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[], 'train_time':[], 'valid_time': []}
        train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size)
        train_data_on_specified_device = ToDeviceLoader(train_dl, device)

        model_histories={}

        distill_loss_function=nn.KLDivLoss(reduction="batchmean")                                                    
        student_loss_function=nn.CrossEntropyLoss()
        #student_loss_function= cross_entropy

        for epoch in range(num_of_epochs):
            t0 = time.time()
            train_loss, train_correct=model.train_epoch(dataloader=train_data_on_specified_device,
                                                                    loss_fn=criterion,
                                                                    optimizer=optimizer,
                                                                    teacher=teacher,
                                                                    temperature = temperature,
                                                                    distill_loss_function=distill_loss_function,
                                                                    student_loss_function=student_loss_function,
                                                                    options =options,
                                                                    device = device)
            t1 = time.time()
        
            avg_train_loss_per_epoch = train_loss / len(train_dataset)
            avg_train_acc_per_epoch = train_correct / len(train_dataset) * 100


            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Training Acc {:.2f} % Training Time Taken: {:.2f} "
                .format(epoch + 1,
                num_of_epochs,
                avg_train_loss_per_epoch,
                avg_train_acc_per_epoch,
                t1-t0,
                ))
            
            history['train_loss'].append(avg_train_loss_per_epoch)    
            history['train_acc'].append(avg_train_acc_per_epoch)  
            history['train_time'].append(t1-t0)

            model_histories['history'] = history 

        return model_histories
     

def train_knowledge_distilation(train_dataset,
          model_optimizer,
          student_model,
          teacher_model,
          name_of_teacher_model,
          input_channels,
          output_channels,
          learning_rate,
          criterion,
          device,
          temperature,
          options,
          batch_size,
          random_state,
          num_of_epochs = 100,
          k_splits = 5):

    model_histories={}

    try:
        teacher = to_device(teacher_model(input_channels=input_channels,num_classes=output_channels),device=device)
        teacher.load('./Data/Models/{}.pt'.format(name_of_teacher_model))
        for param_tensor in teacher.state_dict():
            print(param_tensor, "\t", teacher.state_dict()[param_tensor].size())
            print(param_tensor, "\t", teacher.state_dict()[param_tensor])
    except:
         raise Exception("Could not load teacher model")
    
    
    #Get k folds
    k_folds =KFold(n_splits=k_splits,shuffle=True,random_state=random_state)
    for fold, (train_idx,val_idx) in enumerate(k_folds.split(torch.arange(len(train_dataset)))):
            print('Fold {}'.format(fold + 1))

            model = to_device(student_model(input_channels=input_channels,num_classes=output_channels), device)
            optimizer =  model_optimizer(model.parameters(), lr=learning_rate)
        
            #History for current fold
            history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[], 'train_time':[], 'valid_time': []}
            # Initializes training and validation data loaders
            train_sampler = SubsetRandomSampler(train_idx)
            validation_sampler = SubsetRandomSampler(val_idx)

            train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
            validation_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=validation_sampler)

            train_data_on_specified_device = ToDeviceLoader(train_dl, device)
            validation_data_on_specified_device= ToDeviceLoader(validation_dl, device)
            
            distill_loss_function=nn.KLDivLoss(reduction="batchmean")                                                    
            student_loss_function=nn.CrossEntropyLoss()
            
            for epoch in range(num_of_epochs):
                t0 = time.time()

                train_loss, train_correct=model.train_epoch(dataloader=train_data_on_specified_device,
                                                                    loss_fn=criterion,
                                                                    optimizer=optimizer,
                                                                    teacher=teacher,
                                                                    temperature = temperature,
                                                                    distill_loss_function=distill_loss_function,
                                                                    student_loss_function=student_loss_function,
                                                                    options = options,
                                                                    device = device)
                t1 = time.time()
                validation_loss, validation_correct=model.valid_epoch(dataloader=validation_data_on_specified_device,
                                                                                loss_fn=criterion,
                                                                                teacher = teacher,
                                                                                temperature = temperature,
                                                                                distill_loss_function=distill_loss_function,
                                                                                student_loss_function=student_loss_function,
                                                                                options =options,
                                                                                device = device)
                t2 = time.time()

                avg_train_loss_per_epoch = train_loss / len(train_dl.sampler)
                avg_train_acc_per_epoch = train_correct / len(train_dl.sampler) * 100
                avg_validation_loss_per_epoch = validation_loss / len(validation_dl.sampler)
                avg_validation_acc_per_epoch = validation_correct / len(validation_dl.sampler) * 100

                print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} % Training Time Taken: {:.2f} Seconds Validation Time Taken {:.2f} Seconds"
                    .format(epoch + 1,
                    num_of_epochs,
                    avg_train_loss_per_epoch,
                    avg_validation_loss_per_epoch,
                    avg_train_acc_per_epoch,
                    avg_validation_acc_per_epoch,
                    t1-t0,
                    t2-t1
                    ))
                
                history['train_loss'].append(avg_train_loss_per_epoch)    
                history['valid_loss'].append(avg_validation_loss_per_epoch)  
                history['train_acc'].append(avg_train_acc_per_epoch)  
                history['valid_acc'].append(avg_validation_acc_per_epoch)
                history['train_time'].append(t1-t0)
                history['valid_time'].append(t2-t1)
        
            model_histories['fold{}'.format(fold+1)] = history 

    return model_histories
