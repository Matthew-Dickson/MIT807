import torch
import time
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100, MNIST
from sklearn.model_selection import KFold
from Data.Utilities.device_loader import get_device, ToDeviceLoader, to_device
from Data.Utilities.data_transformer import trainingAugmentation
from Models.Blocks.ResidualBlock import ResidualBlock
from Models.ResNet import ResNet
from Models.DummyTeacherModel import DummyTeacherModel
from Models.DummyStudentModel import DummyStudentModel
import torch.nn as nn
import sys

sys.setrecursionlimit(100000000)


import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

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

def train(train_dataset,
          specified_optimizer,
          specified_model,
          teacher_model,
          input_channels,
          learning_rate,
          criterion,
          device,
          name_of_model,
          knowledge_distillation = True,
          num_of_epochs = 100,
          save = False,
          k_splits = 5):

    model_histories={}

    if knowledge_distillation:
        teacher = teacher_model(input_channels=input_channels)
        teacher.load('./Data/Models/{}.pt'.format(name_of_model))


    if save is False:
        #Get k folds
        k_folds =KFold(n_splits=k_splits,shuffle=True,random_state=RANDOM_STATE)
        for fold, (train_idx,val_idx) in enumerate(k_folds.split(torch.arange(len(train_dataset)))):
                print('Fold {}'.format(fold + 1))

                model = to_device(specified_model(input_channels=input_channels), device)
                optimizer =  specified_optimizer(model.parameters(), lr=learning_rate)
            
                #History for current fold
                history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[], 'train_time':[], 'valid_time': []}
                # Initializes training and validation data loaders
                train_sampler = SubsetRandomSampler(train_idx)
                validation_sampler = SubsetRandomSampler(val_idx)

                train_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
                validation_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

                train_data_on_specified_device = ToDeviceLoader(train_dl, device)
                validation_data_on_specified_device= ToDeviceLoader(validation_dl, device)
                
                for epoch in range(num_of_epochs):
                    t0 = time.time()
                    train_loss, train_correct=model.train_epoch(train_data_on_specified_device,criterion,optimizer, device)
                    t1 = time.time()
                    validation_loss, validation_correct=model.valid_epoch(validation_data_on_specified_device,criterion, device)
                    t2 = time.time()

                    avg_train_loss_per_epoch = train_loss / len(train_dl.sampler)
                    avg_train_acc_per_epoch = train_correct / len(train_dl.sampler) * 100
                    avg_validation_loss_per_epoch = validation_loss / len(validation_dl.sampler)
                    avg_validation_acc_per_epoch = validation_correct / len(validation_dl.sampler) * 100

                    print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} % Training Time Taken: {} Seconds Validation Time Taken {} Seconds"
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
                loop_through_layers(model)
    else:

        model = to_device(specified_model(input_channels=input_channels), device)
        optimizer =  specified_optimizer(model.parameters(), lr=learning_rate)
        #History for current fold
        history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[], 'train_time':[], 'valid_time': []}
        train_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
        train_data_on_specified_device = ToDeviceLoader(train_dl, device)

        for epoch in range(num_of_epochs):
            t0 = time.time()
            train_loss, train_correct=model.train_epoch(train_data_on_specified_device,criterion,optimizer, device)
            t1 = time.time()
         
            avg_train_loss_per_epoch = train_loss / len(train_dataset)
            avg_train_acc_per_epoch = train_correct / len(train_dataset) * 100


            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Training Acc {:.2f} % Training Time Taken: {} "
                .format(epoch + 1,
                num_of_epochs,
                avg_train_loss_per_epoch,
                avg_train_acc_per_epoch,
                t1-t0,
                ))
            
            history['train_loss'].append(avg_train_loss_per_epoch)    
            history['train_acc'].append(avg_train_acc_per_epoch)  
            history['train_time'].append(t1-t0)

        model.save('./Data/Models/{}.pt'.format(name_of_model))
    return history





BATCH_SIZE = 128
K_SPLITS = 5
NUMBER_OF_EPOCHS = 5
RANDOM_STATE = 42
OPTIMIZER = torch.optim.Adam
CRITERION = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 0.002
RUN_DUMMY = True
SAVE=False
NAME_OF_MODEL = "dummyParent"
KNOWLEDGE_DISTILLATION = False
# teacher_model_number = 3
# teacher_model_number = 18 

if __name__ == '__main__':

    #Get GPU if avialable
    device = get_device()

    data = None
    input_channels = None
  
    if(RUN_DUMMY):
        data = 'MNIST'
        input_channels = 1
    else:
        data = 'CIFAR100'
        input_channels = 3


    #loading data 
    train_dataset = None
    test_dataset = None

    if(data == "CIFAR100"):
        train_dataset = CIFAR100(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = CIFAR100(root='Data/', train=False)
    
    if(data == "MNIST"):
        train_dataset = MNIST(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = MNIST(root='Data/', train=False)

    test_dl = ToDeviceLoader(DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False), device)
    
    model_histories=train(train_dataset=train_dataset,
                           specified_optimizer=OPTIMIZER,
                           specified_model=DummyStudentModel,
                           teacher_model=DummyTeacherModel,
                           input_channels=input_channels,
                           learning_rate=LEARNING_RATE,
                           criterion=CRITERION,
                           device=device,
                           name_of_model=NAME_OF_MODEL,
                           num_of_epochs=NUMBER_OF_EPOCHS,
                           k_splits=K_SPLITS,
                           save=SAVE,
                           knowledge_distillation=KNOWLEDGE_DISTILLATION)
    
    print(model_histories)

    

