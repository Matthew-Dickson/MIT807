import torch
from torchvision.datasets import CIFAR100, MNIST
from Data.Utilities.device_loader import get_device
from Data.Utilities.data_transformer import trainingAugmentation
from Functions.LossFunctions.loss_functions import knowledge_distillation_loss
from Models.DummyTeacherModel import DummyTeacherModel
from Models.ResNet2 import ResNet110
from training_scheme import train
import random
import numpy as np

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)



BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 100

OPTIMIZER = torch.optim.Adam
TEACHER_CRITERION =  torch.nn.CrossEntropyLoss()
STUDENT_CRITERION = knowledge_distillation_loss
LEARNING_RATE = 0.01
TRAIN_VALID_SPLIT = 0.8

RUN_ON = "CIFAR100"
TEACHER_MODEL = ResNet110
DISTILLATION_TYPE = "None"
SAVE_TEACHER_PATH = './Data/Models/dummyParent.pt'


EARLY_STOPING_OPTIONS = {
    "patience": 5,
    "min_delta": 0
}

LOSS_OPTIONS = {
      "distillation_type": DISTILLATION_TYPE,
}


TRAIN_OPTIONS = {"learning_rate": LEARNING_RATE,
                  "optimizer" : OPTIMIZER,
                  "batch_size": BATCH_SIZE,
                  "number_of_epochs": NUMBER_OF_EPOCHS,
                  "loss_parameters": LOSS_OPTIONS,
                  "early_stopping": EARLY_STOPING_OPTIONS}



def split_dataset(dataset, split_percentage):
    first_partition_size = int(split_percentage * len(dataset))
    second_partition_size = len(dataset) - first_partition_size
    first_partition, second_partition = torch.utils.data.random_split(dataset,[first_partition_size,second_partition_size])
    return first_partition, second_partition


if __name__ == '__main__':

    #Get GPU if avialable
    device = get_device()

    #Model dimensions
    input_channels = None
    output_channels = None
  
    #loading data 
    train_dataset = None
    valid_dataset = None
    test_dataset = None

    if(RUN_ON == "CIFAR100"):
        input_channels = 3
        output_channels = 100
        train_val_dataset = CIFAR100(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = CIFAR100(root='Data/', train=False,transform=trainingAugmentation())
    
    if(RUN_ON == "MNIST"):
        input_channels = 1
        output_channels = 10
        train_val_dataset = MNIST(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = MNIST(root='Data/', train=False,transform=trainingAugmentation())

   
    train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset,split_percentage=TRAIN_VALID_SPLIT)
    model_information=train(train_dataset=train_dataset,
                    valid_dataset = valid_dataset,
                    student_model=TEACHER_MODEL,
                    teacher_model=None,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    device=device,
                    train_options = TRAIN_OPTIONS)
    model = model_information["model"]
    model.save(SAVE_TEACHER_PATH)
    
    
    

    

