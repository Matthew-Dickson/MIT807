import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, MNIST
from Data.Utilities.device_loader import get_device, ToDeviceLoader
from Data.Utilities.data_transformer import trainingAugmentation
from Functions.LossFunctions.loss_functions import knowledge_distillation_loss
from Models.DummyTeacherModel import DummyTeacherModel
from Models.DummyStudentModel import DummyStudentModel
from training_scheme import train, train_knowledge_distilation, train_knowledge_distilation_no_k
from utils.fileUtil import FileUtil
import random
import numpy as np

BATCH_SIZE = 128
K_SPLITS = 10
NUMBER_OF_EPOCHS = 10
RANDOM_STATE = 42
OPTIONS = {"alpha" : 1}
TEMPERATURE = 5
OPTIMIZER = torch.optim.Adam
TEACHER_CRITERION =  torch.nn.CrossEntropyLoss()
STUDENT_CRITERION = knowledge_distillation_loss
LEARNING_RATE = 0.5
RUN_ON = "MNIST"
RUN_K_FOLD = False
NAME_OF_TEACHER_MODEL = "dummyParent"
NAME_OF_STUDENT_MODEL = "dummyStudent"
KNOWLEDGE_DISTILLATION = True
SAVE_HISTORY_FILE_PATH = "./Data/history.txt"
# teacher_model_number = 3
# teacher_model_number = 18 
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if __name__ == '__main__':

    #Get GPU if avialable
    device = get_device()

    model_histories = None

    #Model dimensions
    input_channels = None
    output_channels = None
  
    #loading data 
    train_dataset = None
    test_dataset = None

    if(RUN_ON == "CIFAR100"):
        input_channels = 3
        output_channels = 100
        train_dataset = CIFAR100(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = CIFAR100(root='Data/', train=False)
    
    if(RUN_ON == "MNIST"):
        input_channels = 1
        output_channels = 10
        train_dataset = MNIST(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = MNIST(root='Data/', train=False)

    test_dl = ToDeviceLoader(DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False), device)
    
    if(KNOWLEDGE_DISTILLATION): 

        if(RUN_K_FOLD):
            model_histories=train_knowledge_distilation(train_dataset=train_dataset,
                               model_optimizer=OPTIMIZER,
                               student_model=DummyStudentModel,
                               teacher_model=DummyTeacherModel,
                               name_of_teacher_model = NAME_OF_TEACHER_MODEL,
                               input_channels=input_channels,
                               output_channels=output_channels,
                               learning_rate=LEARNING_RATE,
                               criterion=STUDENT_CRITERION,
                               device=device,
                               temperature = TEMPERATURE,
                               options=OPTIONS,
                               num_of_epochs=NUMBER_OF_EPOCHS,
                               k_splits=K_SPLITS,
                               batch_size = BATCH_SIZE,
                               random_state=RANDOM_STATE)

        else:
             model_histories=train_knowledge_distilation_no_k(train_dataset=train_dataset,
                            model_optimizer=OPTIMIZER,
                            student_model=DummyStudentModel,
                            teacher_model=DummyTeacherModel,
                            name_of_teacher_model = NAME_OF_TEACHER_MODEL,
                            input_channels=input_channels,
                            output_channels=output_channels,
                            learning_rate=LEARNING_RATE,
                            criterion=STUDENT_CRITERION,
                            device=device,
                            temperature = TEMPERATURE,
                            options=OPTIONS,
                            num_of_epochs=NUMBER_OF_EPOCHS,
                            batch_size = BATCH_SIZE)
    else:
         model_histories=train(train_dataset=train_dataset,
                           specific_model=DummyTeacherModel,
                           name_of_model=NAME_OF_TEACHER_MODEL,
                           model_optimizer=OPTIMIZER,
                           input_channels=input_channels,
                           output_channels=output_channels,
                           criterion=TEACHER_CRITERION,
                           device=device,
                           learning_rate=LEARNING_RATE,
                           num_of_epochs=NUMBER_OF_EPOCHS,
                           batch_size = BATCH_SIZE)
         
    
    file_helper = FileUtil()
    data = file_helper.save_to_file(model_histories,SAVE_HISTORY_FILE_PATH)
    

    

