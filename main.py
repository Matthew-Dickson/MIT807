import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, MNIST
from Data.Utilities.device_loader import get_device, ToDeviceLoader, to_device
from Data.Utilities.data_transformer import trainingAugmentation
from Functions.LossFunctions.loss_functions import knowledge_distillation_loss
from Models.DummyTeacherModel import DummyTeacherModel
from Models.DummyStudentModel import DummyStudentModel
from training_scheme import train
from utils.fileUtil import FileUtil
import random
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)



BATCH_SIZE = 128
K_SPLITS = 10
NUMBER_OF_EPOCHS = 1000



TEMPERATURE = 20
OPTIMIZER = torch.optim.Adam
TEACHER_CRITERION =  torch.nn.CrossEntropyLoss()
STUDENT_CRITERION = knowledge_distillation_loss
LEARNING_RATE = 0.01
TRAIN_VALID_SPLIT = 0.8

RUN_ON = "MNIST"
RUN_K_FOLD = False
TRAIN_TEACHER = False
TEACHER_MODEL = DummyTeacherModel
DISTILLATION_TYPE = "filter"
FILE_PATH_OF_TEACHER = "./Data/Models/dummyParent.pt"
SAVE_HISTORY_FILE_PATH = "./Data/history.json"
# teacher_model_number = 3
# teacher_model_number = 18 
SAVE_TEACHER_PATH = './Data/Models/dummyParent.pt'


LOSS_OPTIONS = {
      "temperature": TEMPERATURE,
      "distillation_type": DISTILLATION_TYPE,
      "alpha" : 0.4,
      "beta" : 0.3
}

TRAIN_OPTIONS = {"learning_rate": LEARNING_RATE,
                  "optimizer" : OPTIMIZER,
                  "batch_size": BATCH_SIZE,
                  "number_of_epochs": NUMBER_OF_EPOCHS,
                  "loss_parameters": LOSS_OPTIONS}





def split_dataset(dataset, split_percentage):
    first_partition_size = int(split_percentage * len(dataset))
    second_partition_size = len(dataset) - first_partition_size
    first_partition, second_partition = torch.utils.data.random_split(dataset,[first_partition_size,second_partition_size])
    return first_partition, second_partition


if __name__ == '__main__':

    #Get GPU if avialable
    device = get_device()

    model_histories={}

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
        #train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset, split_percentage=TRAIN_VALID_SPLIT)
        test_dataset = CIFAR100(root='Data/', train=False,transform=trainingAugmentation())
    
    if(RUN_ON == "MNIST"):
        input_channels = 1
        output_channels = 10
        train_val_dataset = MNIST(root='Data/', train=True, download=True, transform=trainingAugmentation())
        #train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset, split_percentage=TRAIN_VALID_SPLIT)
        test_dataset = MNIST(root='Data/', train=False,transform=trainingAugmentation())

    test_dl = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_data_on_specified_device = ToDeviceLoader(test_dl, device)
    


    if(TRAIN_TEACHER == False):
            file_helper = FileUtil()
            config = {
                 "train_options" : {
                      "learning_rate": LEARNING_RATE,
                      "optimizer" : OPTIMIZER.__name__,
                      "batch_size": BATCH_SIZE,
                      "number_of_epochs": NUMBER_OF_EPOCHS,
                      "loss_parameters": LOSS_OPTIONS
                      }
                 }
            model_histories['config'] = config
            teacher = None
            
            try:
                teacher = to_device(TEACHER_MODEL(input_channels=input_channels,num_classes=output_channels),device=device)
                teacher.load(FILE_PATH_OF_TEACHER)
            except:
                raise Exception("Could not load teacher model")
            
            if(RUN_K_FOLD):
                #Get k folds
                k_folds =KFold(n_splits=K_SPLITS)
                for fold, (train_idx,val_idx) in enumerate(k_folds.split(train_val_dataset)):

                    print('Fold {}'.format(fold + 1))
                    #History for current fold
                    history = {'train_losses': [], 'valid_losses': [],'train_accs':[],'valid_accs':[], 'train_times':[], 'valid_times': [],'test_accuracy': None, 'convergence_iteration': None}
                    #Gets data
                    train_dataset = itemgetter(*train_idx)(train_val_dataset)
                    valid_dataset = itemgetter(*val_idx)(train_val_dataset)

                    model_information=train(train_dataset=train_dataset,
                            valid_dataset = valid_dataset,
                            student_model=DummyStudentModel,
                            teacher_model=teacher,
                            input_channels=input_channels,
                            output_channels=output_channels,
                            device=device,
                            train_options = TRAIN_OPTIONS)
                    
            
                    history['train_losses'].append(model_information["results"]["avg_train_loss_per_epochs"])    
                    history['valid_losses'].append(model_information["results"]["avg_validation_loss_per_epochs"])  
                    history['train_accs'].append(model_information["results"]["avg_train_acc_per_epochs"])  
                    history['valid_accs'].append(model_information["results"]["avg_validation_acc_per_epochs"])
                    history['train_times'].append(model_information["results"]["avg_train_run_times"])
                    history['valid_times'].append(model_information["results"]["avg_validation_run_times"])
                    history['convergence_iteration'] = model_information["train_info"]["convergence_iteration"]
                    
                   

                    model = model_information["model"]
                    _, correct = model.predict(test_data_on_specified_device,device)
                    test_accuracy = correct / len(test_dataset) * 100 
                    history['test_accuracy'] = test_accuracy
                    model_histories['fold{}'.format(fold+1)] = history 
                    file_helper.save_to_file(model_histories,SAVE_HISTORY_FILE_PATH)
            else:
                history = {'train_losses': [], 'valid_losses': [],'train_accs':[],'valid_accs':[], 'train_times':[], 'valid_times': [],'test_accuracy': None, 'convergence_iteration': None}
                #Split data
                train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset,split_percentage=TRAIN_VALID_SPLIT)
                model_information=train(train_dataset=train_dataset,
                            valid_dataset = valid_dataset,
                            student_model=DummyStudentModel,
                            teacher_model=teacher,
                            input_channels=input_channels,
                            output_channels=output_channels,
                            device=device,
                            train_options = TRAIN_OPTIONS)
                
                    
                history['train_losses'].append(model_information["results"]["avg_train_loss_per_epochs"])    
                history['valid_losses'].append(model_information["results"]["avg_validation_loss_per_epochs"])  
                history['train_accs'].append(model_information["results"]["avg_train_acc_per_epochs"])  
                history['valid_accs'].append(model_information["results"]["avg_validation_acc_per_epochs"])
                history['train_times'].append(model_information["results"]["avg_train_run_times"])
                history['valid_times'].append(model_information["results"]["avg_validation_run_times"])
                history['convergence_iteration'] = model_information["train_info"]["convergence_iteration"]
            
                model = model_information["model"]
                _, correct = model.predict(test_data_on_specified_device,device)
                test_accuracy = correct / len(test_dataset) * 100 
                history['test_accuracy'] = test_accuracy
                model_histories = history 
                file_helper = FileUtil()
                data = file_helper.save_to_file(model_histories,SAVE_HISTORY_FILE_PATH)
    else:
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
    
    
    

    

