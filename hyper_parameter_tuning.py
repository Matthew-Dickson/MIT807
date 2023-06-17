import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, MNIST
from data.utilities.device_loader import get_device, ToDeviceLoader, to_device
from data.utilities.data_transformer import trainingAugmentation
from functions.loss.filter_knowledge_distillation_loss import FilterKnowledgeDistillationLoss
from functions.loss.traditional_distillation_loss import TraditionalKnowledgeDistillationLoss
from functions.loss_type import LossType
from models.DummyTeacherModel import DummyTeacherModel
from models.DummyStudentModel import DummyStudentModel
from models.ResNet import resnet32, resnet110
from training_scheme import train
from utils.fileUtil import FileUtil
import random
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter
import itertools
import argparse
import json
import torch.nn as nn

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BATCH_SIZE = 128
K_SPLITS = 10
NUMBER_OF_EPOCHS = 10

TEMPERATURE = [5,10,15,20]
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = [0.01,0.05,0.005]
TRAIN_VALID_SPLIT = 0.8
ALPHA = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
BETA = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
RUN_ON = "MNIST"
RUN_K_FOLD = True
TEACHER_MODEL = DummyTeacherModel
DISTILLATION_TYPE = 1
FILE_PATH_OF_TEACHER = "./Data/Models/dummyParent.pt"
SAVE_HISTORY_FILE_PATH = "./Data/"
SAVE_TEACHER_PATH = './Data/Models/dummyParent.pt'
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MINIMUM_DELTA = 0

START = 23

parser = argparse.ArgumentParser()
parser.add_argument('--numberOfEpochs', help='Number of epochs to train for', default=NUMBER_OF_EPOCHS)
parser.add_argument('--learningRates', help='List of The learning rate applied to the optimizer', default=LEARNING_RATE)
parser.add_argument('--batchSize', help='The batch size to backpropagate on', default=BATCH_SIZE)
parser.add_argument('--lossFunction', help='The loss function to use', default=DISTILLATION_TYPE)
parser.add_argument('--runOn', help='The data set to run on', default=RUN_ON)
parser.add_argument('--runKFold', help='Run training with kfold cross validation', default=RUN_K_FOLD)
parser.add_argument('--ksplits', help='Number of K fold splits', default=K_SPLITS)
parser.add_argument('--trainValidSplit', help='Split percentage of training and validation', default=TRAIN_VALID_SPLIT)
parser.add_argument('--temperatures', help='List of Temperate to use for softmax', default=TEMPERATURE)
parser.add_argument('--alphas', help='List of Alpha to use for traditional knowledge distillation loss component', default=ALPHA)
parser.add_argument('--betas', help='List of Beta to use for knowledge distillation filter loss component', default=BETA)
parser.add_argument('--teacherFilePath', help='The path to the teacher model', default=FILE_PATH_OF_TEACHER)
parser.add_argument('--earlyStoppingPatience', help='The early stopping patience', default=EARLY_STOPPING_PATIENCE)
parser.add_argument('--earlyStoppingMinimumDelta', help='The early stopping minimum delta', default=EARLY_STOPPING_MINIMUM_DELTA)
parser.add_argument('--saveHistoryFilePath', help='The path to save results', default=SAVE_HISTORY_FILE_PATH)
parser.add_argument('--indexConfig', help='Where to start in the hyper parameter search', default=START)
args = parser.parse_args()


EARLY_STOPING_OPTIONS = {
    "patience": int(args.earlyStoppingPatience),
    "min_delta": float(args.earlyStoppingMinimumDelta)
}


def split_dataset(dataset, split_percentage):
    first_partition_size = int(split_percentage * len(dataset))
    second_partition_size = len(dataset) - first_partition_size
    first_partition, second_partition = torch.utils.data.random_split(dataset,[first_partition_size,second_partition_size])
    return first_partition, second_partition

def create_hyperparams_grid(config):
    keys, values = zip(*config.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts

def get_loss_function(loss_type, loss_options):
    loss_function = None
    if(LossType.FILTER.name == LossType(loss_type).name):
        loss_function = FilterKnowledgeDistillationLoss(options=loss_options)
    elif(LossType.TRADITIONAL.name == LossType(loss_type).name):
        loss_function = TraditionalKnowledgeDistillationLoss(options=loss_options)
    elif(LossType.ATTENTION.name == LossType(loss_type).name):
        loss_function = nn.CrossEntropyLoss()
    elif(LossType.CE.name == LossType(loss_type).name):
        loss_function = nn.CrossEntropyLoss()
    return loss_function


def execute_hyperparameter_tuning(hyper_parameters,teacher,test_data_on_specified_device,start=0):


    for index, config in enumerate(hyper_parameters):

        if(index < start-1):
            continue

        LOSS_OPTIONS = {
        "temperature": int(config["TEMPERATURE"]),
        "distillation_type": LossType(args.lossFunction).name,
        "alpha" : float(config["ALPHA"]),
        "beta" : float(config["BETA"])
            }
        
        

        TRAIN_OPTIONS = {"learning_rate": float(config["LEARNING_RATE"]),
                  "optimizer" : OPTIMIZER,
                  "batch_size": int(args.batchSize),
                  "number_of_epochs": int(args.numberOfEpochs),
                  "loss_parameters": LOSS_OPTIONS,
                  "early_stopping": EARLY_STOPING_OPTIONS}

        model_histories={}
        file_helper = FileUtil()
        current_configuration = {
                "train_options" : {
                    "learning_rate": config["LEARNING_RATE"],
                    "optimizer" : OPTIMIZER.__name__,
                    "batch_size": int(args.batchSize),
                    "number_of_epochs": int(args.numberOfEpochs),
                    "loss_parameters": {
                            "temperature": int(config["TEMPERATURE"]),
                                "distillation_type": LossType(args.lossFunction).name,
                                "alpha" : float(config["ALPHA"]),
                                "beta" : float(config["BETA"])
                        }
                    }
                }
        model_histories['config'] = current_configuration

        if(bool(args.runKFold)):

            #Get k folds
            k_folds =KFold(n_splits=int(args.ksplits))
            for fold, (train_idx,val_idx) in enumerate(k_folds.split(train_val_dataset)):
                if(args.runOn == "CIFAR100"):
                    input_channels = 3
                    output_channels = 100
                    STUDENT_MODEL = resnet32()
            
                if(args.runOn == "MNIST"):
                    input_channels = 1
                    output_channels = 10
                    STUDENT_MODEL = DummyStudentModel(input_channels=input_channels,num_classes=output_channels)

                print('Fold {}'.format(fold + 1))
                #History for current fold
                history = {'train_losses': [], 'valid_losses': [],'train_accs':[],'valid_accs':[], 'train_times':[], 'valid_times': [],'test_accuracy': None, 'convergence_iteration': None}
                #Gets data
                train_dataset = itemgetter(*train_idx)(train_val_dataset)
                valid_dataset = itemgetter(*val_idx)(train_val_dataset)

                loss_function = get_loss_function(int(args.lossFunction), loss_options=LOSS_OPTIONS)

                model_information=train(train_dataset=train_dataset,
                        loss_function = loss_function,
                        valid_dataset = valid_dataset,
                        student_model=STUDENT_MODEL,
                        teacher_model=teacher,
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
                file_helper.save_to_file(model_histories,args.saveHistoryFilePath+"/"+LossType(args.lossFunction).name+"_history_"+"hyper_parameter_configuration_"+str(index)+".json")
        else:
            if(args.runOn == "CIFAR100"):
                input_channels = 3
                output_channels = 100
                STUDENT_MODEL = resnet32(num_classes=output_channels,input_channels=input_channels)
            
            if(args.runOn == "MNIST"):
                input_channels = 1
                output_channels = 10
                STUDENT_MODEL = DummyStudentModel(input_channels=input_channels,num_classes=output_channels)
            history = {'train_losses': [], 'valid_losses': [],'train_accs':[],'valid_accs':[], 'train_times':[], 'valid_times': [],'test_accuracy': None, 'convergence_iteration': None}
            #Split data
            train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset,split_percentage=args.trainValidSplit)
            loss_function = get_loss_function(int(args.lossFunction), loss_options=LOSS_OPTIONS)
            model_information=train(train_dataset=train_dataset,
                        loss_function = loss_function,
                        valid_dataset = valid_dataset,
                        student_model=STUDENT_MODEL,
                        teacher_model=teacher,
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
            model_histories["results"] = history 
            file_helper.save_to_file(model_histories,args.saveHistoryFilePath+"/"+LossType(args.lossFunction).name+"_history_"+"hyper_parameter_configuration_"+str(index)+".json")




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

    if(args.runOn== "CIFAR100"):
        input_channels = 3
        output_channels = 100
        train_val_dataset = CIFAR100(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = CIFAR100(root='Data/', train=False,transform=trainingAugmentation())
        TEACHER_MODEL = resnet110(num_classes=output_channels,input_channels=input_channels) 
    
    if(args.runOn == "MNIST"):
        input_channels = 1
        output_channels = 10
        train_val_dataset = MNIST(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = MNIST(root='Data/', train=False,transform=trainingAugmentation())
        TEACHER_MODEL = DummyTeacherModel(num_classes=output_channels,input_channels=input_channels) 

    test_dl = DataLoader(dataset=test_dataset, batch_size=int(args.batchSize), shuffle=False)
    test_data_on_specified_device = ToDeviceLoader(test_dl, device)
    
    try:
        teacher = to_device(TEACHER_MODEL,device=device)
        teacher.load(FILE_PATH_OF_TEACHER)
    except:
        raise Exception("Could not load teacher model")
    

    if isinstance(args.learningRates,str):

        hyperparams_grid = create_hyperparams_grid(config={
            "LEARNING_RATE":  json.loads(args.learningRates),
            "ALPHA": json.loads(args.alphas),
            "BETA": json.loads(args.betas),
            "TEMPERATURE": json.loads(args.temperatures)
        })
    else:
          hyperparams_grid = create_hyperparams_grid(config={
            "LEARNING_RATE":  args.learningRates,
            "ALPHA": args.alphas,
            "BETA": args.betas,
            "TEMPERATURE": args.temperatures
        })


    execute_hyperparameter_tuning(hyperparams_grid,
                                  teacher=teacher,
                                  test_data_on_specified_device=test_data_on_specified_device,
                                  start=START)
           
    
    

    

