import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, MNIST
from Data.Utilities.device_loader import get_device, ToDeviceLoader, to_device
from Data.Utilities.data_transformer import trainingAugmentation
from Models.DummyTeacherModel import DummyTeacherModel
from Models.DummyStudentModel import DummyStudentModel
from Models.ResNet2 import ResNet110, ResNet34
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

#Defaults
BATCH_SIZE = 128
K_SPLITS = 10
NUMBER_OF_EPOCHS = 10
TEMPERATURE = 20
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 0.01
TRAIN_VALID_SPLIT = 0.8
RUN_ON = "MNIST"
RUN_K_FOLD = False
DISTILLATION_TYPE = "filter"
FILE_PATH_OF_TEACHER = './Data/Models/dummyParent.pt'
SAVE_HISTORY_FILE_PATH = "./Data/"
ALPHA = 0.4
BETA = 0.3
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MINIMUM_DELTA = 0


parser = argparse.ArgumentParser()
parser.add_argument('--numberOfEpochs', help='Number of epochs to train for', default=NUMBER_OF_EPOCHS)
parser.add_argument('--learningRate', help='The learning rate applied to the optimizer', default=LEARNING_RATE)
parser.add_argument('--batchSize', help='The batch size to backpropagate on', default=BATCH_SIZE)
parser.add_argument('--lossFunction', help='The loss function to use', default=DISTILLATION_TYPE)
parser.add_argument('--runOn', help='The data set to run on', default=RUN_ON)
parser.add_argument('--runKFold', help='Run training with kfold cross validation', default=RUN_K_FOLD)
parser.add_argument('--ksplits', help='Number of K fold splits', default=K_SPLITS)
parser.add_argument('--trainValidSplit', help='Split percentage of training and validation', default=TRAIN_VALID_SPLIT)
parser.add_argument('--temperature', help='Temperate to use for softmax', default=TEMPERATURE)
parser.add_argument('--alpha', help='Alpha to use for traditional knowledge distillation loss component', default=ALPHA)
parser.add_argument('--beta', help='Beta to use for knowledge distillation filter loss component', default=BETA)
parser.add_argument('--teacherFilePath', help='The path to the teacher model', default=FILE_PATH_OF_TEACHER)
parser.add_argument('--earlyStoppingPatience', help='The early stopping patience', default=EARLY_STOPPING_PATIENCE)
parser.add_argument('--earlyStoppingMinimumDelta', help='The early stopping minimum delta', default=EARLY_STOPPING_MINIMUM_DELTA)
parser.add_argument('--saveHistoryFilePath', help='The path to save results', default=SAVE_HISTORY_FILE_PATH)
args = parser.parse_args()


LOSS_OPTIONS = {
      "temperature": int(args.temperature),
      "distillation_type": args.lossFunction,
      "alpha" : float(args.alpha),
      "beta" : float(args.beta)
}

EARLY_STOPING_OPTIONS = {
    "patience": int(args.earlyStoppingPatience),
    "min_delta": float(args.earlyStoppingMinimumDelta)
}


TRAIN_OPTIONS = {"learning_rate": float(args.learningRate),
                  "optimizer" : OPTIMIZER,
                  "batch_size": int(args.batchSize),
                  "number_of_epochs": int(args.numberOfEpochs),
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

    if(args.runOn == "CIFAR100"):
        input_channels = 3
        output_channels = 100
        train_val_dataset = CIFAR100(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = CIFAR100(root='Data/', train=False,transform=trainingAugmentation())
        TEACHER_MODEL = ResNet110(num_classes=output_channels,input_channels=input_channels) 
    
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
        teacher.load(args.teacherFilePath)
    except:
        raise Exception("Could not load teacher model")
    

    model_histories={}
    file_helper = FileUtil()
    configuration = {
            "train_options" : {
                "learning_rate": float(args.learningRate),
                "optimizer" : OPTIMIZER.__name__,
                "batch_size": int(args.batchSize),
                "number_of_epochs": int(args.numberOfEpochs),
                "loss_parameters": {
                        "temperature": float(args.temperature),
                            "distillation_type": args.lossFunction,
                            "alpha" : float(args.alpha),
                            "beta" : float(args.beta)
                    }
                }
            }
    model_histories['config'] = configuration

    if(args.runKFold):
        #Get k folds
        k_folds =KFold(n_splits=int(args.ksplits))
        for fold, (train_idx,val_idx) in enumerate(k_folds.split(train_val_dataset)):

            if(args.runOn == "CIFAR100"):
                input_channels = 3
                output_channels = 100
                STUDENT_MODEL = ResNet34(num_classes=output_channels,input_channels=input_channels)
            
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

            model_information=train(train_dataset=train_dataset,
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
            file_helper.save_to_file(model_histories,args.saveHistoryFilePath+"/"+args.lossFunction+"_history_.json")
    else:
        if(args.runOn == "CIFAR100"):
            input_channels = 3
            output_channels = 100
            STUDENT_MODEL = ResNet34(num_classes=output_channels,input_channels=input_channels)
        
        if(args.runOn == "MNIST"):
            input_channels = 1
            output_channels = 10
            STUDENT_MODEL = DummyStudentModel(input_channels=input_channels,num_classes=output_channels)
        history = {'train_losses': [], 'valid_losses': [],'train_accs':[],'valid_accs':[], 'train_times':[], 'valid_times': [],'test_accuracy': None, 'convergence_iteration': None}
        #Split data
        train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset,split_percentage=float(args.trainValidSplit))
        model_information=train(train_dataset=train_dataset,
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
        file_helper.save_to_file(model_histories,args.saveHistoryFilePath+"/"+args.lossFunction+"_history_.json")
        

    

    

