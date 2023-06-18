import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, MNIST
from functions.loss.attention_knowledge_distillation_loss import AttentionKnowledgeDistillationLoss
from functions.utilities.device_loader import get_device, ToDeviceLoader, to_device
from functions.utilities.data_transformer import cifar_testing_augmentation, cifar_training_augmentation, mnist_testing_augmentation, mnist_training_augmentation
from functions.loss.filter_knowledge_distillation_loss import FilterKnowledgeDistillationLoss
from functions.loss.traditional_distillation_loss import TraditionalKnowledgeDistillationLoss
from functions.loss.loss_type import LossType
from models.dummy_teacher_model import DummyTeacherModel
from models.dummy_student_model import DummyStudentModel
from models.resnet import resnet110, resnet32
from training_loop import train
from functions.utilities.file_util import FileUtil
import random
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter
import torch.nn as nn

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
DISTILLATION_TYPE = 1
FILE_PATH_OF_TEACHER = './data/models/dummyParent.pt'
SAVE_HISTORY_FILE_PATH = "./data/"
ALPHA = 0.4
BETA = 0.3
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MINIMUM_DELTA = 0


MOMENTUM=0.9,
WEIGHT_DECAY=1e-4
SCHEDULE = [81,122]
GAMMA = 0.1


parser = argparse.ArgumentParser()
parser.add_argument('--number-of-epochs', help='Number of epochs to train for', type=int, default=NUMBER_OF_EPOCHS)
parser.add_argument('--learning-rate', help='The learning rate applied to the optimizer', type=float, default=LEARNING_RATE)
parser.add_argument('--batch-size', help='The batch size to backpropagate on', type=int, default=BATCH_SIZE)
parser.add_argument('--loss-function', help='The loss function to use', default=DISTILLATION_TYPE)
parser.add_argument('--run-on', help='The data set to run on', default=RUN_ON)
parser.add_argument('--run-kfold', help='Run training with kfold cross validation', type=bool, default=RUN_K_FOLD)
parser.add_argument('--k-splits', help='Number of K fold splits', type=int, default=K_SPLITS)
parser.add_argument('--train-valid-split', help='Split percentage of training and validation', type=float, default=TRAIN_VALID_SPLIT)
parser.add_argument('--temperature', help='Temperate to use for softmax', type=int, default=TEMPERATURE)
parser.add_argument('--alpha', help='Alpha to use for traditional knowledge distillation loss component', type=float, default=ALPHA)
parser.add_argument('--beta', help='Beta to use for knowledge distillation filter loss component', type=float, default=BETA)
parser.add_argument('--teacher-file-path', help='The path to the teacher model', default=FILE_PATH_OF_TEACHER)
parser.add_argument('--early-stopping-patience', help='The early stopping patience', type=int, default=EARLY_STOPPING_PATIENCE)
parser.add_argument('--early-stopping-minimum-delta', help='The early stopping minimum delta', type=float, default=EARLY_STOPPING_MINIMUM_DELTA)
parser.add_argument('--save-history-file-path', help='The path to save results', default=SAVE_HISTORY_FILE_PATH)

parser.add_argument('--schedule', type=int, nargs='+', default=SCHEDULE,
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=GAMMA, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=MOMENTUM, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=WEIGHT_DECAY, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()


LOSS_OPTIONS = {
      "temperature": args.temperature,
      "distillation_type": LossType(args.loss_function).name,
      "alpha" : args.alpha,
      "beta" : args.beta
}

EARLY_STOPING_OPTIONS = {
    "patience": args.early_stopping_patience,
    "min_delta": args.early_stopping_minimum_delta
}

SCHEDULER_OPTIONS = {
    "schedule": args.schedule,
    "gamma": args.gamma
}

OPTIMIZER_OPTIONS = {
    "optimizer": OPTIMIZER,
    "weight_decay": args.weight_decay,
    "momentum": args.momentum,
    "learning_rate": args.learning_rate
}

TRAIN_OPTIONS = { "batch_size": args.batch_size,
                  "number_of_epochs": args.number_of_epochs}


OPTIONS = { "optimizer_options" : OPTIMIZER_OPTIONS,
                  "train_options": TRAIN_OPTIONS,
                  "loss_options": LOSS_OPTIONS,
                  "early_stopping_options": EARLY_STOPING_OPTIONS,
                  "scheduler_options": SCHEDULER_OPTIONS}


def split_dataset(dataset, split_percentage):
    first_partition_size = int(split_percentage * len(dataset))
    second_partition_size = len(dataset) - first_partition_size
    first_partition, second_partition = torch.utils.data.random_split(dataset,[first_partition_size,second_partition_size])
    return first_partition, second_partition

def get_loss_function(loss_type, loss_options):
    loss_function = None
    if(LossType.FILTER.name == LossType(loss_type).name):
        loss_function = FilterKnowledgeDistillationLoss(options=loss_options)
    elif(LossType.TRADITIONAL.name == LossType(loss_type).name):
        loss_function = TraditionalKnowledgeDistillationLoss(options=loss_options)
    elif(LossType.ATTENTION.name == LossType(loss_type).name):
        loss_function = AttentionKnowledgeDistillationLoss(options=loss_options)
    elif(LossType.CE.name == LossType(loss_type).name):
        loss_function = nn.CrossEntropyLoss()
    return loss_function

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

    if(args.run_on == "CIFAR100"):
        input_channels = 3
        output_channels = 100
        train_val_dataset = CIFAR100(root='data/', train=True, download=True, transform=cifar_training_augmentation())
        test_dataset = CIFAR100(root='data/', train=False,download=True,transform=cifar_testing_augmentation())
        TEACHER_MODEL = resnet110()
    
    if(args.run_on == "MNIST"):
        input_channels = 1
        output_channels = 10
        train_val_dataset = MNIST(root='data/', train=True, download=True, transform=mnist_training_augmentation())
        test_dataset = MNIST(root='data/', train=False,download=True,transform=mnist_testing_augmentation())
        TEACHER_MODEL = DummyTeacherModel(num_classes=output_channels,input_channels=input_channels) 

    test_dl = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    test_data_on_specified_device = ToDeviceLoader(test_dl, device)
    
    try:
        teacher = to_device(TEACHER_MODEL,device=device)
        teacher.load(args.teacher_file_path)
    except:
        raise Exception("Could not load teacher model")
    

    model_histories={}
    file_helper = FileUtil()
    configuration = {
            "train_options" : {
                "learning_rate": args.learning_rate,
                "optimizer" : OPTIMIZER.__name__,
                "batch_size": args.batch_size,
                "number_of_epochs": args.number_of_epochs,
                "loss_parameters": {
                        "temperature": args.temperature,
                        "distillation_type": LossType(args.loss_function).name,
                        "alpha" : args.alpha,
                        "beta" : args.beta
                    }
                }
            }
    model_histories['config'] = configuration

    if(args.run_kfold):
        #Get k folds
        k_folds =KFold(n_splits=args.k_splits)
        for fold, (train_idx,val_idx) in enumerate(k_folds.split(train_val_dataset)):

            if(args.run_on == "CIFAR100"):
                input_channels = 3
                output_channels = 100
                STUDENT_MODEL = resnet32()
            
            if(args.run_on == "MNIST"):
                input_channels = 1
                output_channels = 10
                STUDENT_MODEL = DummyStudentModel(input_channels=input_channels,num_classes=output_channels)

            print('Fold {}'.format(fold + 1))
            #History for current fold
            history = {'train_losses': [], 'valid_losses': [],'train_accs':[],'valid_accs':[], 'train_times':[], 'valid_times': [],'test_accuracy': None, 'convergence_iteration': None}
            #Gets data
            train_dataset = itemgetter(*train_idx)(train_val_dataset)
            valid_dataset = itemgetter(*val_idx)(train_val_dataset)
            loss_function = get_loss_function(args.loss_function,LOSS_OPTIONS)

            model_information=train(train_dataset=train_dataset,
                    loss_function = loss_function,
                    valid_dataset = valid_dataset,
                    student_model=STUDENT_MODEL,
                    teacher_model=teacher,
                    device=device,
                    options = OPTIONS)
            
    
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
            file_helper.save_to_file(model_histories,args.save_history_file_path+"/"+LossType(args.loss_function).name.lower()+"_history_.json")
    else:
        if(args.run_on == "CIFAR100"):
            input_channels = 3
            output_channels = 100
            STUDENT_MODEL = resnet32()
        
        if(args.run_on == "MNIST"):
            input_channels = 1
            output_channels = 10
            STUDENT_MODEL = DummyStudentModel(input_channels=input_channels,num_classes=output_channels)
        history = {'train_losses': [], 'valid_losses': [],'train_accs':[],'valid_accs':[], 'train_times':[], 'valid_times': [],'test_accuracy': None, 'convergence_iteration': None}
        #Split data
        train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset,split_percentage=args.train_valid_split)
        loss_function = get_loss_function(args.loss_function, LOSS_OPTIONS)
        model_information=train(train_dataset=train_dataset,
                    valid_dataset = valid_dataset,
                    loss_function=loss_function,
                    student_model=STUDENT_MODEL,
                    teacher_model=teacher,
                    device=device,
                    options = OPTIONS)
        
            
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
        file_helper.save_to_file(model_histories,args.save_history_file_path+"/"+LossType(args.loss_function).name.lower()+"_history_.json")
        

    

    

