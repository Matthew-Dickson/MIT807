import torch
import argparse
from torchvision.datasets import CIFAR100, MNIST
from functions.utilities.device_loader import get_device
from functions.utilities.data_transformer import cifar_testing_augmentation, cifar_training_augmentation, mnist_testing_augmentation, mnist_training_augmentation
from functions.loss.filter_knowledge_distillation_loss import FilterKnowledgeDistillationLoss
from functions.loss.traditional_distillation_loss import TraditionalKnowledgeDistillationLoss
from functions.loss.loss_type import LossType
from models.dummy_teacher_model import DummyTeacherModel
from models.resnet import resnet110
from training_loop import train
import random
import numpy as np
import torch.nn as nn

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10000
EARLY_STOPPING_MINIMUM_DELTA = 0

OPTIMIZER = torch.optim.SGD
LEARNING_RATE = 0.1
TRAIN_VALID_SPLIT = 0.9

RUN_ON = "CIFAR100"
DISTILLATION_TYPE = 4
SAVE_TEACHER_PATH = './data/models/Resnet110test.pt'

MOMENTUM=0.9
WEIGHT_DECAY=1e-4
SCHEDULE = [81,122]
GAMMA = 0.1


parser = argparse.ArgumentParser()
parser.add_argument('--number-of-epochs', help='Number of epochs to train for', type=int, default=NUMBER_OF_EPOCHS)
parser.add_argument('--learning-rate', help='The learning rate applied to the optimizer', type=float, default=LEARNING_RATE)
parser.add_argument('--batch-size', help='The batch size to backpropagate on', type=int, default=BATCH_SIZE)
parser.add_argument('--loss-function', help='The loss function to use', default=DISTILLATION_TYPE)
parser.add_argument('--run-on', help='The data set to run on', default=RUN_ON)
parser.add_argument('--train-valid-split', help='Split percentage of training and validation', type=float, default=TRAIN_VALID_SPLIT)
parser.add_argument('--save-teacher-file-path', help='The path to the teacher model', default=SAVE_TEACHER_PATH)
parser.add_argument('--early-stopping-patience', help='The early stopping patience', type=int, default=EARLY_STOPPING_PATIENCE)
parser.add_argument('--early-stopping-minimum-delta', help='The early stopping minimum delta', type=float, default=EARLY_STOPPING_MINIMUM_DELTA)

parser.add_argument('--schedule', type=int, nargs='+', default=SCHEDULE,
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=GAMMA, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=MOMENTUM, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=WEIGHT_DECAY, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

EARLY_STOPING_OPTIONS = {
    "patience": args.early_stopping_patience,
    "min_delta": args.early_stopping_minimum_delta
}

LOSS_OPTIONS = {
      "distillation_type": LossType(args.loss_function).name
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
        loss_function = nn.CrossEntropyLoss()
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
        train_val_dataset = CIFAR100(root='Data/', train=True, download=True, transform=cifar_training_augmentation())
        test_dataset = CIFAR100(root='Data/', train=False,transform=cifar_testing_augmentation())
        TEACHER_MODEL = resnet110()
    
    if(args.run_on == "MNIST"):
        input_channels = 1
        output_channels = 10
        train_val_dataset = MNIST(root='Data/', train=True, download=True, transform=mnist_training_augmentation())
        test_dataset = MNIST(root='Data/', train=False,transform=mnist_testing_augmentation())
        TEACHER_MODEL = DummyTeacherModel(num_classes=output_channels,input_channels=input_channels) 

   
    train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset,split_percentage=args.train_valid_split)
    loss_function = get_loss_function(args.loss_function,LOSS_OPTIONS)
    model_information=train(train_dataset=train_dataset,
                    loss_function=loss_function,
                    valid_dataset = valid_dataset,
                    student_model=TEACHER_MODEL,
                    teacher_model=None,
                    device=device,
                    options = OPTIONS)
    model = model_information["model"]
    model.save(args.save_teacher_file_path)
    
    
    

    

