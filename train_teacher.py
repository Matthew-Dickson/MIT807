import torch
import argparse
from torchvision.datasets import CIFAR100, MNIST
from data.utilities.device_loader import get_device
from data.utilities.data_transformer import trainingAugmentation
from functions.loss.filter_knowledge_distillation_loss import FilterKnowledgeDistillationLoss
from functions.loss.traditional_distillation_loss import TraditionalKnowledgeDistillationLoss
from functions.loss_type import LossType
from models.DummyTeacherModel import DummyTeacherModel
from models.ResNet import resnet110
from training_scheme import train
import random
import numpy as np
import torch.nn as nn

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 164
EARLY_STOPPING_PATIENCE = 10000
EARLY_STOPPING_MINIMUM_DELTA = 0

OPTIMIZER = torch.optim.SGD
LEARNING_RATE = 0.1
TRAIN_VALID_SPLIT = 0.9

RUN_ON = "CIFAR100"
DISTILLATION_TYPE = 4
SAVE_TEACHER_PATH = './data/models/Resnet110.pt'


parser = argparse.ArgumentParser()
parser.add_argument('--numberOfEpochs', help='Number of epochs to train for', default=NUMBER_OF_EPOCHS)
parser.add_argument('--learningRate', help='The learning rate applied to the optimizer', default=LEARNING_RATE)
parser.add_argument('--batchSize', help='The batch size to backpropagate on', default=BATCH_SIZE)
parser.add_argument('--lossFunction', help='The loss function to use', default=DISTILLATION_TYPE)
parser.add_argument('--runOn', help='The data set to run on', default=RUN_ON)
parser.add_argument('--trainValidSplit', help='Split percentage of training and validation', default=TRAIN_VALID_SPLIT)
parser.add_argument('--saveTeacherFilePath', help='The path to save results', default=SAVE_TEACHER_PATH)
parser.add_argument('--earlyStoppingPatience', help='The early stopping patience', default=EARLY_STOPPING_PATIENCE)
parser.add_argument('--earlyStoppingMinimumDelta', help='The early stopping minimum delta', default=EARLY_STOPPING_MINIMUM_DELTA)
args = parser.parse_args()


EARLY_STOPING_OPTIONS = {
    "patience": int(args.earlyStoppingPatience),
    "min_delta": float(args.earlyStoppingMinimumDelta)
}

LOSS_OPTIONS = {
      "distillation_type": LossType(args.lossFunction).name
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

    if(args.runOn == "CIFAR100"):
        input_channels = 3
        output_channels = 100
        train_val_dataset = CIFAR100(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = CIFAR100(root='Data/', train=False,transform=trainingAugmentation())
        TEACHER_MODEL = resnet110()
    
    if(args.runOn == "MNIST"):
        input_channels = 1
        output_channels = 10
        train_val_dataset = MNIST(root='Data/', train=True, download=True, transform=trainingAugmentation())
        test_dataset = MNIST(root='Data/', train=False,transform=trainingAugmentation())
        TEACHER_MODEL = DummyTeacherModel(num_classes=output_channels,input_channels=input_channels) 

   
    train_dataset, valid_dataset = split_dataset(dataset=train_val_dataset,split_percentage=args.trainValidSplit)
    loss_function = get_loss_function(int(args.lossFunction),LOSS_OPTIONS)
    model_information=train(train_dataset=train_dataset,
                    loss_function=loss_function,
                    valid_dataset = valid_dataset,
                    student_model=TEACHER_MODEL,
                    teacher_model=None,
                    device=device,
                    train_options = TRAIN_OPTIONS)
    model = model_information["model"]
    model.save(args.saveTeacherFilePath)
    
    
    

    

