import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, MNIST
from functions.utilities.device_loader import get_device, ToDeviceLoader,to_device
from functions.utilities.data_transformer import cifar_testing_augmentation, mnist_testing_augmentation
from models.dummy_teacher_model import DummyTeacherModel
import random
import numpy as np
from models.resnet import resnet110


RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

#Defaults
MODEL_PATH = './data/models/Resnet110.pt'
RUN_ON = "CIFAR100"
BATCH_SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', help='Path to the model', default=MODEL_PATH)
parser.add_argument('--batch-size', help='The batch size to backpropagate on', type=int, default=BATCH_SIZE)
parser.add_argument('--run-on', help='The data set to run on', default=RUN_ON)
args = parser.parse_args()


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
        test_dataset = CIFAR100(root='Data/', train=False,transform=cifar_testing_augmentation())
        TEACHER_MODEL = resnet110()
    
    if(args.run_on == "MNIST"):
        input_channels = 1
        output_channels = 10
        test_dataset = MNIST(root='Data/', train=False,transform=mnist_testing_augmentation())
        TEACHER_MODEL = DummyTeacherModel(num_classes=output_channels,input_channels=input_channels) 
    
    test_dl = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    test_data_on_specified_device = ToDeviceLoader(test_dl, device)
    
    try:
        teacher = to_device(TEACHER_MODEL,device=device)
        teacher.load(args.model_path)
    except:
        raise Exception("Could not load teacher model")
    
    _, correct = teacher.predict(test_data_on_specified_device,device)
    test_accuracy = correct / len(test_dataset) * 100 

    print("Test Accuracy:{:.3f}"
                    .format(test_accuracy))

   