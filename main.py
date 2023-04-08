import torch
import time
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100
from sklearn.model_selection import KFold
from Data.Utilities.device_loader import get_device, ToDeviceLoader, to_device
from Data.Utilities.data_transformer import trainingAugmentation
from Models.Blocks.ResidualBlock import ResidualBlock
from Models.ResNet import ResNet
from Models.DummyTeacherModel import DummyTeacherModel

BATCH_SIZE = 128
K_SPLITS = 10
NUMBER_OF_EPOCHS = 10
RANDOM_STATE = 42
OPTIMIZER = torch.optim.Adam
CRITERION = torch.nn.CrossEntropyLoss()
teacher_model_number = 3
# teacher_model_number = 18 


def train_epoch(model,dataloader,loss_fn,optimizer,device="cpu"):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct
  
def valid_epoch(model,dataloader,loss_fn,device="cpu"):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss=loss_fn(output,labels)
            valid_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct

if __name__ == '__main__':

    #device
    device = get_device()

    #loading data 
    train_dataset = CIFAR100(root='Data/', train=True, download=True, transform=trainingAugmentation())
    test_dataset = CIFAR100(root='Data/', train=False)
    # dataset = ConcatDataset([train_dataset, test_dataset])
    
    test_dl = ToDeviceLoader(DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False), device)
    

    model_histories={}
    #Get k folds
    k_folds =KFold(n_splits=K_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    for fold, (train_idx,val_idx) in enumerate(k_folds.split(torch.arange(len(train_dataset)))):
         print('Fold {}'.format(fold + 1))

         #model = to_device(ResNet(ResidualBlock, [teacher_model_number, teacher_model_number, teacher_model_number]), device)
         model = to_device(DummyTeacherModel(), device)
         optimizer =  OPTIMIZER(model.parameters(), lr=0.002)
        
         #History for current fold
         history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[], 'train_time':[], 'valid_time': []}
         # Initializes training and validation data loaders
         train_sampler = SubsetRandomSampler(train_idx)
         validation_sampler = SubsetRandomSampler(val_idx)

         train_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
         validation_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

         train_data_gpu = ToDeviceLoader(train_dl, device)
         validation_data_gpu = ToDeviceLoader(validation_dl, device)
         
         for epoch in range(NUMBER_OF_EPOCHS):
             t0 = time.time()
             train_loss, train_correct=train_epoch(model,train_data_gpu,CRITERION,optimizer, device)
             t1 = time.time()
             validation_loss, validation_correct=valid_epoch(model,validation_data_gpu,CRITERION, device)
             t2 = time.time()
            #  train_loss, train_correct=0,0
            #  validation_loss, validation_correct=0,0

             avg_train_loss_per_epoch = train_loss / len(train_dl.sampler)
             avg_train_acc_per_epoch = train_correct / len(train_dl.sampler) * 100
             avg_validation_loss_per_epoch = validation_loss / len(validation_dl.sampler)
             avg_validation_acc_per_epoch = validation_correct / len(validation_dl.sampler) * 100

             print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} % Training Time Taken: {} Seconds Validation Time Taken {} Seconds"
                   .format(epoch + 1,
                    NUMBER_OF_EPOCHS,
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
             history['train_time'].append(avg_validation_acc_per_epoch)
             history['valid_time'].append(avg_validation_acc_per_epoch)
        
         model_histories['fold{}'.format(fold+1)] = history 

