from Models.BaseModel import BaseModel
from Models.Layers.Convolution import conv3x3
import torch.nn as nn


class DummyTeacherModel(BaseModel):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv = conv3x3(3, 32)  # 3 = red, greed, blue
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(32, 16)
        self.relu2 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(16)
        self.fc = nn.Linear(64, num_classes)

        # self.network = nn.Sequential(
        #     conv3x3(3, 16) ,
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
        
        #     conv3x3(3, 32),
        #     nn.ReLU(),
        #     nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
            
        #     conv3x3(3, 64),
        #     nn.ReLU(),
        #     nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2),
            
        #     nn.Flatten(),
        #     nn.Linear(82944,1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(64,num_classes)
        # )
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out