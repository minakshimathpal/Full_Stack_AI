import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.fc=nn.Linear(32*2*2,10)
        # Block 1: Starting with 1 channel
        self.network  = nn.Sequential(
             #  layer 1
            nn.Conv2d(1, 12, kernel_size=3, padding=1), # RF 3x3  
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(p=0.025) ,

            nn.Conv2d(12, 12, kernel_size=3, padding=1), # RF 3x3  
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(p=0.025) ,

            #  layer 2
            nn.Conv2d(12, 16, kernel_size=1, padding=1), # RF 5x5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # RF 10x10
            nn.Dropout2d(p=0.025) ,

            #layer3
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # RF 12x12
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.025) ,
            #layer4
             nn.Conv2d(16, 32, kernel_size=3, padding=1), # RF 12x12
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.025) ,
            #layer5
            nn.Conv2d(32, 32, kernel_size=1, padding=1), # RF 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # RF 14x14
            nn.Dropout2d(p=0.025) ,            
            #  output layer
            nn.Conv2d(32, 32, 1),           
            nn.AvgPool2d(3)

        )


        
        
        # Block 3: Starting with 36 channels
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(36, 40, kernel_size=3, padding=1),
        #     nn.Conv2d(40, 44, kernel_size=3, padding=1),
        #     nn.Conv2d(44, 48, kernel_size=3, padding=1),
        #     nn.Conv2d(48, 52, kernel_size=3, padding=1),
        #     nn.Conv2d(52, 56, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )
        
        # Fully connected layers
       
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        output = self.network(x)
        output=output.view(batch_size,-1)
        output = self.fc(output)
        return F.log_softmax(output,dim=1)