import torch
import torch.nn as nn
import torch.nn.functional as F



import numpy as np

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1, bias=False)
        self.conv2 = nn.Conv2d(32,64,3,padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 12 * 12,128)
        self.fc2 = nn.Linear(128,10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,stride=2,return_indices=True)[0]
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,stride=2,return_indices=True)[0]
        x = self.bn2(x)

        #Flatten
        x = x.view(x.shape[0],-1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.3, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x,p=0.3, training=self.training)

        return F.log_softmax(x,dim=-1)



class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN,self).__init__()
        self.conv   = nn.Conv2d(1,32,3,padding=1,bias=False)
        self.bn     = nn.BatchNorm2d(32)
        
        self.mp     = nn.MaxPool2d(2,stride=2)
        self.fc     = nn.Linear(32*14*14,10)
    
    def forward(self,x):
        x   = self.conv(x)
        x   = F.relu(x)
        x   = self.mp(x)
        x   = self.bn(x)
        x   = x.view(x.shape[0],-1)
        print(x.shape)
        x   = self.fc(x)
        x   = F.relu(x)

        return F.log_softmax(x,dim=-1)
