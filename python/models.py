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

        self.fc1 = nn.Linear(64 * 7 * 7,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,stride=2)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,stride=2)
        x = self.bn2(x)

        #Flatten
        x = x.view(x.shape[0],-1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return F.log_softmax(x,dim=-1)
