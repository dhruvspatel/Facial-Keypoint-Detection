## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 5) # output size = (W-F)/S +1 = (224-5)/1 + 1 = 220/2 = (32, 110, 110)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 5) # output size = (64, 53, 53)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 3) # output size = (64, 25, 25)
        self.norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2)
        self.drop3 = nn.Dropout(p=0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 3) # output size = (64, 11, 11)
        self.norm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2,2)
        self.drop4 = nn.Dropout(p=0.4)
        
        self.fc1 = nn.Linear(256*11*11, 2000)
        self.norm5 = nn.BatchNorm1d(2000)
        self.drop5 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(2000, 500)
        self.norm6 = nn.BatchNorm1d(500)
        self.drop6 = nn.Dropout(p=0.6)
        
        self.fc3 = nn.Linear(500, 68*2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.drop1(self.pool1(F.relu(self.norm1(self.conv1(x)))))
        
        x = self.drop2(self.pool2(F.relu(self.norm2(self.conv2(x)))))
        
        x = self.drop3(self.pool3(F.relu(self.norm3(self.conv3(x)))))
        
        x = self.drop4(self.pool4(F.relu(self.norm4(self.conv4(x)))))
        
        
        # flatten
        
        x = x.view(x.size(0),-1)
        
        # fc layers
        
        x = self.drop5(self.norm5(self.fc1(x)))
        
        x = self.drop6(self.norm6(self.fc2(x)))
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        return x
