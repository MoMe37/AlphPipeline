import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=4,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),   
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,              
                out_channels=64,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),   
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(64 * 4 * 4, 8),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 64 * 16 * 16)
        # x = x.view(x.size(0), -1) 
        x = self.flatten(x)      
        output = self.seq(x)
        return output

saved_cnn = CNN()
saved_cnn.load_state_dict(torch.load("model.pth"))