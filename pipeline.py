import sys
from matplotlib.pyplot import axis
import numpy as np
import os
import torch

from visualization import *
from alphsistant import *

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

if __name__ == "__main__":
    
    input_list = []
    for i in range(119):
        X = np.loadtxt("../AlphData/fadg0/spectrogram/sa1/face_" + '{:03}'.format(i+1) + ".txt")
        X = torch.tensor(X)
        X = torch.split(X, 32)
        X = torch.stack(X, axis=0).float()
        input_list.append(X)
    input_list = torch.stack(input_list, axis=0)
    print("Input created")

    model = CNN()
    model.load_state_dict(torch.load('./model_creation/model.pth'))
    y = model(input_list)
    print("Output computed")

    vertice_file_path = "./prediction"
    files=os.listdir(vertice_file_path)
    for i in range(0,len(files)):
        os.remove(vertice_file_path+'/'+files[i])
    output_extraction(y,vertice_file_path)
    print("Output extracted !")

    face_file_path = "./alphsistant/data/alphsistant_face_tris.txt"

    for filename in os.listdir(vertice_file_path):
        filename_we = os.path.splitext(filename)[0]
        with open("./prediction/" + filename_we + ".obj", 'w+') as obj_file:
            obj_file.write("# obj {:s}\n\n".format(filename_we))
            obj_file.write("o {:s}\n\n".format(filename_we))
            with open(vertice_file_path + "/" + filename, 'r') as v_file:
                for v in v_file:
                    array = [float(x) for x in v.split(' ')]
                    obj_file.write("v {:.4f} {:.4f} {:.4f}\n".format(array[0], array[1], array[2]))
            obj_file.write("\n")
            with open(face_file_path, 'r') as f_file:
                for f in f_file:
                    array = [int(float(x)) for x in f.split(' ')]
                    obj_file.write("f {:d} {:d} {:d}\n".format(array[0]+1, array[1]+1, array[2]+1))
                f_file.close()
            obj_file.close()

    title = ('Video', 'Shape Keys', 'Prediction')
    visu = Visualization(1, 3, title)

    visu.update_fig(1, 1, '../AlphData/fadg0/face_mesh/sa1')
    visu.update_fig(1, 2, '../AlphData/fadg0/sk/sa1')
    visu.update_fig(1, 3, './prediction')

    visu.animate()
    visu.set_camera()
    visu.afficher()