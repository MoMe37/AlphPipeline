import sys
from matplotlib.pyplot import axis
import numpy as np
import os
import torch

from visualization import *
from alphsistant import *

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import itertools

class CustomSKDataset(Dataset):
    def __init__(self, csv_file):  
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.loc[idx, "sequence"]
        frame = self.data.loc[idx, "frame"]
        X = np.loadtxt("../AlphData/fadg0/spectrogram_norm/" + record + "/" + frame + ".txt")
        y = list(self.data.loc[idx, ["Basis.txt", "jaw_open.txt", "left_eye_closed.txt", "mouth_open.txt", "right_eye_closed.txt", "smile.txt", "smile_left.txt", "smile_right.txt"]])     
        X = torch.tensor(X)
        X = torch.split(X, 32)
        X = torch.stack(X, axis=0)
        y = torch.flatten(torch.tensor(y).float())     
        # X et y sont des tenseurs flat
        return X.float(), y


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
            nn.Linear(64 * 4 * 4, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 8),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)      
        output = self.seq(x)
        return output



def train(num_epochs, cnn, loaders):
    
    cnn.train()

    loss_values = []
    accuracy_test_values = []
    accuracy_train_values = []
        
    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        loss_cum = 0
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            output = cnn(images)              
            loss = loss_func(output, labels)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    

            # apply gradients             
            optimizer.step() 

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        loss_values.append(loss_cum/total_step)
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_step, total_step, loss.item()))

        cnn.eval()

        accuracy_test = 0
        accuracy_train = 0

        with torch.no_grad():

            for images, labels in loaders['test']:
                test_output = cnn(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                #accuracy_test += (pred_y == labels).sum().item() / float(labels.size(0))
            
            for images, labels in loaders['train']:
                train_output = cnn(images)
                pred_y = torch.max(train_output, 1)[1].data.squeeze()
                #accuracy_train += (pred_y == labels).sum().item() / float(labels.size(0))

        cnn.train()

        #accuracy_test /= len(loaders['test'])
        #accuracy_train /= len(loaders['train'])
        #accuracy_test_values.append(accuracy_test*100)
        #accuracy_train_values.append(accuracy_train*100)

    #x = []
    #for i in range(num_epochs):
    #    x.append(i+1)
    #plt.figure(figsize=(10,15))
    #plt.subplot(3, 1, 1)
    #plt.title('Loss function')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.plot(x ,np.array(loss_values), 'r')
    #plt.subplot(3, 1, 2)
    #plt.title('Accuracy test data')
    #plt.xlabel('Epoch')
    #plt.ylabel('%')
    #plt.plot(x ,np.array(accuracy_test_values), 'g')
    #plt.subplot(3, 1, 3)
    #plt.title('Accuracy train data')
    #plt.xlabel('Epoch')
    #plt.ylabel('%')
    #plt.plot(x ,np.array(accuracy_train_values), 'b')



def test(cnn, loaders):
    # Test the model
    all_preds = []
    all_labels = []
    cnn.eval()
    accuracy = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            test_output = cnn(images).numpy().flatten()
            #pred_y = torch.max(test_output, 1)[1].data.squeeze()
            #accuracy += (pred_y == labels).sum().item() / float(labels.size(0))

            all_preds.append(test_output)
            all_labels.append(labels.numpy().flatten())
            break

    #accuracy /= len(loaders['test'])
    print('Test Accuracy of the model on the {} test : {:.2f}%'.format(len(test_dataset), (accuracy*100)))
    return all_labels, all_preds



if __name__ == "__main__":

    filepath = "./alphsistant/data/ds_weights.csv"
    sk_dataset = CustomSKDataset(filepath)

    train_size = int(0.8 * len(sk_dataset)) 
    test_size = len(sk_dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(sk_dataset, [train_size, test_size])

    batch_size = 32
    loaders = {
        'train' : torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True),
        
        'test'  : torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True),
    }

    cnn = CNN()
    loss_func = nn.MSELoss()   
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001) 

    num_epochs = 10

    train(num_epochs, cnn, loaders)
    print('Model has been trained {} times with {} datas'.format(num_epochs, len(train_dataset)))

    all_labels, all_preds = test(cnn, loaders)

    torch.save(cnn.state_dict(), "model-cnn.pth")

    input_list = []
    for i in range(119):
        X = np.loadtxt("../AlphData/fadg0/spectrogram_norm/sa1/face_" + '{:03}'.format(i+1) + ".txt")
        X = torch.tensor(X)
        X = torch.split(X, 32)
        X = torch.stack(X, axis=0).float()
        input_list.append(X)
    input_list = torch.stack(input_list, axis=0)
    print("Input created")

    y = cnn(input_list)
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