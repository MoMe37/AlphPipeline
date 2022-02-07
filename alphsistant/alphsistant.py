import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import phoneme_recognition.phoneme_recognition as phde

class CustomSKDataset(Dataset):
    def __init__(self, csv_file):  
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = list(data.loc[idx, ["a1", "o1", "c1", "g1", "l1", "b1", "f1", "t1", "j1", "u1", "q1", "a2", "o2", "c2", "g2", "l2", "b2", "f2", "t2", "j2", "u2", "q2","a3", "o3", "c3", "g3", "l3", "b3", "f3", "t3", "j3", "u3", "q3"]])
        y = list(data.loc[idx, ["Basis.txt", "jaw_open.txt", "left_eye_closed.txt", "mouth_open.txt", "right_eye_closed.txt", "smile.txt", "smile_left.txt", "smile_right.txt"]])     
        X = torch.flatten(torch.tensor(X).float()) 
        y = torch.flatten(torch.tensor(y).float())          
        # X et y sont des tenseurs flat
        return X, y

def data_normalization(y_train, y_test):
    values = []
    for i in range(len(y_train)):
        for y in range(len(y_train[i])):
            values.append(y_train[i][y])
    for i in range(len(y_test)):
        for y in range(len(y_test[i])):
            values.append(y_test[i][y])    
    mean = np.mean(values)
    st = np.std(values)
    for i in range(len(y_train)):
        for y in range(len(y_train[i])):
            y_train[i][y] = (y_train[i][y]-mean)/st
    for i in range(len(y_test)):
        for y in range(len(y_test[i])):
            y_test[i][y] = (y_test[i][y]-mean)/st
    return (y_train, y_test)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

def input_creation(phoneme_dataframe):
    df_frame =  pd.DataFrame(0, index = [], columns=['a1', 'o1', 'c1', 'g1', 'l1', 'b1', 'f1', 't1', 'j1', 'u1', 'q1', 'a2', 'o2', 'c2', 'g2', 'l2', 'b2', 'f2', 't2', 'j2', 'u2', 'q2','a3', 'o3', 'c3', 'g3', 'l3', 'b3', 'f3', 't3', 'j3', 'u3', 'q3'])

    for i in range(len(phoneme_dataframe)-3):
        for y in range(3):
            df_frame.loc[i, ['a'+ str(y+1), 'o'+ str(y+1), 'c'+ str(y+1), 'g'+ str(y+1), 'l'+ str(y+1), 'b'+ str(y+1), 'f'+ str(y+1), 't'+ str(y+1), 'j'+ str(y+1), 'u'+ str(y+1), 'q'+ str(y+1)]] = list(phoneme_dataframe.loc[i+y, ['a', 'o', 'c', 'g', 'l', 'b', 'f', 't', 'j', 'u', 'q']])
    X = []
    for i in range(len(df_frame)):
        X.append(list(df_frame.loc[i, ['a1', 'o1', 'c1', 'g1', 'l1', 'b1', 'f1', 't1', 'j1', 'u1', 'q1', 'a2', 'o2', 'c2', 'g2', 'l2', 'b2', 'f2', 't2', 'j2', 'u2', 'q2','a3', 'o3', 'c3', 'g3', 'l3', 'b3', 'f3', 't3', 'j3', 'u3', 'q3']]))
    X = torch.tensor(X).float()
    return(X)

def output_extraction(y, save_path):
    y = y.detach().numpy()
    output = []
    basis = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/Basis.txt')
    jaw_open = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/jaw_open.txt')
    left_eye_closed = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/left_eye_closed.txt')
    mouth_open = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/mouth_open.txt')
    right_eye_closed = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/right_eye_closed.txt')
    smile_left = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/smile_left.txt')
    smile_right = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/smile_right.txt')
    smile = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/smile.txt')

    

    for i in range(len(y)):
        output = y[i][0] * basis + y[i][1] * jaw_open + y[i][2] * left_eye_closed + y[i][3] * mouth_open + y[i][4] * right_eye_closed+ y[i][5] * smile + y[i][6] * smile_left + y[i][7] * smile_right 
        np.savetxt(save_path + "/face_" + '{:03}'.format(i) + ".txt", output)
    print("Extraction DONE")