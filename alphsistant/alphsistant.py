import pandas as pd
import numpy as np
import librosa

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
        record = self.data.loc[idx, "sequence"]
        frame = self.data.loc[idx, "frame"]
        X = np.loadtxt("../AlphData/fadg0/spectrogram/" + record + "/" + frame + ".txt", dtype=float)
        y = list(self.data.loc[idx, ["Basis.txt", "jaw_open.txt", "left_eye_closed.txt", "mouth_open.txt", "right_eye_closed.txt", "smile.txt", "smile_left.txt", "smile_right.txt"]])     
        X = torch.tensor(X)
        X = torch.split(X, 32)
        y = torch.flatten(torch.tensor(y).float())          
        # X et y sont des tenseurs flat
        return X, y

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

def output_extraction(y, save_path):
    y = y.detach().numpy()
    output = []
    basis = np.loadtxt('../../AlphData/shape_keys_v0/Basis.txt')
    jaw_open = np.loadtxt('../../AlphData/shape_keys_v0/jaw_open.txt')
    left_eye_closed = np.loadtxt('../../AlphData/shape_keys_v0/left_eye_closed.txt')
    mouth_open = np.loadtxt('../../AlphData/shape_keys_v0/mouth_open.txt')
    right_eye_closed = np.loadtxt('../../AlphData/shape_keys_v0/right_eye_closed.txt')
    smile_left = np.loadtxt('../../AlphData/shape_keys_v0/smile_left.txt')
    smile_right = np.loadtxt('../../AlphData/shape_keys_v0/smile_right.txt')
    smile = np.loadtxt('../../AlphData/shape_keys_v0/smile.txt')

    

    for i in range(len(y)):
        output = y[i][0] * basis + y[i][1] * jaw_open + y[i][2] * left_eye_closed + y[i][3] * mouth_open + y[i][4] * right_eye_closed+ y[i][5] * smile + y[i][6] * smile_left + y[i][7] * smile_right 
        np.savetxt(save_path + "/face_" + '{:03}'.format(i) + ".txt", output)
    print("Extraction DONE")