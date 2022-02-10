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
        
        y = list(data.loc[idx, ["Basis.txt", "jaw_open.txt", "left_eye_closed.txt", "mouth_open.txt", "right_eye_closed.txt", "smile.txt", "smile_left.txt", "smile_right.txt"]])     
        X = torch.flatten(sample) 
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

def mel_spectrogram_creation(audio_path):
    samples, sample_rate = librosa.load(audio_path, sr=None) #on charge le son 
    spectrogram = librosa.stft(samples) #short time fourier transform
    sgram_mag, _ = librosa.magphase(spectrogram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    return mel_sgram, sample_rate

def mel_spec_sample(mel_spectrogram, window_size, nbr_sample):
    sample_list = []
    for i in range(nbr_sample-1):
        sample = mel_spectrogram[:,int(i*(len(mel_spectrogram[0])/nbr_sample)):int(5+i*(len(mel_spectrogram[0])/nbr_sample))]
        tens = torch.tensor(sample)
        sample_list.append(tens)
    sample = mel_spectrogram[:,len(mel_spectrogram[0])-5:len(mel_spectrogram[0])]
    tens = torch.tensor(sample).flatten
    sample_list.append(tens)
    return sample_list

def input_data_creation(sk_df):
    files = ['sa1', 'sa2']
    for record in files:
        filepath = "../fadg0/audio/" + record + ".wav"
        mel_spec, sample_rate = mel_spectrogram_creation(filepath)
        sample = mel_spec_sample(mel_spec, 103)
        sk_weights = []
        for i in np.where(sk_df.loc[:,'sequence'] == record):
            sk_weights.append(list(sk_df.loc[i, ["Basis.txt", "jaw_open.txt", "left_eye_closed.txt", "mouth_open.txt", "right_eye_closed.txt", "smile.txt", "smile_left.txt", "smile_right.txt"]]))
        print(len(sk_weights))