import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import librosa
import librosa.display
import pandas as pd
from pydub import AudioSegment

if __name__ == "__main__":

    #Récupération des données
    file = pd.read_csv("alphsistant/data/ds_weights.csv")
    file_jaw_open = file.loc[(file['sequence'] == 'sa1') & (abs(file['jaw_open.txt']) > 0.8)]
    file_mouth_open = file.loc[(file['sequence'] == 'sa1') & (abs(file['mouth_open.txt']) > 0.8)]
    file_smile = file.loc[(file['sequence'] == 'sa1') & (abs(file['smile.txt']) > 0.8)]
    file_smile_left = file.loc[(file['sequence'] == 'sa1') & (abs(file['smile_left.txt']) > 0.8)]
    file_smile_right = file.loc[(file['sequence'] == 'sa1') & (abs(file['smile_right.txt']) > 0.8)]

    #Affichage des données (jaw_open) 
    lt_spec_jaw_open = []
    lt_sk_jaw_open = []
    for i in range(len(file_jaw_open)) :
        X = np.loadtxt("../AlphData/fadg0/spectrogram/sa1/" + file_jaw_open.iloc[i, 10] + ".txt")
        lt_spec_jaw_open.append(X)
        lt_sk_jaw_open.append([file_jaw_open.iloc[i, 0], file_jaw_open.iloc[i, 1], file_jaw_open.iloc[i, 2], file_jaw_open.iloc[i, 3], file_jaw_open.iloc[i, 4], file_jaw_open.iloc[i, 5], file_jaw_open.iloc[i, 6], file_jaw_open.iloc[i, 7]])

    lt_spec_mouth_open = []
    lt_sk_mouth_open = []
    for i in range(len(file_mouth_open)) :
        X = np.loadtxt("../AlphData/fadg0/spectrogram/sa1/" + file_mouth_open.iloc[i, 10] + ".txt")
        lt_spec_mouth_open.append(X)
        lt_sk_mouth_open.append([file_mouth_open.iloc[i, 0], file_mouth_open.iloc[i, 1], file_mouth_open.iloc[i, 2], file_mouth_open.iloc[i, 3], file_mouth_open.iloc[i, 4], file_mouth_open.iloc[i, 5], file_mouth_open.iloc[i, 6], file_mouth_open.iloc[i, 7]])

    lt_spec_smile = []
    lt_sk_smile = []
    for i in range(len(file_smile)) :
        X = np.loadtxt("../AlphData/fadg0/spectrogram/sa1/" + file_smile.iloc[i, 10] + ".txt")
        lt_spec_smile.append(X)
        lt_sk_smile.append([file_smile.iloc[i, 0], file_smile.iloc[i, 1], file_smile.iloc[i, 2], file_smile.iloc[i, 3], file_smile.iloc[i, 4], file_smile.iloc[i, 5], file_smile.iloc[i, 6], file_smile.iloc[i, 7]])


    lt_lenght = [len(lt_spec_jaw_open), len(lt_spec_mouth_open), len(lt_spec_smile)]
    lenght = max(lt_lenght)

    plt.figure(figsize=(20, 12))

    for i in range(len(lt_spec_jaw_open)):

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*0)+1)
        plt.axis('off')
        plt.title('{:.3}'.format(lt_sk_jaw_open[i][1]))
        librosa.display.specshow(lt_spec_jaw_open[i])

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*1)+1)
        plt.axis('off')
        onset_env = librosa.onset.onset_strength(S=lt_spec_jaw_open[i])
        plt.plot(1 + onset_env / onset_env.max())

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*2)+1)
        plt.axis('off')
        plt.plot(librosa.feature.rms(S=lt_spec_jaw_open[i], frame_length=255)[0])

    for i in range(len(lt_spec_mouth_open)):

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*3)+1)
        plt.axis('off')
        plt.title('{:.3}'.format(lt_sk_mouth_open[i][3]))
        librosa.display.specshow(lt_spec_mouth_open[i])

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*4)+1)
        plt.axis('off')
        onset_env = librosa.onset.onset_strength(S=lt_spec_mouth_open[i])
        plt.plot(1 + onset_env / onset_env.max())

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*5)+1)
        plt.axis('off')
        plt.plot(librosa.feature.rms(S=lt_spec_mouth_open[i], frame_length=255)[0])

    for i in range(len(lt_spec_smile)):

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*6)+1)
        plt.axis('off')
        plt.title('{:.3}'.format(lt_sk_smile[i][5]))
        librosa.display.specshow(lt_spec_smile[i])
            
        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*7)+1)
        plt.axis('off')
        onset_env = librosa.onset.onset_strength(S=lt_spec_smile[i])
        plt.plot(1 + onset_env / onset_env.max())

        plt.subplot(len(lt_lenght)*3,lenght,i+(lenght*8)+1)
        plt.axis('off')
        plt.plot(librosa.feature.rms(S=lt_spec_smile[i], frame_length=255)[0])

    plt.show()