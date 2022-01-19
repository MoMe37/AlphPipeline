import pandas as pd
import numpy as np

import torch

import phoneme_detection as phde

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
        output = y[i][0] * basis + y[i][1] * jaw_open + y[i][2] * left_eye_closed + y[i][3] * mouth_open + y[i][4] * right_eye_closed + y[i][5] * smile_left + y[i][6] * smile_right + y[i][7] * smile
        np.savetxt(save_path + "/face_" + '{:03}'.format(i) + ".txt", output)
    print("Extraction DONE")