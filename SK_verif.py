from visualization.visualization import double_animation
import pandas as pd

import numpy as np
import os

df = pd.read_csv('./alphsistant/data/ds_weights.csv')

vertice_file_path = "./prediction"
files=os.listdir(vertice_file_path)
for i in range(0,len(files)):
    os.remove(vertice_file_path+'/'+files[i])

basis = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/Basis.txt')
jaw_open = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/jaw_open.txt')
left_eye_closed = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/left_eye_closed.txt')
mouth_open = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/mouth_open.txt')
right_eye_closed = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/right_eye_closed.txt')
smile_left = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/smile_left.txt')
smile_right = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/smile_right.txt')
smile = np.loadtxt('C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/shape_keys_v0/smile.txt')

for i in range(len(df)):
    if df.loc[i, 'sequence'] == 'sa1':
        y = df.loc[i, ['Basis.txt', 'jaw_open.txt', 'left_eye_closed.txt', 'mouth_open.txt', 'right_eye_closed.txt', 'smile.txt', 'smile_left.txt', 'smile_right.txt']]
        output = y[0] * basis + y[1] * jaw_open + y[2] * left_eye_closed + y[3] * mouth_open + y[4] * right_eye_closed + y[5] * smile + y[6] * smile_left + y[7] * smile_right
        np.savetxt("./prediction/" + df.loc[i, 'frame'] + ".txt", output)

for filename in os.listdir('./prediction'):
        filename_we = os.path.splitext(filename)[0]
        with open("./prediction/" + filename_we + ".obj", 'w+') as obj_file:
            obj_file.write("# obj {:s}\n\n".format(filename_we))
            obj_file.write("o {:s}\n\n".format(filename_we))
            with open(vertice_file_path + "/" + filename, 'r') as v_file:
                for v in v_file:
                    array = [float(x) for x in v.split(' ')]
                    obj_file.write("v {:.4f} {:.4f} {:.4f}\n".format(array[0], array[1], array[2]))
            obj_file.write("\n")
            with open('./alphsistant/data/alphsistant_face_tris.txt', 'r') as f_file:
                for f in f_file:
                    array = [int(float(x)) for x in f.split(' ')]
                    obj_file.write("f {:d} {:d} {:d}\n".format(array[0]+1, array[1]+1, array[2]+1))
                f_file.close()
            obj_file.close()

folder1 = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/face_mesh/sa1"
folder2 = "./prediction"
double_animation(folder1, folder2)