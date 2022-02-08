import visualization.visualization as vis
import pandas as pd

import numpy as np
import os

from phoneme_recognition import *
from alphsistant import *

df = pd.read_csv('./alphsistant/data/ds_weights.csv')

vertice_file_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/sk/sa1"
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
        np.savetxt(vertice_file_path + "/" + df.loc[i, 'frame'] + ".txt", output)

for filename in os.listdir(vertice_file_path):
        filename_we = os.path.splitext(filename)[0]
        with open(vertice_file_path + "/" + filename_we + ".obj", 'w+') as obj_file:
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

face_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/alphsistant_face_tris.txt"
audio_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/audio/sa1.wav"

df = phoneme_csv_creation(audio_path)
X = input_creation(df)
print("Input created")

model = torch.load('C:/Users/Enzo.Magal/Documents/Enzo2021/models/sk_model.pth')
print("Model structure: ", model, "/n\n")
y = model(X)
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


title = ('Vidéo', 'Shape Keys', 'Prédiction', 'Vidéo retargeting', 'Shape Keys retargeting', 'Prédiction retargeting')
visu = vis.Visualization(2, 3, title)

visu.update_fig(1, 1, 'C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/face_mesh/sa1')
visu.update_fig(1, 2, 'C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/sk/sa1')
visu.update_fig(1, 3, './prediction')
visu.update_fig_retargeting(2, 1, './alphsistant/data/suzanne_test/video_retargeting.yml')
visu.update_fig_retargeting(2, 2, './alphsistant/data/suzanne_test/sk_retargeting.yml')
visu.update_fig_retargeting(2, 3, './alphsistant/data/suzanne_test/prediction_retargeting.yml')

visu.animate()
visu.set_camera()
visu.afficher()