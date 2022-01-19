import sys
import numpy as np
import os
import torch


sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization")
from visualization.config import ConfigFile
from animation import animate
from meshlib import Mesh

sys.path.append("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/model_testing")
from model_test import input_creation, output_extraction
import phoneme_detection as phde

if __name__ == "__main__":
    face_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/alphsistant_face_tris.txt"
    audio_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/audio/sa1.wav"

    df = phde.phoneme_csv_creation(audio_path)
    X = input_creation(df)
    print("Input created")

    model = torch.load('C:/Users/Enzo.Magal/Documents/Enzo2021/models/sk_model.pth')
    print("Model structure: ", model, "/n\n")
    y = model(X)
    print("Output computed")

    vert_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/prediction"
    files=os.listdir(vert_path)
    for i in range(0,len(files)):
        os.remove(vert_path+'/'+files[i])
    output_extraction(y,vert_path)
    print("Output extracted !")

    vertice_file_path = "./prediction"
    face_file_path = "./alphsistant_face_tris.txt"

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

    cfg = ConfigFile.load("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/suzanne_test/markers_test.yml")
    #cfg = ConfigFile.load("C:/Users/Enzo.Magal/Documents/Enzo2021/alphsistant_code/deformation_external/models/lowpoly/markers-cat-voxel.yml")
    animate(vert_path, face_path, cfg)