import os

vertice_file_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/fadg0/face_mesh/sa1"
face_file_path = "C:/Users/Enzo.Magal/Documents/Enzo2022/AlphPipeline/alphsistant/data/alphsistant_face_tris.txt"

for filename in os.listdir(vertice_file_path):
    filename_we = os.path.splitext(filename)[0]
    with open(vertice_file_path + "/" + filename_we + ".obj", 'w+') as obj_file:
        obj_file.write("# obj {:s}\n\n".format(filename_we))
        obj_file.write("o {:s}\n\n".format(filename_we))
        with open(vertice_file_path + "/" + filename, 'r') as v_file:
            for v in v_file:
                array = [float(x) for x in v.split(' ')]
                obj_file.write("v {:.4f} {:.4f} {:.4f}\n".format(array[0], array[1], array[2]))
        print("Vertices done !")
        obj_file.write("\n")
        with open(face_file_path, 'r') as f_file:
            for f in f_file:
                array = [int(float(x)) for x in f.split(' ')]
                obj_file.write("f {:d} {:d} {:d}\n".format(array[0]+1, array[1]+1, array[2]+1))
            f_file.close()
        print("Faces done !")
        obj_file.close()