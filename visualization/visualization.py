import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Iterator, Sequence

import numpy as np
import yaml
import pywavefront
import glob
from plotly.subplots import make_subplots
from plotly.graph_objs import Figure
import plotly.graph_objects as go

from .vector import Vec3f, Vector3D
from .config import ConfigFile
from .transformation import Transformation
from .correspondence import get_correspondence
from render.plot import BrowserVisualizer
from meshlib.mesh import Mesh

class Visualization:
    def __init__(self, nbr_subplot_x, nbr_subplot_y):
        specs = []
        self.filepath = []
        for i  in range(nbr_subplot_x):
            liste = []
            self.filepath.append([])
            for y in range(nbr_subplot_y):
                self.filepath[i].append('')
                liste.append({'type': 'surface'})
            specs.append(liste)
        self.fig = make_subplots(rows = nbr_subplot_x, cols = nbr_subplot_y, specs = specs)
        self.subplot_x = nbr_subplot_x
        self.subplot_y = nbr_subplot_y
        self.mesh_kwargs = dict(
            color='#666',
            opacity=1.0,
            flatshading=True,
            lighting=dict(
                ambient=0.1,
                diffuse=1.0,
                facenormalsepsilon=0.0000000000001,
                roughness=0.3,
                specular=0.7,
                fresnel=0.001
            ),
            lightposition=dict(
                x=-10000,
                y=10000,
                z=5000
            )
        
        
    )
    
    def afficher(self):
        self.fig.show()
    
    def update_fig(self, subplot_x, subplot_y, filepath):
        targetPattern = "/*.obj"
        path_list = glob.glob(filepath + targetPattern)
        
        mesh_list = []
        for i in range(len(path_list)):
            mesh_list.append(Mesh.load(path_list[i]))

        fig = Figure(
            data=[BrowserVisualizer.make_mesh(mesh_list[0], **self.mesh_kwargs)],
            layout=dict(
                updatemenus=[
                    dict(type="buttons",
                        buttons=[
                            dict(
                                label="Play figure " + str(subplot_x) + "," + str(subplot_y),
                                method="animate",
                                args=[None, {
                                    "mode": "afterall",
                                    "frame": {"duration": 40, "redraw": True},
                                    "fromcurrent": False,
                                    "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                                }]
                            )
                        ])
                ],
            ),
            frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list[i], **self.mesh_kwargs)]) for i in range(len(mesh_list))]
        )
        self.fig.add_trace(fig['data'][0], subplot_x, subplot_y)
        self.filepath[subplot_x-1][subplot_y-1] = filepath
    
    def animate(self):

        targetPattern = "/*.obj"
        frames = []
        meshlist = []
        for i in range(self.subplot_x):
            meshlist.append([])
            for y in range(self.subplot_y):
                path_list = glob.glob(self.filepath[i][y] + targetPattern)
                mesh_sub_list = []
                for n in range(len(path_list)):
                    mesh_sub_list.append(Mesh.load(path_list[n]))
                meshlist[i].append(mesh_sub_list)

        size_list = []
        for i in range(len(meshlist)):
            for y in range(len(meshlist[i])):
                size_list.append(len(meshlist[i][y]))

        for t in range(min(size_list)):
            data = []
            for i in range(self.subplot_x):
                for y in range(self.subplot_y):
                    data.append(BrowserVisualizer.make_mesh(meshlist[i][y][t], **self.mesh_kwargs))
            frames.append(go.Frame(data = data))

        self.fig.frames=frames
        button = dict(
                    label='Play',
                    method='animate',
                    args=[None, {
                                    "mode": "afterall",
                                    "frame": {"duration": 40, "redraw": True},
                                    "fromcurrent": False,
                                    "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                                }])
        self.fig.update_layout(updatemenus=[dict(type='buttons', buttons=[button])])

    def set_camera(self):
        camera = dict(
            up=dict(x=0, y=1, z=0)
        )
        scene = dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Z',
            zaxis_title='Y',
            camera=camera,
            dragmode='turntable'
        )
        self.fig.update_layout(
            scene=scene,
            scene2=scene,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis2=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0),
            # scene_camera=camera
        )

def make_animation(transf: Transformation, poses: Sequence[Mesh], mesh_list):
    assert poses
    results = [transf(pose) for pose in poses]

    mesh_kwargs = dict(
        color='#666',
        opacity=1.0,
        flatshading=True,
        lighting=dict(
            ambient=0.1,
            diffuse=1.0,
            facenormalsepsilon=0.0000000000001,
            roughness=0.3,
            specular=0.7,
            fresnel=0.001
        ),
        lightposition=dict(
            x=-10000,
            y=10000,
            z=5000
        )
    )

    fig = make_subplots(
    rows=1, cols=2, subplot_titles=('Prédiction', 'Retargeting'),
    specs=[[{'type': 'surface'}, {'type': 'surface'}]])

    fig1 = Figure(
        data=[BrowserVisualizer.make_mesh(mesh_list[0], **mesh_kwargs)],
        layout=dict(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(
                             label="Play",
                             method="animate",
                             args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }]
                         )
                     ])
            ],
        ),
        frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list[i], **mesh_kwargs)]) for i in range(len(mesh_list))]
    )

    fig2 = Figure(
        data=[BrowserVisualizer.make_mesh(results[0].transpose((0, 2, 1)), **mesh_kwargs)],
        layout=dict(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(
                             label="Play",
                             method="animate",
                             args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }]
                         )
                     ])
            ],
        ),
        frames=[go.Frame(data=[BrowserVisualizer.make_mesh(p.transpose((0, 2, 1)), **mesh_kwargs)]) for p in results]
    )

    fig.append_trace(fig1['data'][0], 1, 1)
    fig.append_trace(fig2['data'][0], 1, 2)

    frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list[i], **mesh_kwargs),
                            BrowserVisualizer.make_mesh(results[i].transpose((0, 2, 1)), **mesh_kwargs)])
                             for i in range(min(len(results), len(mesh_list)))]
    fig.frames=frames
    button = dict(
                 label='Play',
                 method='animate',
                 args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }])
    fig.update_layout(updatemenus=[dict(type='buttons', buttons=[button])])

    camera = dict(
        up=dict(x=0, y=1, z=0)
    )
    scene = dict(
        aspectmode='data',
        xaxis_title='X',
        yaxis_title='Z',
        zaxis_title='Y',
        camera=camera,
        dragmode='turntable'
    )
    fig.update_layout(
        scene=scene,
        scene2=scene,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        yaxis2=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0),
        # scene_camera=camera
    )
    return fig

def animate(vert_path, face_path, cfg: ConfigFile, identity=False):
    targetPattern = "/*.obj"
    path_list = glob.glob(vert_path + targetPattern)

    mesh_list = []
    for i in range(len(path_list)):
        mesh_list.append(Mesh.load(path_list[i]))

    corr_markers = cfg.markers  # List of vertex-tuples (source, target)
    if identity:
        corr_markers = np.ascontiguousarray(np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T)

    original_source = Mesh.load(cfg.source.reference)
    original_target = Mesh.load(cfg.target.reference)
    if identity:
        original_target = Mesh.load(cfg.source.reference)

    mapping = get_correspondence(original_source, original_target, corr_markers)
    transf = Transformation(original_source, original_target, mapping, smoothness=1)

    fig = make_animation(transf, list(cfg.source.load_poses()), mesh_list)
    fig.show(renderer="browser")

def double_animation(folder1, folder2):

    targetPattern = "/*.obj"
    path_list1 = glob.glob(folder1 + targetPattern)
    path_list2 = glob.glob(folder2 + targetPattern)

    mesh_list1 = []
    for i in range(len(path_list1)):
        mesh_list1.append(Mesh.load(path_list1[i]))

    mesh_list2 = []
    for i in range(len(path_list2)):
        mesh_list2.append(Mesh.load(path_list2[i]))

    mesh_kwargs = dict(
        color='#666',
        opacity=1.0,
        flatshading=True,
        lighting=dict(
            ambient=0.1,
            diffuse=1.0,
            facenormalsepsilon=0.0000000000001,
            roughness=0.3,
            specular=0.7,
            fresnel=0.001
        ),
        lightposition=dict(
            x=-10000,
            y=10000,
            z=5000
        )
    )

    fig = make_subplots(
    rows=1, cols=2, subplot_titles=('Animation 1', 'Animation 2'),
    specs=[[{'type': 'surface'}, {'type': 'surface'}]])

    fig1 = Figure(
        data=[BrowserVisualizer.make_mesh(mesh_list1[0], **mesh_kwargs)],
        layout=dict(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(
                             label="Play",
                             method="animate",
                             args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }]
                         )
                     ])
            ],
        ),
        frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list1[i], **mesh_kwargs)]) for i in range(len(mesh_list2))]
    )

    fig2 = Figure(
        data=[BrowserVisualizer.make_mesh(mesh_list2[0], **mesh_kwargs)],
        layout=dict(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(
                             label="Play",
                             method="animate",
                             args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }]
                         )
                     ])
            ],
        ),
        frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list2[i], **mesh_kwargs)]) for i in range(len(mesh_list2))]
    )

    fig.append_trace(fig1['data'][0], 1, 1)
    fig.append_trace(fig2['data'][0], 1, 2)

    frames=[go.Frame(data=[BrowserVisualizer.make_mesh(mesh_list1[i], **mesh_kwargs),
                            BrowserVisualizer.make_mesh(mesh_list2[i], **mesh_kwargs)])
                             for i in range(min(len(mesh_list1), len(mesh_list2)))]
    fig.frames=frames
    button = dict(
                 label='Play',
                 method='animate',
                 args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }])
    fig.update_layout(updatemenus=[dict(type='buttons', buttons=[button])])

    camera = dict(
        up=dict(x=0, y=1, z=0)
    )
    scene = dict(
        aspectmode='data',
        xaxis_title='X',
        yaxis_title='Z',
        zaxis_title='Y',
        camera=camera,
        dragmode='turntable'
    )
    fig.update_layout(
        scene=scene,
        scene2=scene,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        yaxis2=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0),
        # scene_camera=camera
    )
    fig.show()