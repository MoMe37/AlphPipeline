import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Iterator

import numpy as np
import yaml
import pywavefront

from .vector import Vec3f, Vector3D

@dataclass
class Mesh:
    """
    First simple data structure holding only the vertices and faces in a numpy array

    @param vertices th positions of triangle corners (x,y,z)
    @param faces the triangles (Triple of vertices indices)
    """
    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def from_pywavefront(cls, obj: pywavefront.Wavefront) -> "Mesh":
        """
        Load a mesh from a pywavefront object
        :param obj:
        :return:
        """
        assert obj.mesh_list
        return cls(
            vertices=np.array(obj.vertices),
            faces=np.array(obj.mesh_list[0].faces)
        )

    @classmethod
    def load_obj(cls, file: str, **kwargs) -> "Mesh":
        """
        Load a mesh from a .obj file
        :param file:
        :param kwargs:
        :return:
        """
        assert os.path.isfile(file), f"Mesh file is missing: {file}"
        kwargs.setdefault("encoding", "UTF-8")
        return cls.from_pywavefront(pywavefront.Wavefront(file, collect_faces=True, **kwargs))

    @classmethod
    def load_npz(cls, file: str, **kwargs) -> "Mesh":
        """
        Load a mesh from a numpy file .npz
        :param file:
        :param kwargs:
        :return:
        """
        assert os.path.isfile(file), f"Mesh file is missing: {file}"
        data = np.load(file)
        return cls(data["vertices"], data["faces"])

    @classmethod
    def load_txt(cls, vert_file: str, face_file: str, **kwargs) -> "Mesh":
        """
        Load a mesh from two text files .txt
        :param file:
        :param kwargs:
        :return:
        """
        faces = np.loadtxt(face_file)
        vertices = np.loadtxt(vert_file)
        return cls(vertices, faces)

    @classmethod
    def load(cls, file: str, **kwargs) -> "Mesh":
        if file.endswith(".obj") or file.endswith(".pose"):
            return cls.load_obj(file, **kwargs)
        elif file.endswith(".npz"):
            return cls.load_npz(file, **kwargs)
        raise ValueError("Invalid file format")

    def get_centroids(self) -> np.ndarray:
        return self.vertices[self.faces[:, :3]].mean(axis=1)

    def scale(self, factor: float):
        """
        Scale the mesh
        :param factor:
        :return:
        """
        self.vertices *= factor
        return self

    def box(self) -> Tuple[Vec3f, Vec3f]:
        """
        Get the bounding box
        :return:
        """
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def size(self) -> Vec3f:
        """
        Get the size of the mesh
        :return:
        """
        a, b = self.box()
        return b - a

    def move(self, offset: Vec3f):
        """
        Move the mesh
        :param offset:
        :return:
        """
        self.vertices += offset

    def span_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the triangle span components of each surface with the offset v1
        :return:
            Tuple of the three triangle spans
        """
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        c = (tmp.T / np.sqrt(np.linalg.norm(tmp, axis=1))).T
        return a, b, c

    @property
    def span(self) -> np.ndarray:
        """
        Calculates the triangle spans of each surface with the offset v1.
        The span components are ordered in columns.
        :return:
            triangles Nx3x3
        """
        a, b, c = self.span_components()
        return np.transpose((a, b, c), (1, 2, 0))

    @property
    def v1(self):
        return self.vertices[self.faces[:, 0]]

    def get_dimension(self) -> int:
        return self.faces.shape[1]

    def is_fourth_dimension(self) -> bool:
        return self.get_dimension() == 4

    def to_fourth_dimension(self, copy=True) -> "Mesh":
        if self.is_fourth_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        a, b, c = self.span_components()
        v4 = self.v1 + c
        new_vertices = np.concatenate((self.vertices, v4), axis=0)
        v4_indices = np.arange(len(self.vertices), len(self.vertices) + len(c))
        new_faces = np.concatenate((self.faces, v4_indices.reshape((-1, 1))), axis=1)
        return Mesh(new_vertices, new_faces)

    def is_third_dimension(self) -> bool:
        return self.faces.shape[1] == 3

    def to_third_dimension(self, copy=True) -> "Mesh":
        if self.is_third_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        new_faces = self.faces[:, :3]
        new_vertices = self.vertices[:np.max(new_faces) + 1]
        return Mesh(new_vertices, new_faces)

    def transpose(self, shape=(0, 1, 2)):
        shape = np.asarray(shape)
        assert shape.shape == (3,)
        return Mesh(
            vertices=self.vertices[:, shape],
            faces=self.faces
        )

    def normals(self) -> np.ndarray:
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        vns = np.cross(v2 - v1, v3 - v1)
        return (vns.T / np.linalg.norm(vns, axis=1)).T

class ConfigFile:
    """File that configures the both source & target models and the markers"""

    def __init__(self, file: str, cfg: Dict[str, Any]) -> None:
        assert "source" in cfg and isinstance(cfg["source"], dict)
        assert "target" in cfg and isinstance(cfg["target"], dict)
        self.file = file
        basepath = os.path.dirname(file)
        self.source = ModelConfig(cfg["source"], basepath)
        self.target = ModelConfig(cfg["target"], basepath)
        self.markers = self._load_markers(cfg.get("markers", None), basepath)

    @classmethod
    def _load_markers(cls, markers, basepath: str) -> np.ndarray:
        if not markers:
            return np.array([])
        elif isinstance(markers, dict):
            return np.array([(int(s), int(t)) for s, t in markers.items()], dtype=np.int)
        elif isinstance(markers, (list, tuple)):
            result: List[Tuple[int, int]] = []
            for e in markers:
                if isinstance(e, str):
                    s, t = e.split(":", maxsplit=1)
                    result.append((int(s), int(t)))
                else:
                    assert len(e) == 2
                    result.append((int(e[0]), int(e[1])))
            return np.array(result, dtype=np.int)
        elif isinstance(markers, str) and os.path.isfile(os.path.join(basepath, markers)):
            return np.asarray(get_markers(os.path.join(basepath, markers)), dtype=np.int)
        else:
            raise ValueError(f"invalid marker format: {type(markers)}")

    @classmethod
    def load(cls, file: str):
        with open(file, mode='rt') as fp:
            return cls(file, cfg=yaml.safe_load(fp))

    class Paths:
        class lowpoly:
            catdog = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization/models/lowpoly/markers-cat-dog.yml"
            catvoxel = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization/models/lowpoly/markers-cat-voxel.yml"

        class highpoly:
            cat_lion = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization/models/highpoly/markers-cat-lion.yml"
            horse_camel = "C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/visualization/models/highpoly/markers-horse-camel.yml"

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
    targetPattern = "/*.txt"
    path_list = glob.glob(vert_path + targetPattern)

    mesh_list = []
    for i in range(len(path_list)):
        mesh_list.append(Mesh.load_txt(vert_file = path_list[i], face_file = face_path))

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