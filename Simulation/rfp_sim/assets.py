"""
Meshes for the simulation: the catfish (a .3mf or .stl, or a plain stand-in shape) and the
FR5's links: FAIRINO's own STL files from frcobot_ros2 (a ROS install, a configured folder,
or downloaded once and kept in Simulation/models/fr5/), else plain shapes.
"""

import hashlib
import math
import os
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv

UNITS = {"micron": 1e-6, "millimeter": 1e-3, "centimeter": 1e-2, "inch": 0.0254,
         "foot": 0.3048, "meter": 1.0}  # fmt: skip
AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
        "+z": (0, 0, 1), "-z": (0, 0, -1)}  # fmt: skip
OUTLINE_MM = 2.0  # resolution of the top-view outline


@dataclass
class FishModel:
    """A fish in its own frame: x toward the head, z up, origin at the center of its top-view
    outline (where vision puts the grasp point), z = 0 at its lowest point."""

    mesh: pv.PolyData
    outline: np.ndarray  # (n, 2) top-view outline (m)
    length: float  # m, tip to tip along x
    source: str
    # (3,) m: where it balances, for a uniform density (the closed mesh's volume centroid)
    center_of_mass: np.ndarray | None = None


def read_3mf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices in meters, triangles) of every mesh object of a 3MF file, as built."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".model"))
        root = ET.fromstring(archive.read(name))
    ns = "{" + root.tag.split("}")[0].strip("{") + "}"
    scale = UNITS.get(root.get("unit", "millimeter"), 1e-3)
    meshes = {}
    for obj in root.iter(f"{ns}object"):
        mesh = obj.find(f"{ns}mesh")
        if mesh is not None:
            vertices = [[float(v.get(a)) for a in "xyz"] for v in mesh.find(f"{ns}vertices")]
            triangles = [[int(t.get(a)) for a in ("v1", "v2", "v3")]
                         for t in mesh.find(f"{ns}triangles")]  # fmt: skip
            meshes[obj.get("id")] = (np.array(vertices), np.array(triangles))
    items = list(root.iter(f"{ns}item")) or [ET.Element("item", objectid=k) for k in meshes]
    all_vertices, all_triangles, offset = [], [], 0
    for item in items:
        if item.get("objectid") not in meshes:
            continue  # a component assembly: not used by slicer exports of one part
        vertices, triangles = meshes[item.get("objectid")]
        if item.get("transform"):  # 3x4 affine, row vectors: v' = [v 1] @ M
            m = np.array([float(x) for x in item.get("transform").split()]).reshape(4, 3)
            vertices = vertices @ m[:3] + m[3]
        all_vertices.append(vertices)
        all_triangles.append(triangles + offset)
        offset += len(vertices)
    if not all_vertices:
        raise ValueError(f"{path.name}: no mesh in the file")
    return np.vstack(all_vertices) * scale, np.vstack(all_triangles)


def polydata(vertices: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
    return pv.PolyData(np.asarray(vertices, float), faces)


def load_fish(path: Path | None, length: float, head: str = "-x", up: str = "+z") -> FishModel:
    """The fish of a .3mf/.stl file turned into the fish frame and scaled to `length`; the
    plain stand-in shape if `path` is None or missing."""
    if path is None or not path.is_file():
        return stand_in_fish(length)
    if path.suffix.lower() == ".3mf":
        vertices, triangles = read_3mf(path)
    else:
        mesh = pv.read(path).triangulate()
        vertices, triangles = np.asarray(mesh.points), mesh.faces.reshape(-1, 4)[:, 1:]
    x, z = np.array(AXES[head], float), np.array(AXES[up], float)
    if abs(x @ z) > 1e-9:
        raise ValueError(f"fish_head {head} and fish_up {up} must be different axes")
    rotation = np.array([x, np.cross(z, x), z])  # rows: the fish frame's axes in the model
    vertices = vertices @ rotation.T
    vertices = vertices * (length / np.ptp(vertices[:, 0]))
    return _centered(vertices, triangles, path.name)


def _centered(vertices: np.ndarray, triangles: np.ndarray, source: str) -> FishModel:
    """Origin at the top-view outline's centroid, on the table."""
    low = vertices[:, :2].min(axis=0)
    pixels = np.round((vertices[:, :2] - low) * 1000 / OUTLINE_MM).astype(np.int32)
    size = pixels.max(axis=0) + 3
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(mask, list(pixels[triangles]), 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2) * OUTLINE_MM / 1000 + low
    moments = cv2.moments(mask, binaryImage=True)
    center = np.array([moments["m10"], moments["m01"]]) / moments["m00"] * OUTLINE_MM / 1000 + low
    vertices = vertices - [center[0], center[1], vertices[:, 2].min()]
    return FishModel(
        polydata(vertices, triangles), outline - center, float(np.ptp(vertices[:, 0])), source,
        _center_of_mass(vertices, triangles),
    )  # fmt: skip


def _center_of_mass(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Volume centroid of a closed triangle mesh: a tetrahedron from the origin to each
    triangle, signed. An open mesh has no volume: then the vertices' mean."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = volumes.sum()
    if abs(total) < 1e-9:
        return vertices.mean(axis=0)
    return ((a + b + c) / 4.0 * volumes[:, None]).sum(axis=0) / total


@dataclass
class GripperModel:
    """The gripper (cups, plate, camera and its bracket) in the FR5's tip link frame: z out
    of the flange toward the cups, the flange face at z = FLANGE_Z."""

    mesh: pv.PolyData  # m
    tcp_z: float  # m: the plane of the cup faces (the cups point along +z)
    cups: np.ndarray  # (n, 2) m: cup centers in x, y
    cup_radius: float  # m


FLANGE_Z = 0.10  # the FR5's flange face in its last link's frame
GRIPPER_TRIANGLES = 100_000  # an imported gripper is cut down to this many (file size, speed)


def import_gripper(
    mesh: pv.PolyData,
    units: str = "millimeter",
    up: str = "+z",
    turn_deg: float = 0.0,
    triangles: int = GRIPPER_TRIANGLES,
) -> GripperModel:
    """
    A gripper from its CAD file into the tip link frame. `up`: the CAD axis from the cups
    toward the flange. The gripper's end farthest along it is the face bolted to the
    flange; its middle goes on the flange's axis. `turn_deg` turns the gripper about that
    axis, as bolting it on another way round would.
    """
    mesh = mesh.triangulate().scale(UNITS[units], inplace=False)
    toward_arm = np.array(AXES[up], float)
    if toward_arm[2] > 0.5:  # cups down in the CAD file: turn it over about y
        rotation = np.diag([-1.0, 1.0, -1.0])
    else:  # the shortest turn taking `up` to -z
        axis = np.cross(toward_arm, (0.0, 0.0, -1.0))
        angle = math.acos(float(np.clip(-toward_arm[2], -1.0, 1.0)))
        rotation = _axis_angle(axis / np.linalg.norm(axis), angle)
    turn = math.radians(turn_deg)
    rotation = _axis_angle(np.array([0.0, 0.0, 1.0]), turn) @ rotation
    points = np.asarray(mesh.points) @ rotation.T
    face = points[points[:, 2] < points[:, 2].min() + 0.0005]
    middle = (face[:, :2].min(axis=0) + face[:, :2].max(axis=0)) / 2
    mesh.points = points - [middle[0], middle[1], points[:, 2].min() - FLANGE_Z]
    if mesh.n_cells > triangles:
        mesh = mesh.decimate(1.0 - triangles / mesh.n_cells)
    return measure_gripper(mesh)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def measure_gripper(mesh: pv.PolyData) -> GripperModel:
    """The cups of a gripper mesh in the tip link frame: each cup lip's ring in the plane
    farthest along z."""
    tcp_z = float(mesh.points[:, 2].max())
    lips = mesh.clip("z", origin=(0, 0, tcp_z - 0.001), invert=False)
    lips = lips.extract_surface(algorithm="dataset_surface").connectivity("all")
    region = lips.point_data["RegionId"]
    cups, radii = [], []
    for i in np.unique(region):
        ring = np.asarray(lips.points[region == i])[:, :2]
        center = (ring.min(axis=0) + ring.max(axis=0)) / 2
        cups.append(center)
        radii.append(float(np.linalg.norm(ring - center, axis=1).max()))
    order = np.lexsort(np.array(cups).T[::-1]) if cups else []
    return GripperModel(mesh, tcp_z, np.array(cups)[order], float(np.median(radii or [0.0])))


def load_gripper(path: Path) -> GripperModel:
    """A gripper written by import_gripper.py: meters, in the tip link frame."""
    if not path.is_file():
        raise FileNotFoundError(
            f"gripper model not found: {path} (make it with Simulation/import_gripper.py)"
        )
    return measure_gripper(pv.read(path).triangulate())


def stand_in_fish(length: float) -> FishModel:
    """A plain catfish-like body (tapered, with a tail fin), head toward +x, lying flat."""
    widths = 0.75 * np.array(
        [0.04, 0.13, 0.15, 0.13, 0.11, 0.08, 0.08, 0.13, 0.16, 0.18, 0.22, 0.24, 0.25, 0.24,
         0.22, 0.22, 0.23, 0.24, 0.24, 0.25, 0.26, 0.27, 0.27, 0.28, 0.27, 0.27, 0.26, 0.25,
         0.24, 0.23, 0.21, 0.19, 0.18, 0.16, 0.15, 0.14, 0.13, 0.11, 0.09, 0.05]
    ) * length  # fmt: skip  # tail to head, from the stand-in fish's labeled outlines
    stations = (np.arange(len(widths)) + 0.5) / len(widths) * length - length / 2
    heights = np.minimum(widths, 0.18 * length) * 0.75
    heights[:6] = np.maximum(heights[:6] * 0.25, 0.004)  # the tail fin is flat
    ring = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    vertices = [[x, w / 2 * np.cos(a), h / 2 * (1 + np.sin(a))]
                for x, w, h in zip(stations, widths, heights) for a in ring]  # fmt: skip
    n, m = len(stations), len(ring)
    triangles = []
    for i in range(n - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            triangles += [[a, b, b + m], [a, b + m, a + m]]
    tail, head = len(vertices), len(vertices) + 1
    vertices += [[stations[0], 0, heights[0] / 2], [stations[-1], 0, heights[-1] / 2]]
    for j in range(m):
        triangles += [[tail, (j + 1) % m, j], [head, (n - 1) * m + j, (n - 1) * m + (j + 1) % m]]
    return _centered(np.array(vertices), np.array(triangles), "plain stand-in shape")


# FAIRINO's FR5 link meshes in frcobot_ros2, at a fixed commit. They carry no license, so
# they aren't part of this repository: they are downloaded for local use and checked.
FR5_COMMIT = "eb6d7ce40071aeb1a0c06e40e95f3be46bd7594e"
FR5_URL = (
    "https://raw.githubusercontent.com/FAIR-INNOVATION/frcobot_ros2/{commit}/"
    "fairino_description/meshes/fairino5_v6/{name}.STL"
)
FR5_SHA256 = {
    "base_link": "ce96425ae7401b8a7262a1c1fd40c8b02bbf152a975d33debcb3ed4369e3938e",
    "shoulder_link": "c4cc46eea722cab4b57779f0c2cc99091cb0c698a31fc58736c120a0e6dd1324",
    "upperarm_link": "407e0b0f98eb569b9255b6d3258a7c3ba8b0d8b48c8f3c073bd0d9e59ad5faf0",
    "forearm_link": "28387d67b50263ffe67cf8a6a1f2f7f11a12bd2bedc65340f920c5311eb1ef92",
    "wrist1_link": "0f6f96c39bd237f2ff5beeab4179074900cba0d73664dc0bab682d151b777ac6",
    "wrist2_link": "63c43f1f17a86a1d74b07a172185d1cb3ff4133143511832a2bcd31b2b6a9bdf",
    "wrist3_link": "4c4d1706c0a1895d59d70af0f444e72bb80abdb526f1a22f24cd4ad3adfa00fb",
}


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read()


def download_fr5_meshes(folder: Path, fetch=_fetch) -> str | None:
    """Download FAIRINO's FR5 meshes into `folder`, each checked against its SHA-256.
    None when done; else why not (no network, a changed file)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in FR5_SHA256.items():
        target = folder / f"{name}.STL"
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        try:
            data = fetch(FR5_URL.format(commit=FR5_COMMIT, name=name))
        except (urllib.error.URLError, OSError) as e:
            return f"can't download the FR5 meshes ({e})"
        if hashlib.sha256(data).hexdigest() != digest:
            return f"the downloaded {name}.STL isn't the expected file"
        partial = target.with_suffix(".partial")
        partial.write_bytes(data)
        partial.replace(target)
    return None


def find_robot_meshes(
    configured: str, base: Path, cache: Path | None = None, download: bool = False, fetch=_fetch
) -> tuple[Path | None, str]:
    """(folder with the FR5's link meshes or None, where they came from). Looks in the
    configured folder, then a sourced ROS workspace's fairino_description, then `cache`,
    which is filled from FAIRINO's repository if `download` is on."""
    if configured:
        folder = Path(configured).expanduser()
        folder = folder if folder.is_absolute() else base / folder
        return (folder, str(folder)) if folder.is_dir() else (None, f"no folder {folder}")
    for prefix in filter(None, os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)):
        folder = Path(prefix) / "share" / "fairino_description" / "meshes" / "fairino5_v6"
        if folder.is_dir():
            return folder, f"ROS: {folder}"
    if cache is None:
        return None, "no FR5 meshes"
    complete = all((cache / f"{name}.STL").is_file() for name in FR5_SHA256)
    if complete:
        return cache, f"FAIRINO's meshes in {cache}"
    if not download:
        return None, "no FR5 meshes (download_robot_meshes is off)"
    problem = download_fr5_meshes(cache, fetch)
    if problem:
        return None, problem
    return cache, f"FAIRINO's meshes, downloaded to {cache}"


def robot_link_meshes(folder: Path | None, links: list[str]) -> dict[str, pv.PolyData]:
    """Each link's mesh in its own frame, from `folder`; {} if any is missing."""
    if folder is None:
        return {}
    meshes = {}
    for link in links:
        found = [p for p in folder.iterdir() if p.stem == link and p.suffix.lower() == ".stl"]
        if not found:
            return {}
        meshes[link] = pv.read(found[0])
    return meshes
