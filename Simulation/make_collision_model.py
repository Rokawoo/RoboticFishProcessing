#!/usr/bin/env python3
"""
Make the arm's collision model: the FR5's links (FAIRINO's meshes) and the gripper as built
(Simulation/models/gripper.stl) as spheres, written to Arm/configs/collision_model.yaml. The
arm checks every planned path and pose against it so it never runs into itself
(rfp_arm/collision.py, NumPy only: no meshes on the robot PC). Run it again after changing
the gripper's model (a test checks the model was made from the gripper file there now).

Each body is made solid (holes capped, gaps and hollows narrower than --closed counted as
solid: the cups and the space between them, the camera bracket), then covered: every point
of its surface lies in a sphere, and no sphere reaches more than its margin past the solid
(--margin; less for the forearm and wrist 2, MARGINS). Neighbors at a joint always touch
there and aren't checked against each other; nor are bodies whose meshes never come within
NEVER of each other in random arm poses (where their spheres do touch): the checks follow
the real shapes, not the margins.

Usage:
    python Simulation/make_collision_model.py                # write the model
    python Simulation/make_collision_model.py --margin 0.015 # bigger spheres, fewer of them
"""

import argparse
import hashlib
import heapq
import itertools
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "Vision" / "data"))  # pipeline_common: fix_console
try:
    from pipeline_common import fix_console
except ImportError:

    def fix_console() -> None:
        pass


import numpy as np
import pyvista as pv
from rfp_arm.collision import BODY_NAMES, DEFAULT_MODEL, GRIPPER, load_collision_model
from rfp_arm.kinematics import FR5
from rfp_sim.assets import find_robot_meshes, load_gripper, robot_link_meshes
from rfp_sim.settings import MESH_CACHE, SIM_DIR, gripper_path, load_settings

SPACING = 0.003  # m between surface samples
VOXEL = 0.003  # m: the grid that tells inside from outside
GRID = 0.008  # m between candidate sphere centers
POSES = 20_000  # random arm poses, for the pairs that never touch
LOOKS = 300  # of those, at most this many per pair are checked on the meshes
NEVER = 0.01  # m: meshes never this close in any of them: the pair is not checked
FLOOR = {"base_link"}  # bolted down: nothing reaches under it
# Wherever the tool points down (j5 at 90 deg), the forearm and wrist 2 are 19 mm apart: with
# 1 cm spheres on both they would seem to touch (7 mm: 5 mm apart). They do touch with the
# wrist folded (j5).
MARGINS = {"forearm_link": 0.007, "wrist2_link": 0.007}


def surface_samples(mesh: pv.PolyData, spacing: float = SPACING, seed: int = 0) -> np.ndarray:
    """Points all over the mesh's surface, about `spacing` apart, and all its vertices."""
    mesh = mesh.triangulate()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    a, b, c = (mesh.points[faces[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    index = np.repeat(np.arange(len(faces)), np.ceil(area / (0.5 * spacing**2)).astype(int))
    rng = np.random.default_rng(seed)
    u, v = rng.random(len(index)), rng.random(len(index))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = a[index] + u[:, None] * (b - a)[index] + v[:, None] * (c - a)[index]
    return np.vstack([_thinned(points, spacing), mesh.points])


def _thinned(points: np.ndarray, cell: float) -> np.ndarray:
    """One point per `cell`-sized cube."""
    _, first = np.unique(np.floor(points / cell).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def _grown(mask: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown[:, :, 1:] |= mask[:, :, :-1]
        grown[:, :, :-1] |= mask[:, :, 1:]
        mask = grown
    return mask


def solid(samples: np.ndarray, closed: float, floor: bool, voxel: float = VOXEL):
    """(centers of the voxels of the solid body, centers of its outermost ones): the voxels
    the surface passes through, and those the outside can't reach once every gap narrower
    than `closed` is shut (a morphological closing). `floor`: nothing below the body."""
    steps = max(1, round(closed / voxel))
    low = samples.min(axis=0) - (steps + 3) * voxel
    shape = np.ceil((samples.max(axis=0) + (steps + 3) * voxel - low) / voxel).astype(int) + 1
    wall = np.zeros(shape, bool)
    wall[tuple(np.floor((samples - low) / voxel).astype(int).T)] = True
    below = int((samples[:, 2].min() - low[2]) / voxel)  # layers under the body's foot
    blocked = _grown(wall, steps)
    if floor:  # it seals the underside, as it is: not thickened, not part of the body
        blocked[:, :, :below] = True
    outside = np.zeros(shape, bool)
    outside[[0, -1]] = True
    outside[:, [0, -1]] = True
    outside[:, :, [0, -1]] = True
    outside &= ~blocked
    while True:  # flood the outside, around the thickened walls
        grown = _grown(outside, 1) & ~blocked
        if grown.sum() == outside.sum():
            break
        outside = grown
    body = ~_grown(outside, steps) | wall
    if floor:
        body[:, :, :below] = False
    edge = body & _grown(~body, 1)

    def centers(mask):
        return low + (np.argwhere(mask) + 0.5) * voxel

    return centers(body), centers(edge)


def _nearest(points: np.ndarray, others: np.ndarray, chunk: int = 300) -> np.ndarray:
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        d = np.linalg.norm(points[s : s + chunk, None] - others[None], axis=2)
        out[s : s + chunk] = d.min(axis=1)
    return out


def fit_spheres(mesh: pv.PolyData, margin: float, closed: float, floor: bool = False) -> np.ndarray:
    """Spheres (n x 4: x, y, z, radius) covering every surface sample of the mesh, each
    centered in the solid body and at most `margin` past it, few of them (greedy cover)."""
    samples = surface_samples(mesh.triangulate().fill_holes(0.3))
    body, edge = solid(samples, closed, floor)
    # inside, and on the surface close enough that every sample can be covered
    centers = np.vstack([_thinned(body, GRID), _thinned(samples, margin / 2)])
    depth = np.maximum(_nearest(centers, edge) - VOXEL / 2, 0.0)  # to the solid's surface
    radius = depth + margin
    covers = []
    for s in range(0, len(centers), 200):
        d = np.linalg.norm(centers[s : s + 200, None] - samples[None], axis=2)
        covers.extend(np.flatnonzero(row) for row in d <= radius[s : s + 200, None])
    uncovered = np.ones(len(samples), bool)
    queue = [(-len(c), i) for i, c in enumerate(covers)]
    heapq.heapify(queue)
    chosen = []
    while uncovered.any() and queue:  # lazy greedy: the sphere covering most of what's left
        _, i = heapq.heappop(queue)
        gain = int(uncovered[covers[i]].sum())
        if gain == 0:
            continue
        if queue and gain < -queue[0][0]:
            heapq.heappush(queue, (-gain, i))
            continue
        chosen.append(i)
        uncovered[covers[i]] = False
    if uncovered.any():
        raise RuntimeError(f"{uncovered.sum()} surface points left uncovered")
    return np.column_stack([centers[chosen], radius[chosen]])


def never_touching(model, samples: dict, poses: int = POSES, seed: int = 0) -> dict:
    """{pair: closest the meshes came (m)} for the pairs (of the model's checked ones) whose
    meshes, sampled at `samples` (body name: points in its link's frame), never came within
    NEVER of each other in random arm poses (within the joint limits) where their spheres
    touch."""
    rng = np.random.default_rng(seed)
    lower = np.array([j.lower for j in FR5.joints])
    upper = np.array([j.upper for j in FR5.joints])
    positions = lower + (upper - lower) * rng.random((poses, len(lower)))
    touching = np.vstack(
        [model.touching(FR5, positions[s : s + 500]) for s in range(0, poses, 500)]
    )
    links = [FR5.base_link, *FR5.links]
    never = {}
    for k, (a, b) in enumerate(model.pairs):
        closest = np.inf
        for row in np.flatnonzero(touching[:, k])[:LOOKS]:
            frames = FR5.link_poses(positions[row])
            ends = []
            for body in (model.bodies[a], model.bodies[b]):
                t = frames[links.index(body.link)]
                ends.append(samples[body.name] @ t[:3, :3].T + t[:3, 3])
            closest = min(closest, _gap(*ends))
            if closest < NEVER:
                break
        if closest >= NEVER:
            never[(a, b)] = closest
    return never


def _gap(first: np.ndarray, second: np.ndarray, near: float = 0.03) -> float:
    """The closest two point sets come (m; `near` if they are farther than that)."""
    low, high = second.min(axis=0) - near, second.max(axis=0) + near
    first = first[((first >= low) & (first <= high)).all(axis=1)]
    if len(first):
        low, high = first.min(axis=0) - near, first.max(axis=0) + near
        second = second[((second >= low) & (second <= high)).all(axis=1)]
    if not len(first) or not len(second):
        return near
    return min(
        float(np.linalg.norm(first[s : s + 500, None] - second[None], axis=2).min())
        for s in range(0, len(first), 500)
    )


def _number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if value else "0"


def write_model(path: Path, bodies: dict, ignore: list, closed: float, source: str) -> None:
    """`bodies`: {name: (link, margin, spheres)}; `ignore`: [(name, name, why)]."""
    lines = [
        "# The FR5 and the gripper as built, as spheres, for the arm's own collision checks",
        "# (rfp_arm/collision.py). Made by Simulation/make_collision_model.py from FAIRINO's",
        "# meshes and Simulation/models/gripper.stl: don't edit by hand, run it again. Each",
        "# sphere is [x, y, z, radius] in meters in its link's frame; together a body's spheres",
        "# cover its surface, none reaching more than its `margin` past it (gaps and hollows",
        "# narrower than `closed` counted as solid).",
        f"closed: {_number(closed)}",
        f"gripper_sha256: {source}",
        "# Pairs not checked against each other",
        "ignore:",
        *(f"  - [{a}, {b}]  # {why}" for a, b, why in ignore),
        "bodies:",
    ]
    for name, (link, margin, spheres) in bodies.items():
        lines += [f"  {name}:", f"    link: {link}", f"    margin: {_number(margin)}"]
        lines.append("    spheres:")
        for x, y, z, r in spheres:  # 0.1 mm: the radius rounded up, by the centers' rounding too
            r = np.ceil((r + 1e-4) * 1e4) / 1e4
            lines.append(f"      - [{', '.join(_number(v) for v in (x, y, z, r))}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    fix_console()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--margin", type=float, default=0.01, help="m past the body (0.01)")
    parser.add_argument("--closed", type=float, default=0.015, help="m: narrower gaps are solid")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL, help="the model file")
    args = parser.parse_args()

    sim = load_settings()
    folder, _ = find_robot_meshes(sim.robot_meshes, SIM_DIR, MESH_CACHE, sim.download_robot_meshes)
    if folder is None:
        sys.exit("❌ No FR5 meshes: run the setup (it downloads them), then this again.")
    links = ["base_link", *FR5.links]
    meshes = robot_link_meshes(folder, links)
    missing = [link for link in links if link not in meshes]
    if missing:
        sys.exit(f"❌ No mesh for {', '.join(missing)} in {folder}")
    stl = gripper_path(sim)
    meshes[GRIPPER] = load_gripper(stl).mesh

    bodies, samples = {}, {}
    for link, mesh in meshes.items():
        started = time.time()
        margin = min(args.margin, MARGINS.get(link, args.margin))
        spheres = fit_spheres(mesh, margin, args.closed, floor=link in FLOOR)
        name = BODY_NAMES[link]
        bodies[name] = (FR5.tip_link if link == GRIPPER else link, margin, spheres)
        samples[name] = _thinned(surface_samples(mesh, 0.004), 0.004)
        print(f"{name:>10}: {len(spheres):3d} spheres ({time.time() - started:.1f} s)")

    neighbors = [(BODY_NAMES[a], BODY_NAMES[b], "neighbors") for a, b in itertools.pairwise(links)]
    neighbors.append((BODY_NAMES[FR5.tip_link], BODY_NAMES[GRIPPER], "neighbors"))
    digest = hashlib.sha256(stl.read_bytes()).hexdigest()
    write_model(args.out, bodies, neighbors, args.closed, digest)
    load_collision_model.cache_clear()
    model = load_collision_model(args.out)
    started = time.time()
    ignore = list(neighbors)
    for (a, b), closest in never_touching(model, samples).items():
        first, second = model.bodies[a].name, model.bodies[b].name
        why = (
            f"never closer than {closest * 1000:.0f} mm"
            if np.isfinite(closest)
            else f"not even their spheres touch in {POSES} poses"
        )
        ignore.append((first, second, why))
        print(f"{first} and {second}: {why}, not checked")
    print(f"(pairs checked on the meshes in {time.time() - started:.0f} s)")
    write_model(args.out, bodies, ignore, args.closed, digest)
    load_collision_model.cache_clear()
    model = load_collision_model(args.out)
    total = sum(len(body.spheres) for body in model.bodies)
    print(f"✅ {args.out}: {total} spheres in {len(model.bodies)} bodies, {len(model.pairs)} pairs")
    print("Next: pytest Arm/tests Simulation/tests (the model is checked against the meshes)")


if __name__ == "__main__":
    main()
