"""
Measuring the camera on the gripper (AS-60), before a dry run or run while it isn't
calibrated: small moves of the arm around a fish (or anything the model sees), and the mount
that puts every sighting on one point (`camera.mount`, the lens frame in the tip link).
Moves along x, y and z give its turn (also a camera mounted turned or mirrored); turns and
tilts about the tool give where the lens is across it. Along the tool's axis the drawing's
value stays (a tilt shifts the object's apparent middle). Start over a fish in the middle of
the image (calibrate_first: from the first observation pose). No ROS.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rfp_arm.poses import axis_rotation, from_parts, matrix_rpy, rotation_angle
from rfp_vision.camera import Frame
from rfp_vision.geometry import polygon_mask, touches_border

STEP_M = 0.04
TURN_DEG = 20.0
TILT_DEG = 8.0
FRAMES = 3  # per pose, averaged
MIN_DEPTH_SHARE = 0.3  # of the object's inside with depth, for its distance
EDGE_PX = 12  # an outline this close to the image's edge: not wholly in view


@dataclass
class Calibration:
    mount: np.ndarray  # 4x4 T_tip_camera that fits the sightings best
    target: np.ndarray  # where the object is in {B} (m)
    rms_m: float  # how far the sightings are from one point with this mount
    poses: int
    turned_deg: float  # how far this mount is turned from the workcell's
    mirrored: bool  # the moves fit only a mirror image: no mount explains it

    def lines(self, current: np.ndarray) -> list[str]:
        moved = np.linalg.norm(self.mount[:3, 3] - current[:3, 3]) * 1000
        text = [
            f"camera.mount measured: {mount_line(self.mount)}",
            (
                f"   from {self.poses} poses, the object's sightings {self.rms_m * 1000:.1f} mm "
                f"(RMS) from one point; {self.turned_deg:.0f} deg turned and {moved:.0f} mm "
                "moved from the workcell's"
            ),
        ]
        if self.mirrored:
            text.append(
                "   ❌ the image is mirrored: no mount can explain it. Check the camera's "
                "settings (no flipped stream), then measure again."
            )
        elif self.turned_deg > 150:
            text.append("   the camera is mounted half a turn round from what the workcell says:")
            text.append("   this is why the arm went the other way than the fish in the image")
        return text


def poses_around(start_tcp: np.ndarray) -> list[np.ndarray]:
    """TCP poses for the measurement, each to be reached from `start_tcp` and back."""
    poses = []
    for axis in range(3):  # along the base's x, y and z, both ways (z: up only and back)
        for sign in (1.0, -1.0) if axis < 2 else (1.0,):
            pose = start_tcp.copy()
            pose[axis, 3] += sign * STEP_M
            poses.append(pose)
    turns = [(start_tcp[:3, 2], TURN_DEG), (start_tcp[:3, 2], -TURN_DEG)]
    turns += [(start_tcp[:3, 0], TILT_DEG), (start_tcp[:3, 0], -TILT_DEG)]
    turns += [(start_tcp[:3, 1], TILT_DEG), (start_tcp[:3, 1], -TILT_DEG)]
    for axis, degrees in turns:  # about the TCP
        rotation = axis_rotation(axis, math.radians(degrees))
        poses.append(from_parts(rotation @ start_tcp[:3, :3], start_tcp[:3, 3]))
    return poses


def object_in_camera(frame: Frame, polygon: np.ndarray, depth_range) -> np.ndarray | None:
    """The middle of an object's outline as a point in the camera frame, at the median depth
    inside it; None if it isn't wholly in view or has too little depth."""
    if frame.depth is None or touches_border(polygon, frame.size, EDGE_PX):
        return None
    inside = polygon_mask(polygon, frame.depth.shape).astype(bool)
    values = frame.depth[inside]
    valid = values[(values >= depth_range[0]) & (values <= depth_range[1])]
    if valid.size < MIN_DEPTH_SHARE * max(values.size, 1):
        return None
    rows, columns = np.nonzero(inside)
    middle = np.array([[columns.mean(), rows.mean()]])
    return frame.intrinsics.rays(middle)[0] * float(np.median(valid))


def solve(tips: list[np.ndarray], points: list[np.ndarray], current: np.ndarray) -> Calibration:
    """The mount X (T_tip_camera) that best puts `tip @ X @ point` on one place for every
    (tip pose, point in the camera frame) pair; the first pair is the start, the next ones
    the moves (`poses_around`'s order: translations first)."""
    tips = [np.asarray(t, float) for t in tips]
    points = [np.asarray(p, float) for p in points]
    start = tips[0]
    # 1. Rotation, from the moves that only translate: the arm moving by d (in the tip
    # frame) moves a still object by -R^T d in the camera frame
    a, b = [], []
    for tip, point in zip(tips[1:], points[1:]):
        if rotation_angle(tip, start) < math.radians(0.5):  # a straight move ends turned ~0
            a.append(-(point - points[0]))
            b.append(start[:3, :3].T @ (tip[:3, 3] - start[:3, 3]))
    if len(a) < 3:
        raise ValueError("the object wasn't seen from enough of the translated poses")
    u, _, vt = np.linalg.svd(np.array(a).T @ np.array(b))
    mirror = np.linalg.det(vt.T @ u.T) < 0
    rotation = vt.T @ np.diag([1.0, 1.0, -1.0 if mirror else 1.0]) @ u.T
    # 2. Where the lens is across the tool's axis and where the object is, linear for that
    # rotation; along the axis: the workcell's
    lens_z = float(current[2, 3])
    params = np.concatenate([_rotation_vector(rotation), _offset(tips, points, rotation, lens_z)])
    # 3. Everything together (Gauss-Newton)
    for _ in range(20):
        residual = _residuals(params, tips, points, lens_z)
        jacobian = np.empty((len(residual), len(params)))
        for i in range(len(params)):
            bumped = params.copy()
            bumped[i] += 1e-7
            jacobian[:, i] = (_residuals(bumped, tips, points, lens_z) - residual) / 1e-7
        step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        params = params + step
        if np.linalg.norm(step) < 1e-9:
            break
    residual = _residuals(params, tips, points, lens_z).reshape(-1, 3)
    mount = from_parts(axis_rotation_vector(params[:3]), (params[3], params[4], lens_z))
    return Calibration(
        mount=mount,
        target=params[5:8],
        rms_m=float(np.sqrt((residual**2).sum(axis=1).mean())),
        poses=len(tips),
        turned_deg=math.degrees(rotation_angle(current, mount)),
        mirrored=bool(mirror),
    )


def _offset(tips, points, rotation, lens_z: float) -> np.ndarray:
    """(the lens's x and y in the tip frame, the object's position P in {B}) for a known
    rotation and lens z: R_i t - P = -(o_i + R_i R p_i) for every pose i, least squares."""
    rows, right = [], []
    for tip, point in zip(tips, points):
        rows.append(np.hstack([tip[:3, :2], -np.eye(3)]))
        right.append(-(tip[:3, 3] + tip[:3, :3] @ rotation @ point + tip[:3, 2] * lens_z))
    return np.linalg.lstsq(np.vstack(rows), np.concatenate(right), rcond=None)[0]


def _residuals(params, tips, points, lens_z: float) -> np.ndarray:
    rotation = axis_rotation_vector(params[:3])
    lens, target = np.array([params[3], params[4], lens_z]), params[5:8]
    out = [
        tip[:3, :3] @ (rotation @ p + lens) + tip[:3, 3] - target for tip, p in zip(tips, points)
    ]
    return np.concatenate(out)


def axis_rotation_vector(vector) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    return axis_rotation(np.asarray(vector) / angle, angle)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    angle = rotation_angle(np.eye(3), rotation)
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
         rotation[1, 0] - rotation[0, 1]]
    )  # fmt: skip
    if np.linalg.norm(axis) < 1e-9:
        if angle < 1e-6:
            return np.zeros(3)
        values, vectors = np.linalg.eig(rotation)  # half a turn: the axis is the fixed vector
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    return axis / np.linalg.norm(axis) * angle


def measure(
    arm,
    vision,
    workcell,
    tick: Callable[[], str | None] | None = None,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] | None = None,
) -> Calibration:
    """Run the moves (THE ARM MOVES), see the object from each pose and solve for the mount.
    `vision`: a VisionSystem (camera and detector). Back where it started at the end."""
    import time

    sleep = sleep or time.sleep
    depth_range = vision.settings.depth_range_m
    start = arm.tcp_pose()
    tips, points = [], []
    anchor: list = []  # where the object is in the image (px) where the arm started

    def the_object(frame: Frame, found) -> np.ndarray | None:
        """The same object in every look, with several in view: the one wholly in view
        nearest the image's middle at the start, then nearest where it was then (the moves
        shift it by some 100 px; fish lie further apart)."""
        middle = np.array([frame.intrinsics.cx, frame.intrinsics.cy])
        target = anchor[0] if anchor else middle
        best, nearest = None, math.inf
        for detection in found:
            point = object_in_camera(frame, detection.polygon, depth_range)
            centroid = np.asarray(detection.polygon, float).mean(axis=0)
            distance = float(np.linalg.norm(centroid - target))
            if point is not None and distance < nearest:
                best, nearest = (point, centroid), distance
        if best is None:
            return None
        if not anchor:
            anchor.append(best[1])
        return best[0]

    def look(what: str) -> None:
        sleep(0.3)
        seen = []
        for _ in range(FRAMES):
            frame = vision.frame()
            point = the_object(frame, vision.detector.detect(frame.color))
            if point is not None:
                seen.append(point)
        if len(seen) < 2:
            raise ValueError(
                f"the object isn't wholly in view {what}: put a fish in the middle of the "
                "image, all of it in view, and start again"
            )
        tips.append(workcell.tip_of_tcp(arm.tcp_pose()))
        points.append(np.mean(seen, axis=0))
        log(f"   {what}: the object {np.linalg.norm(points[-1]) * 1000:.0f} mm from the lens")

    look("where the arm is")
    for i, pose in enumerate(poses_around(start), 1):
        arm.move_lin(pose, workcell.motion.free_speed / 2, workcell.motion.tcp_accel_max, tick)
        look(f"pose {i}")
        arm.move_lin(start, workcell.motion.free_speed / 2, workcell.motion.tcp_accel_max, tick)
    return solve(tips, points, workcell.camera)


def mount_line(mount: np.ndarray) -> str:
    """The workcell file's `mount:` value for a mount."""
    xyz = mount[:3, 3]
    rpy = np.degrees(matrix_rpy(mount))
    return (
        f"{{xyz: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}], "
        f"rpy_deg: [{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}]}}"
    )


def save_mount(path, mount: np.ndarray, version: str, calibrated: bool = False) -> list[str]:
    """Write the measured mount and calibration version (and `calibrated: true`, if asked)
    into the workcell file's camera section, keeping the rest of each line (its comment)
    and a .bak copy. What changed."""
    import re
    import shutil
    from pathlib import Path

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_camera, changed = False, []
    for i, line in enumerate(lines):
        if re.match(r"^\S", line):
            in_camera = line.startswith("camera:")
            continue
        if not in_camera:
            continue
        values = [("mount", mount_line(mount)), ("calibration_version", version)]
        if calibrated:
            values.append(("calibrated", "true"))
        for key, value in values:
            found = re.match(rf"^(\s+{key}:\s*)(\{{[^}}]*\}}|\S+)(.*)$", line.rstrip("\r\n"))
            if found:
                end = "\n" if line.endswith("\n") else ""
                lines[i] = f"{found.group(1)}{value}{found.group(3)}{end}"
                changed.append(key)
    wanted = ["calibration_version", "mount"] + (["calibrated"] if calibrated else [])
    if sorted(changed) != sorted(wanted):
        raise ValueError(f"{path.name}: no camera {', '.join(wanted)} lines to set")
    shutil.copy(path, path.with_name(path.name + ".bak"))
    path.write_text("".join(lines), encoding="utf-8")
    return [f"camera.{key} in {path.name}" for key in changed]
