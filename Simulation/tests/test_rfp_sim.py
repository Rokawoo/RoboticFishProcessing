"""Tests for the 3D simulation: fish models, the virtual cell's contact physics, the virtual
arm and camera, and a whole cycle in it. Rendering is offscreen and small; skipped without
pyvista (the `sim` extra)."""

import math
import sys
import zipfile

import pytest

pv = pytest.importorskip("pyvista")
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from rfp_arm.interfaces import ArmError, GripperState, MotionAborted
from rfp_arm.poses import tool_down, yaw_of
from rfp_arm.workcell import load_workcell
from rfp_sim.assets import load_fish, read_3mf, stand_in_fish
from rfp_sim.devices import VirtualArm, VirtualGripper
from rfp_sim.render import TruthDetector, VirtualCamera
from rfp_sim.settings import SimSettings, load_settings
from rfp_sim.world import Fish, World

CELL = load_workcell(panel=None)


def settings(**changes) -> SimSettings:
    """Small, fast renders and no pacing."""
    return SimSettings(**{"width": 320, "height": 180, "speed": 0.0, **changes})


def world_with_fish(**changes) -> World:
    world = World(CELL, settings(**changes), stand_in_fish(0.65))
    assert world.place_fish(1, 3) == []
    return world


EVEN = -0.03  # m along the stand-in from its middle where all six cups (160 mm) reach it


def press_onto(world: World, fish: Fish, depth: float, along: float = EVEN) -> None:
    """Put the cups `depth` m below the fish's top, `along` from its middle (the arm, by IK)."""
    top = fish.model.mesh.bounds[5]
    xy = fish.pose[:2, 3] + fish.pose[:2, 0] * along
    tcp = tool_down((*xy, CELL.table_z + top - depth), yaw_of(fish.pose))
    world.set_joints(world.chain.ik(CELL.tip_of_tcp(tcp), world.joints))


# ---------------------------------------------------------------------------
# Fish models
# ---------------------------------------------------------------------------
def test_3mf_files_are_read_in_meters_with_their_build_transform(tmp_path):
    model = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources><object id="1" type="model"><mesh>
  <vertices><vertex x="0" y="0" z="0"/><vertex x="100" y="0" z="0"/>
   <vertex x="0" y="50" z="0"/><vertex x="0" y="0" z="20"/></vertices>
  <triangles><triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/>
   <triangle v1="0" v2="3" v3="2"/><triangle v1="1" v2="2" v3="3"/></triangles>
 </mesh></object></resources>
 <build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/></build>
</model>"""
    path = tmp_path / "part.3mf"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("3D/3dmodel.model", model)
    vertices, triangles = read_3mf(path)
    assert triangles.shape == (4, 3)
    assert np.allclose(vertices.min(axis=0), (0.01, 0, 0)) and np.allclose(
        vertices.max(axis=0), (0.11, 0.05, 0.02)
    )
    fish = load_fish(path, 0.5, head="-x", up="+z")  # scaled to 0.5 m, its -x end now at +x
    assert fish.length == pytest.approx(0.5) and fish.mesh.bounds[4] == pytest.approx(0.0)
    tall_end = fish.mesh.points[np.argmax(fish.mesh.points[:, 2])]
    assert tall_end[0] > 0  # the corner that was at x = 0 (the tall one) is toward the head
    with pytest.raises(ValueError, match="different axes"):
        load_fish(path, 0.5, head="+z", up="+z")


def test_the_stand_in_fish_is_centered_like_vision_sees_it():
    fish = stand_in_fish(0.55)
    assert fish.length == pytest.approx(0.55, abs=0.02) and fish.mesh.bounds[4] == pytest.approx(
        0.0
    )
    outline = fish.outline
    area = 0.5 * abs(
        np.dot(outline[:, 0], np.roll(outline[:, 1], 1))
        - np.dot(outline[:, 1], np.roll(outline[:, 0], 1))
    )
    assert area > 0.03 and np.abs(outline.mean(axis=0)).max() < 0.03  # origin near the middle
    assert load_fish(None, 0.4).source == "plain stand-in shape"


def test_settings_file_matches_the_defaults():
    assert load_settings(panel=None) == SimSettings()  # Simulation/sim.yaml
    assert load_settings() == SimSettings()  # the shipped control panel changes nothing


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------
def test_the_arm_starts_at_the_first_observation_pose_with_the_wrist_near_zero():
    world = world_with_fish()
    first = CELL.observation_tcp(CELL.observation_poses[0])
    assert np.allclose(world.tcp(), first, atol=1e-5)
    assert abs(math.degrees(world.joints[-1])) < 30
    fish = world.fish[0]
    assert CELL.pick_area.contains(fish.pose[:2, 3]) and fish.pose[2, 3] == CELL.table_z


def test_cups_feel_the_fish_and_the_vacuum_holds_it_only_when_they_touch():
    world = world_with_fish()
    fish = world.fish[0]
    press_onto(world, fish, -0.02)  # 2 cm above it
    gaps = [gap for gap, _ in world.cup_gaps()]
    assert all(g is not None and 0.0 < g < 0.04 for g in gaps) and world.contact_force() == 0.0
    world.set_vacuum(True)
    assert world.holding is None  # not touching: nothing to hold
    press_onto(world, fish, 0.008)  # 8 mm pressed in
    assert world.holding is fish  # the vacuum was on: it sealed on touching
    force = world.contact_force()
    press_onto(world, fish, 0.011)
    assert world.contact_force() > force > 1.0  # springs: more pressure, more force
    assert fish.pose[2, 3] == CELL.table_z  # the table holds it up while pressed
    press_onto(world, fish, -0.10)  # lifted 10 cm
    assert fish.pose[2, 3] > CELL.table_z + 0.08  # it comes along
    world.set_vacuum(False)
    assert world.holding is None and fish.pose[2, 3] == CELL.table_z  # lies down again
    assert np.allclose(fish.pose[:3, 2], (0, 0, 1))


def test_a_fish_can_be_dropped_on_purpose_to_test_drop_detection():
    world = world_with_fish(drop="lift")
    fish = world.fish[0]
    world.set_vacuum(True)
    press_onto(world, fish, 0.008)
    press_onto(world, fish, -0.05)
    assert world.holding is fish
    press_onto(world, fish, -0.12)  # 8 cm above where it was grabbed: dropped
    assert world.holding is None and fish.pose[2, 3] == CELL.table_z
    world.set_vacuum(False)
    world.set_vacuum(True)
    press_onto(world, fish, 0.008)
    press_onto(world, fish, -0.15)
    assert world.holding is fish  # only the first time


def test_virtual_arm_plans_checks_and_moves():
    world = world_with_fish()
    steps = []
    arm = VirtualArm(world, CELL, on_step=lambda: steps.append(1))
    start = arm.tcp_pose()
    target = start.copy()
    target[:3, 3] += (0.05, 0.10, -0.15)
    arm.move_lin(target, 0.1, 0.5)
    assert np.allclose(arm.tcp_pose(), target, atol=1e-5) and len(steps) >= 30  # 0.18 m at 0.1 m/s
    assert arm.moves[-1][0] == "lin" and arm.moves[-1][1] == pytest.approx(1.87, abs=0.06)  # s
    arm.move_ptp(start)
    assert np.allclose(arm.tcp_pose(), start, atol=1e-5)
    assert abs(arm.wrist_angle(start) - arm.wrist_angle()) < 1e-3
    low = start.copy()
    low[2, 3] = CELL.table_z + 0.001
    with pytest.raises(ArmError, match="above the table"):
        arm.move_lin(low, 0.1, 0.5)
    with pytest.raises(MotionAborted, match="enough"):
        arm.move_lin(target, 0.1, 0.5, tick=lambda: "enough")
    gripper = VirtualGripper(world)
    assert not gripper.has_switch and gripper.state() == GripperState.VACUUM_OFF
    gripper.vacuum_on()
    assert (
        gripper.state() == GripperState.NO_SEAL
        and VirtualGripper(world, True).state() == GripperState.NO_SEAL
    )


# ---------------------------------------------------------------------------
# The camera, and a whole cycle
# ---------------------------------------------------------------------------
def test_virtual_camera_sees_the_table_and_the_fish_where_they_are():
    from rfp_vision.fish_list import FishLocator
    from rfp_vision.settings import VisionSettings

    world = world_with_fish()
    camera = VirtualCamera(world, {})
    detector = TruthDetector(camera)
    try:
        fish = world.fish[0]
        pose = min(
            CELL.observation_poses, key=lambda p: np.linalg.norm(p.camera[:2, 3] - fish.pose[:2, 3])
        )
        world.set_joints(world.chain.ik(CELL.tip_of_tcp(CELL.observation_tcp(pose)), world.joints))
        frame = camera.read()
        assert frame.color.shape == (180, 320, 3) and frame.depth.shape == (180, 320)
        table = frame.depth[0:15, 0:15][frame.depth[0:15, 0:15] > 0]
        assert np.median(table) == pytest.approx(0.45, abs=0.005)  # the table, 450 mm below
        found = detector.detect(frame.color)
        assert len(found) == 1 and found[0].conf == 0.99
        settings = VisionSettings(check_size=False, stable_frames=1)
        located = FishLocator(settings, CELL.table_z, CELL.pick_area.area).locate(
            frame, found, world.camera_pose()
        )
        seen = located.fish[0]
        assert seen.vision_ok, seen.failures
        assert np.linalg.norm(seen.middle[:2] - fish.pose[:2, 3]) < 0.015
        ahead = (seen.grasp_point[:2] - seen.middle[:2]) @ fish.pose[:2, 0]  # toward the head
        # toward the balance point, as far as all six cups (200 mm) still seal on this one
        assert 0.03 < ahead <= 0.126 * seen.total_length + 0.001
        error = math.remainder(seen.heading - yaw_of(fish.pose), math.tau)
        assert abs(error) < math.radians(8)  # head end found from the shape and the slope
    finally:
        camera.close()


def test_a_whole_cycle_in_the_virtual_cell():
    from rfp_controlsys.console import Console
    from rfp_controlsys.cycle import PickPlaceCycle
    from rfp_controlsys.selector import RuleSelector
    from rfp_controlsys.settings import ControlSettings
    from rfp_vision.fish_list import FishLocator
    from rfp_vision.settings import VisionSettings
    from rfp_vision.vision_system import VisionSystem

    world = world_with_fish()
    camera = VirtualCamera(world, {})
    try:
        vision_settings = VisionSettings(check_size=False)
        locator = FishLocator(vision_settings, CELL.table_z, CELL.pick_area.area)
        vision = VisionSystem(camera, TruthDetector(camera), locator, vision_settings)
        arm, gripper = VirtualArm(world, CELL), VirtualGripper(world)
        cycle = PickPlaceCycle(
            arm, gripper, vision, RuleSelector(), CELL, ControlSettings(settle_s=0.0),
            Console(window=False, interactive=False), sleep=lambda seconds: None,
        )  # fmt: skip
        report = cycle.run()
        assert report.ended == "pick area clear" and report.successes == 1, report
        fish = world.fish[0]
        assert np.linalg.norm(fish.pose[:2, 3] - CELL.slots[0]) < 0.03  # in its slot (AS-53)
        assert abs(
            math.remainder(yaw_of(fish.pose) - CELL.placement_heading, math.tau)
        ) < math.radians(8)
        assert world.holding is None and not world.vacuum
    finally:
        camera.close()


def test_a_fish_cut_off_by_the_view_is_placed_after_a_closer_look():
    from rfp_arm.poses import from_parts, rot_z
    from rfp_controlsys.console import Console
    from rfp_controlsys.cycle import PickPlaceCycle
    from rfp_controlsys.selector import RuleSelector
    from rfp_controlsys.settings import ControlSettings
    from rfp_vision.fish_list import FishLocator
    from rfp_vision.settings import VisionSettings
    from rfp_vision.vision_system import VisionSystem

    def run(close_looks: int):
        world = World(CELL, settings(), stand_in_fish(0.79))
        # along X: longer than the 0.50 m the camera sees that way from 450 mm
        world.fish = [Fish("fish 1", world.model, from_parts(rot_z(0.1), (0.36, -0.10, 0)))]
        camera = VirtualCamera(world, {})
        try:
            vision_settings = VisionSettings(check_size=False)
            locator = FishLocator(vision_settings, CELL.table_z, CELL.pick_area.area)
            vision = VisionSystem(camera, TruthDetector(camera), locator, vision_settings)
            console = Console(window=False, interactive=False)
            states = []
            console.state = lambda state, detail="": states.append(f"{state} {detail}")
            cycle = PickPlaceCycle(
                VirtualArm(world, CELL), VirtualGripper(world), vision, RuleSelector(), CELL,
                ControlSettings(settle_s=0.0, close_looks=close_looks), console,
                sleep=lambda seconds: None,
            )  # fmt: skip
            return cycle.run(1), states, world
        finally:
            camera.close()

    report, states, world = run(close_looks=2)
    assert report.successes == 1, report
    assert any("close look at" in state for state in states)
    assert CELL.placement_zone.contains(world.fish[0].pose[:2, 3])
    report, _, world = run(close_looks=0)
    assert report.successes == 0 and "none can be picked" in report.ended, report


def test_fish_can_lie_anywhere_in_the_pick_area_as_real_fish_do():
    world = World(CELL, settings(), stand_in_fish(0.79))
    assert world.scatter_fish(3, seed=4) == []
    zone, base_clear = CELL.work_zone, 0.15
    for fish in world.fish:
        outline = fish.outline()
        assert CELL.pick_area.contains(fish.pose[:2, 3])
        assert (outline[:, 0] >= zone.x[0]).all() and (outline[:, 0] <= zone.x[1]).all()
        assert (outline[:, 1] >= zone.y[0]).all() and (outline[:, 1] <= zone.y[1]).all()
        assert np.hypot(outline[:, 0], outline[:, 1]).min() >= base_clear  # not under the arm
    a, b, c = (f.outline() for f in world.fish)
    for one, other in ((a, b), (a, c), (b, c)):
        assert np.linalg.norm(one[:, None] - other[None], axis=2).min() >= 0.03
    again = World(CELL, settings(), stand_in_fish(0.79))
    again.scatter_fish(3, seed=4)
    assert all(np.allclose(x.pose, y.pose) for x, y in zip(world.fish, again.fish))


def test_a_fish_model_knows_where_it_balances():
    from rfp_sim.assets import _center_of_mass

    box = pv.Box(bounds=(-0.3, 0.5, -0.1, 0.1, 0.0, 0.2)).triangulate()
    faces = box.faces.reshape(-1, 4)[:, 1:]
    assert np.allclose(_center_of_mass(np.asarray(box.points), faces), (0.1, 0.0, 0.1))
    # the plain stand-in is heavier toward its head (+x), like a catfish
    assert stand_in_fish(0.79).center_of_mass[0] > 0.01


def test_the_benchmark_scores_each_fish_against_the_truth():
    from rfp_controlsys.settings import ControlSettings
    from rfp_sim.benchmark import build_bench, report_lines, run_scene, summarize
    from rfp_vision.settings import StandIn, VisionSettings

    sim = settings(fish_model="none.stl", fish_length=0.65, placement="view", seed=3,
                   download_robot_meshes=False)  # fmt: skip  # the plain stand-in shape
    vision = VisionSettings(check_size=False, stand_in=StandIn(balance_ahead_share=0.0))
    bench = build_bench(CELL, vision, ControlSettings(settle_s=0.0), sim, None, False)
    try:
        rows = run_scene(bench, seed=1) + run_scene(bench, seed=2)  # the same cell, reused
    finally:
        bench.session.close()
    assert [r["outcome"] for r in rows] == ["placed", "placed"], rows
    for row in rows:
        assert row["ended"] == "pick area clear" and row["head_right"] == 1
        assert row["slot_error_mm"] < 30 and abs(row["heading_error_deg"]) < 8
        # the cups at the outline's middle (no shift here): behind its center of mass
        assert abs(row["grasp_from_middle_mm"]) < 25 and row["grasp_from_com_mm"] < 0
    summary = summarize(rows)
    assert summary["placed"] == 2 and summary["fish"] == 2 and summary["failures"] == {}
    assert any("(was" in line for line in report_lines(summary, summary))


@pytest.mark.skipif(not pv.system_supports_plotting(), reason="needs a display")
def test_closing_the_3d_window_is_noticed_and_asks_the_run_to_stop():
    from rfp_controlsys.console import Console
    from rfp_sim.render import Viewer

    world = world_with_fish(window=(320, 200))
    viewer = Viewer(world, {}, show=True)
    console = Console(window=False, interactive=False)
    viewer.on_close = console.request_stop
    try:
        assert not viewer.closed and not console.stop_requested()
        viewer.plotter.iren.interactor.ExitCallback()  # the event the close button raises
        assert viewer.closed and console.stop_requested()
        viewer.update()  # harmless afterwards
        viewer.wait(0.05, 1.0)
    finally:
        viewer.close()


@pytest.mark.skipif(
    sys.platform != "win32" or not pv.system_supports_plotting(), reason="needs a Windows display"
)
def test_the_close_button_destroys_the_window_and_the_video_is_still_saved(tmp_path):
    import ctypes

    from rfp_sim.render import Viewer

    video = tmp_path / "view.mp4"
    world = world_with_fish(window=(320, 200))
    viewer = Viewer(world, {}, show=True, video=str(video))
    try:
        viewer.wait(0.2, 1.0)
        user32 = ctypes.windll.user32
        window = int(viewer.plotter.render_window.GetGenericWindowId().split("_")[1], 16)
        assert user32.IsWindow(window)
        user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE, as the X button sends
        viewer.wait(0.2, 1.0)  # used to draw into the destroyed window and raise
        assert viewer.closed and not user32.IsWindow(window)
    finally:
        viewer.close()
    frames = cv2.VideoCapture(str(video))
    assert frames.isOpened() and frames.read()[0]
    frames.release()


# ---------------------------------------------------------------------------
# The arm's model, the gripper, the camera's depth holes
# ---------------------------------------------------------------------------
def test_fr5_meshes_are_downloaded_once_and_checked(tmp_path, monkeypatch):
    import hashlib
    import urllib.error

    from rfp_sim import assets

    files = {"base_link": b"solid base", "wrist3_link": b"solid wrist"}
    monkeypatch.setattr(
        assets, "FR5_SHA256", {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    )
    asked = []

    def fetch(url):
        asked.append(url)
        return files[url.split("/")[-1][:-4]]

    cache = tmp_path / "fr5"
    folder, source = assets.find_robot_meshes("", tmp_path, cache, download=True, fetch=fetch)
    assert folder == cache and "downloaded" in source and len(asked) == 2
    assert assets.FR5_COMMIT in asked[0] and asked[0].startswith(
        "https://raw.githubusercontent.com/"
    )
    assert (cache / "wrist3_link.STL").read_bytes() == b"solid wrist"
    folder, source = assets.find_robot_meshes("", tmp_path, cache, download=True, fetch=fetch)
    assert folder == cache and len(asked) == 2  # kept: not fetched again

    other = tmp_path / "other"
    assert assets.find_robot_meshes("", tmp_path, other, download=False)[0] is None
    tampered = assets.download_fr5_meshes(other, lambda url: b"something else")
    assert "isn't the expected file" in tampered and not list(other.glob("*.STL"))

    def offline(url):
        raise urllib.error.URLError("no network")

    assert "can't download" in assets.download_fr5_meshes(tmp_path / "x", offline)
    assert assets.find_robot_meshes("mine", tmp_path, cache)[0] is None  # configured, missing
    (tmp_path / "mine").mkdir()
    assert assets.find_robot_meshes("mine", tmp_path, cache)[0] == tmp_path / "mine"


def test_the_camera_has_no_depth_at_the_fish_edges_like_the_d405():
    world = world_with_fish(width=640, height=360)
    holes = VirtualCamera(world, {}, labels=True)
    whole = VirtualCamera(world, {}, labels=True)
    whole.world = world
    try:
        fish = world.fish[0]
        pose = min(
            CELL.observation_poses, key=lambda p: np.linalg.norm(p.camera[:2, 3] - fish.pose[:2, 3])
        )
        world.set_joints(world.chain.ik(CELL.tip_of_tcp(CELL.observation_tcp(pose)), world.joints))
        frame = holes.read()
        mask = holes.last_masks[0]
        edge = mask ^ cv2.erode(mask, np.ones((3, 3), np.uint8))
        # no depth along the outline, except where a thin fin lies flat (a step under 1 cm)
        assert (frame.depth[edge > 0] == 0).mean() > 0.75
        inside = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0
        assert (frame.depth[inside] > 0).mean() > 0.97  # the body itself is measured
        world.settings.depth_edge_holes_px = 0
        assert (whole.read().depth[edge > 0] > 0).mean() > 0.9  # switched off: depth everywhere
    finally:
        holes.close()
        whole.close()


def test_the_drawn_gripper_matches_the_settings():
    """The CAD model's cups are where the TCP, the seal and vision's footprint put them, and
    the camera sits on the model's camera, looking out of it."""
    from rfp_sim.assets import load_gripper
    from rfp_sim.render import without_camera_body
    from rfp_sim.settings import gripper_path
    from rfp_vision.settings import load_settings as vision_settings

    settings = load_settings(panel=None)
    gripper = load_gripper(gripper_path(settings))
    assert CELL.tcp[2, 3] == pytest.approx(gripper.tcp_z, abs=0.0005)
    assert np.allclose(CELL.tcp[:2, 3], 0) and np.allclose(CELL.tcp[:3, 2], (0, 0, 1))
    layout = sorted((x, y) for x in settings.cup_rows for y in settings.cup_columns)
    assert np.allclose(sorted(map(tuple, gripper.cups)), layout, atol=0.0005)
    vision = vision_settings(panel=None)
    span = gripper.cups.max(axis=0) - gripper.cups.min(axis=0) + 2 * gripper.cup_radius
    assert np.allclose(vision.cup_footprint_m, span, atol=0.001)
    assert vision.cup_diameter_m == pytest.approx(2 * gripper.cup_radius, abs=0.001)
    low, high = np.array(gripper.mesh.bounds).reshape(3, 2).T
    body = CELL.tool_box
    assert body.x[0] <= low[0] + 1e-3 and body.x[1] >= high[0] - 1e-3  # MoveIt's box holds it
    assert body.y[0] <= low[1] + 1e-3 and body.y[1] >= high[1] - 1e-3
    assert body.z[1] < gripper.tcp_z  # short of the cup faces
    # the lens on the model's surface, and nothing of the gripper ahead of it once its own
    # housing is left out
    lens = CELL.camera[:3, 3]
    _, nearest = gripper.mesh.find_closest_cell(lens, return_closest_point=True)
    assert np.linalg.norm(nearest - lens) < 0.002
    seen = without_camera_body(gripper.mesh, CELL.camera)
    local = (seen.points - lens) @ CELL.camera[:3, :3]
    ahead = local[local[:, 2] > 0.0]
    half_width = math.tan(math.radians(settings.hfov_deg / 2))
    half_height = half_width * settings.height / settings.width
    in_view = (np.abs(ahead[:, 0]) < half_width * ahead[:, 2]) & (
        np.abs(ahead[:, 1]) < half_height * ahead[:, 2]
    )
    assert not in_view.any()  # the cups stay just outside the image (33 deg off the axis)


def test_a_gripper_model_is_imported_into_the_tip_link_frame():
    from rfp_sim.assets import FLANGE_Z, import_gripper

    # millimeters, cups down (-z), the flange disk on top off the origin, a camera on an arm
    parts = [pv.Cylinder(center=(10, 20, 75), direction=(0, 0, 1), radius=40, height=10),
             pv.Box((-90, 110, -10, 50, 0, 70)),
             pv.Box((-11, 31, -100, -10, 30, 50))]  # fmt: skip
    parts += [
        pv.Cylinder(center=(10 + x, 20 + y, -30), direction=(0, 0, 1), radius=20, height=60)
        for x in (-80, 0, 80)
        for y in (-30, 30)
    ]
    cad = pv.merge([p.triangulate() for p in parts])
    gripper = import_gripper(cad)
    assert gripper.tcp_z == pytest.approx(FLANGE_Z + 0.140)
    assert gripper.mesh.points[:, 2].min() == pytest.approx(FLANGE_Z)
    assert np.allclose(sorted(map(tuple, gripper.cups)),
                       sorted((x, y) for x in (-0.08, 0, 0.08) for y in (-0.03, 0.03)),
                       atol=1e-3)  # fmt: skip
    assert gripper.cup_radius == pytest.approx(0.02, abs=1e-3)
    camera = gripper.mesh.points[gripper.mesh.points[:, 1] < -0.05]
    assert camera[:, 0].mean() == pytest.approx(0.0, abs=0.002)  # turned over about y
    turned = import_gripper(cad, turn_deg=90)
    assert turned.mesh.points[:, 0].max() > 0.1  # the camera arm now along +x
    side = import_gripper(cad.rotate_x(-90, inplace=False), up="+y")  # the cups along -y
    assert side.tcp_z == pytest.approx(FLANGE_Z + 0.140) and len(side.cups) == 6


def test_the_vacuum_tube_coils_round_the_wrist_as_it_turns():
    from rfp_sim.render import Viewer

    world = world_with_fish(window=(320, 200))
    viewer = Viewer(world, {}, show=False)
    try:
        assert viewer.scene.tube.n_points > 100
        poses = world.chain.link_poses(world.joints)
        relaxed = viewer.scene.tube_centerline(poses)
        world.joints[-1] = world.relaxed_wrist + math.radians(150)
        twisted = viewer.scene.tube_centerline(world.chain.link_poses(world.joints))
        tip = world.chain.fk(world.joints)
        start = np.linalg.inv(tip) @ np.append(twisted[0], 1.0)
        end = np.linalg.inv(tip) @ np.append(twisted[23], 1.0)  # where it leaves the wrist
        turned = math.atan2(end[1], end[0]) - math.atan2(start[1], start[0])
        assert abs(math.remainder(turned + math.radians(150), math.tau)) < 0.05  # wound by 150 deg
        assert np.allclose(relaxed[-1], twisted[-1])  # the forearm end doesn't move
        viewer.update()
        assert "tube twisted +150" in viewer.text.GetText(2)  # 2: upper left
    finally:
        viewer.close()
