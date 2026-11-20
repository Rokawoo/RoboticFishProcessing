"""Everything of a simulated cell, built from the settings: world, arm, gripper, camera, viewer."""

from collections.abc import Callable
from dataclasses import dataclass

from rfp_arm.kinematics import FR5
from rfp_arm.workcell import Workcell
from rfp_vision.geometry import Intrinsics

from rfp_sim.assets import find_robot_meshes, load_fish, robot_link_meshes
from rfp_sim.devices import VirtualArm, VirtualGripper
from rfp_sim.render import TruthDetector, Viewer, VirtualCamera
from rfp_sim.settings import MESH_CACHE, SIM_DIR, SimSettings, model_path
from rfp_sim.world import World


@dataclass
class SimSession:
    world: World
    arm: VirtualArm
    gripper: VirtualGripper
    camera: VirtualCamera
    viewer: Viewer | None
    detector: TruthDetector | None  # None: use the trained model
    notes: list[str]

    def wait(self, seconds: float) -> None:
        """A pause of the cycle (vacuum building up, settling): simulated time passes."""
        if self.viewer is not None:
            self.viewer.wait(seconds, self.world.settings.speed)

    def close(self) -> None:
        self.camera.close()
        if self.viewer is not None:
            self.viewer.close()


def build_session(
    workcell: Workcell,
    settings: SimSettings,
    show: bool = True,
    video: str | None = None,
    viewer: bool = True,
    log: Callable[[str], None] = print,
    lens: Intrinsics | None = None,
) -> SimSession:
    """`lens`: the camera's image size and field of view (VisionSettings.lens())."""
    path = model_path(settings)
    model = load_fish(path if path.is_file() else None, settings.fish_length, settings.fish_head,
                      settings.fish_up)  # fmt: skip
    notes = [f"fish: {model.source}, {model.length * 1000:.0f} mm long"]
    if not path.is_file():
        notes.append(f"no fish model at {path}: using a plain stand-in shape")
    folder, source = find_robot_meshes(
        settings.robot_meshes, SIM_DIR, MESH_CACHE, settings.download_robot_meshes
    )
    meshes = robot_link_meshes(folder, ["base_link", *FR5.links])
    notes.append(f"arm: {source}" if meshes else f"arm: plain shapes ({source})")
    world = World(workcell, settings, model, lens=lens)
    if settings.placement not in ("view", "anywhere"):
        raise ValueError(f"placement must be 'view' or 'anywhere', not '{settings.placement}'")
    place = world.scatter_fish if settings.placement == "anywhere" else world.place_fish
    notes += place(settings.fish_count, settings.seed)
    camera = VirtualCamera(world, meshes)
    shown = Viewer(world, meshes, camera, show=show, video=video) if viewer else None
    arm = VirtualArm(world, workcell, on_step=shown.update if shown else None)
    detector = TruthDetector(camera) if settings.detector == "truth" else None
    for note in notes:
        log(f"   {note}")
    return SimSession(world, arm, VirtualGripper(world), camera, shown, detector, notes)
