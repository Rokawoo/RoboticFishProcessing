import yaml
from attr import dataclass

@dataclass
class WorkZone:
  enabled: bool
  frame: str
  dimensions: tuple[float, float, float]
  position:   tuple[float, float, float]
  quaternion: tuple[float, float, float, float]
  wall_thickness: float

@dataclass
class Attachment:
  touch_links: tuple[str, ...]

@dataclass
class ExclusionZone:
  id: str
  enabled: bool
  frame: str
  dimensions: tuple[float, float, float]
  position:   tuple[float, float, float]
  quaternion: tuple[float, float, float, float]
  enabled: bool
  attach: Attachment | None = None

@dataclass
class MotionLimits:
  max_linear_speed:  float
  max_angular_speed: float
  max_linear_acceleration:  float
  max_angular_acceleration: float

@dataclass
class WorkspaceConfig:
  work_zone: WorkZone
  exclusion_zones: list[ExclusionZone]
  motion_limits: MotionLimits

def load_workspace_config(path: str) -> WorkspaceConfig:
  with open(path, 'r') as f:
    data = yaml.safe_load(f)

  wz = data['work_zone']
  work_zone = WorkZone(
    enabled=wz["enabled"],
    frame=wz["frame"],
    dimensions=tuple(wz["dimensions"]),
    position=tuple(wz["pose"]["position"]),
    quaternion=tuple(wz["pose"]["quaternion"]),
    wall_thickness=wz["wall_thickness"]
  )

  exclusion_zones = [
    ExclusionZone(
      id=z['id'],
      enabled=z['enabled'],
      frame=z['frame'],
      dimensions=tuple(z['dimensions']),
      position=tuple(z['pose']['position']),
      quaternion=tuple(z['pose']['quaternion']),
      attach=Attachment(touch_links=tuple(z['attach']['touch_links']))
        if 'attach' in z else None
    )
    for z in data['exclusion_zones']
  ]

  ids = [z.id for z in exclusion_zones]
  if len(ids) != len(set(ids)):
      raise ValueError(f"duplicate exclusion zone ids: {ids}")

  motion_limits: MotionLimits
  try:
    motion_limits = MotionLimits(**data["motion_limits"])
  except TypeError as e:
    raise ValueError(f"invalid motion_limits section: {e}") from e
  
  return WorkspaceConfig(
    work_zone=work_zone,
    exclusion_zones=exclusion_zones,
    motion_limits=motion_limits
  )