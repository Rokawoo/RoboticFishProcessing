from typing import Final

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, ObjectColor, PlanningScene

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA

from workspace_config import ExclusionZone, WorkZone, load_workspace_config

# TODO: consider using service calls
class WorkSpaceManager(Node):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__('work_zone_manager', *args, **kwargs)

    self.declare_parameter('workspace_config_path', '')
    path = self.get_parameter('workspace_config_path').value

    if not path:
      raise ValueError('workspace_config_path parameter is not set')

    latched_qos: QoSProfile = QoSProfile(
      durability=DurabilityPolicy.TRANSIENT_LOCAL,
      reliability=ReliabilityPolicy.RELIABLE,
      depth=10
    )

    self._planning_scene_publisher: Publisher = self.create_publisher(
      PlanningScene,
      '/planning_scene',
      latched_qos,
    )
    
    self._id_prefix: Final[str] = 'zone/'
    self.wait_for_move_group()

    self._workspace_config = load_workspace_config(path)
    self.setup_work_zone(self._workspace_config.work_zone)
    self._attached_links: dict[str, str] = {}
    self._active_exclusion_zone_ids: list[str] = []
    for exclusion_zone in self._workspace_config.exclusion_zones:
      self.setup_exclusion_zone(exclusion_zone)

  def setup_work_zone(self, cfg: WorkZone) -> None:
    if not cfg.enabled:
      return
    
    width, length, height = cfg.dimensions
    x, y, z = cfg.position

    boxes = [
      ((width, cfg.wall_thickness, height), (x, y - length / 2, z), f"{self._id_prefix}-y_wall"),
      ((width, cfg.wall_thickness, height), (x, y + length / 2, z), f"{self._id_prefix}+y_wall"),
      ((cfg.wall_thickness, length, height), (x - width / 2, y, z), f"{self._id_prefix}-x_wall"),
      ((cfg.wall_thickness, length, height), (x + width / 2, y, z), f"{self._id_prefix}+x_wall"),
      ((width, length, cfg.wall_thickness), (x, y, z - height / 2), f"{self._id_prefix}-z_wall"),
      ((width, length, cfg.wall_thickness), (x, y, z + height / 2), f"{self._id_prefix}+z_wall"),
    ]

    scene = PlanningScene()
    scene.is_diff = True

    for dims, pos, obj_id in boxes:
      wall = self._create_box(cfg.frame, dims, pos, obj_id)
      
      scene.world.collision_objects.append(wall) # pyright: ignore[reportAttributeAccessIssue]
      
      scene.object_colors.append( # pyright: ignore[reportAttributeAccessIssue]
        self._create_object_color(obj_id, (0.8, 0.8, 0.8, 0.05))
      )

    self._planning_scene_publisher.publish(scene)
    self.get_logger().info('Work zone setup complete.')

  def setup_exclusion_zone(self, cfg: ExclusionZone) -> None:
    if not cfg.enabled:
      return
    
    scene = PlanningScene()
    scene.is_diff = True

    object_id = f"{self._id_prefix}{cfg.id}"

    box = self._create_box(cfg.frame, cfg.dimensions, cfg.position, object_id)

    if cfg.attach is not None:
      scene.robot_state.is_diff = True
      attached = self._build_attached(
        box,
        cfg.frame,
        list(cfg.attach.touch_links),
      )
      scene.robot_state.attached_collision_objects.append(attached) # pyright: ignore[reportAttributeAccessIssue]
    else:
      scene.world.collision_objects.append(box) # pyright: ignore[reportAttributeAccessIssue]

    scene.object_colors.append( # pyright: ignore[reportAttributeAccessIssue]
      self._create_object_color(object_id, (1.0, 0.0, 0.0, 0.5))
    )

    self._planning_scene_publisher.publish(scene)
    self.get_logger().info(f"Added exclusion zone: {object_id}")
    self._active_exclusion_zone_ids.append(object_id)

  def clear_work_zone(self) -> None:
    for object_id in ('-y_wall', '+y_wall', '-x_wall', '+x_wall', '-z_wall', '+z_wall'):
      self._remove_collision_object(object_id)

  def clear_exclusion_zones(self) -> None:
    for object_id in self._active_exclusion_zone_ids:
      self._remove_collision_object(object_id)
    self._active_exclusion_zone_ids.clear()

  def clear_all_zones(self) -> None:
    self.clear_work_zone()
    self.clear_exclusion_zones()

  def _create_box(
    self,
    frame_id: str,
    dimensions: tuple[float, float, float],
    position: tuple[float, float, float],
    object_id: str
  ) -> CollisionObject:
    
    collision_object = CollisionObject()
    collision_object.header.frame_id = frame_id
    collision_object.id = object_id
    collision_object.operation = CollisionObject.ADD

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(dimensions)

    box_pose = Pose()
    box_pose.position.x, box_pose.position.y, box_pose.position.z = position
    box_pose.orientation.w = 1.0

    collision_object.primitives = [primitive]
    collision_object.primitive_poses = [box_pose]
    
    return collision_object

  def _build_attached(
    self,
    collision_object: CollisionObject,
    link_name: str,
    touch_links: list[str]
  ) -> AttachedCollisionObject:
    
    attached = AttachedCollisionObject()
    attached.link_name = link_name
    attached.touch_links = touch_links
    attached.object = collision_object
    return attached

  def _create_object_color(
    self,
    object_id: str,
    color: tuple[float, float, float, float]
  ) -> ObjectColor:
    
    object_color = ObjectColor()
    object_color.id = object_id
    object_color.color = ColorRGBA(
      r=color[0],
      g=color[1],
      b=color[2],
      a=color[3]
    )

    return object_color

  def _remove_collision_object(self, object_id: str) -> None:
    collision_object = CollisionObject()
    collision_object.id = object_id
    collision_object.operation = CollisionObject.REMOVE

    planning_scene = PlanningScene()
    planning_scene.is_diff = True

    link = self._attached_links.pop(object_id, None)
    if link is not None:
      # Attached objects are removed through robot_state, not world
      planning_scene.robot_state.is_diff = True
      attached = AttachedCollisionObject()
      attached.link_name = link
      attached.object = collision_object
      planning_scene.robot_state.attached_collision_objects.append(attached) # pyright: ignore[reportAttributeAccessIssue]
    else:
      planning_scene.world.collision_objects.append(collision_object) # pyright: ignore[reportAttributeAccessIssue]

    self._planning_scene_publisher.publish(planning_scene)

  def wait_for_move_group(self, timeout_sec: float = 5.0) -> bool:
    """Wait until move_group subscribes to /planning_scene."""
    start_time = self.get_clock().now().nanoseconds / 1e9
    while self._planning_scene_publisher.get_subscription_count() == 0:
      if (self.get_clock().now().nanoseconds / 1e9 - start_time) > timeout_sec:
        self.get_logger().warn("Timed out waiting for move_group subscriber on /planning_scene!")
        return False
      rclpy.spin_once(self, timeout_sec=0.1)
    self.get_logger().info("Connected to move_group planning scene subscriber.")
    return True