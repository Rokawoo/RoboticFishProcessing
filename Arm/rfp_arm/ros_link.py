"""
One rclpy node for the whole program, spun in a background thread, with blocking helpers
for the main thread. rclpy is imported here, when a RosLink is made: importing this module
needs no ROS.
"""

import math
import threading
import time
from collections.abc import Callable, Sequence


class RosLink:
    def __init__(self, node_name: str = "rfp_pick_place"):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, daemon=True, name="rclpy")
        self._thread.start()

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException

        try:
            self._executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt):
            pass  # Ctrl+C: the main thread says so

    def wait(self, future, timeout: float):
        """The result of an rclpy future. Raises TimeoutError after `timeout` seconds."""
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            raise TimeoutError(f"no answer in {timeout:g} s")
        return future.result()

    def call(self, client, request, timeout: float):
        """Call a service and wait for the response."""
        return self.wait(client.call_async(request), timeout)

    def joint_reader(self, names: Sequence[str]) -> Callable[..., list[float] | None]:
        """read(timeout=2.0): the joints `names` (rad, in that order) of the first
        /joint_states message after it is called; None if none comes in time. Reads only."""
        from sensor_msgs.msg import JointState

        latest = {"at": -math.inf, "joints": {}}

        def on_joints(msg) -> None:
            latest["joints"] = {**latest["joints"], **dict(zip(msg.name, msg.position))}
            latest["at"] = time.monotonic()

        self.node.create_subscription(JointState, "joint_states", on_joints, 10)

        def read(timeout: float = 2.0) -> list[float] | None:
            asked = time.monotonic()
            while latest["at"] <= asked or not set(names) <= set(latest["joints"]):
                if time.monotonic() > asked + timeout:
                    return None
                time.sleep(0.02)
            joints = latest["joints"]
            return [float(joints[name]) for name in names]

        return read

    def close(self) -> None:
        self._executor.shutdown()
        self.node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
