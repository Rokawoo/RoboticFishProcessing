"""
One rclpy node for the whole program, spun in a background thread, with blocking helpers
for the main thread. rclpy is imported here, when a RosLink is made: importing this module
needs no ROS.
"""

import threading


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
        self._thread = threading.Thread(target=self._executor.spin, daemon=True, name="rclpy")
        self._thread.start()

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

    def close(self) -> None:
        self._executor.shutdown()
        self.node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
