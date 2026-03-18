import rclpy
import numpy as np
from rclpy.node import Node
from python_controllers.Inverse_Kinematics_Numerical import ik_coordinate_descent
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class RectangleTraj(Node):

    def __init__(self):
        super().__init__('rectangle_trajectory')

        # Initial safe home position for a 5-DOF arm
        self._HOME = np.array([
            np.deg2rad(0), np.deg2rad(-105),
            np.deg2rad(70), np.deg2rad(60),
            np.deg2rad(0)
        ])
        
        self._last_q = self._HOME.copy()
        self._beginning = self.get_clock().now()

        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        
        self._marker_pub = self.create_publisher(Marker, 'end_effector_trace', 10)
        self._init_trace_marker()


        timer_period = 0.02  # 25 Hz
        self._timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info("Starting Horizontal Rectangle Trajectory...")

    def get_rectangle_pose(self, dt):
        cycle_time = 10.0  
        t = dt % cycle_time
        fraction = t / cycle_time

        # FIX 1 & 2: Moved into safe workspace and aligned dimensions with perimeter math
        # X delta is exactly 0.10, Y delta is exactly 0.20
        x_min, x_max = -0.10, 0.10  
        y_min, y_max = 0.20, 0.40 
        z = 0.0  # Raised slightly to safely clear the table
        
        roll, pitch, yaw = 0.0, 0.0, 0.0  # Fixed orientation facing downwards 

        perimeter = 0.60
        d = fraction * perimeter  

        if d < 0.10: 
            # Edge 1: Bottom (Moving forward in X)
            x = x_min + (d / 0.10) * (x_max - x_min)
            y = y_min
        elif d < 0.30: 
            # Edge 2: Right (Moving left in Y)
            x = x_max
            y = y_min + ((d - 0.10) / 0.20) * (y_max - y_min)
        elif d < 0.40: 
            # Edge 3: Top (Moving backward in X)
            x = x_max - ((d - 0.30) / 0.10) * (x_max - x_min)
            y = y_max
        else: 
            # Edge 4: Left (Moving right in Y)
            x = x_min
            y = y_max - ((d - 0.40) / 0.20) * (y_max - y_min)

        return x, y, z, roll, pitch, yaw
    
    def _init_trace_marker(self):
        """Sets up the visual properties of the trace line."""
        self._trace_marker = Marker()
        # Change "world" if your robot's base frame is named something else (like "base_link")
        self._trace_marker.header.frame_id = "world" 
        self._trace_marker.ns = "rectangle_trace"
        self._trace_marker.id = 0
        self._trace_marker.type = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        
        # Line width
        self._trace_marker.scale.x = 0.005 
        
        # Bright neon green color (R, G, B, Alpha)
        self._trace_marker.color.r = 0.0
        self._trace_marker.color.g = 1.0
        self._trace_marker.color.b = 0.0
        self._trace_marker.color.a = 1.0

    def timer_callback(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        dt = (now - self._beginning).nanoseconds * (1e-9)
        
        target_x, target_y, target_z, r, p, y = self.get_rectangle_pose(dt)
        
        # FIX 3: Added optimize_orientation=False for the 5-DOF arm
        # Bumped max_iters slightly to ensure the first frame solves cleanly
        ik_result = ik_coordinate_descent(
            target_x, target_y, target_z, r, p, y,
            q_init=self._last_q,
            max_iters=100,
            optimize_orientation=False 
        )

        if ik_result["success"]:
            self._last_q = ik_result["q_raw"]
        else:
            self.get_logger().warn(
                f"IK Failed at X:{target_x:.3f} Y:{target_y:.3f} Z:{target_z:.3f}. Holding position.", 
                throttle_duration_sec=1.0
            )

        point = JointTrajectoryPoint()
        gripper_state = 0.0 
        
        point.positions = [
            float(self._last_q[0]),
            float(self._last_q[1]),
            float(self._last_q[2]),
            float(self._last_q[3]),
            float(self._last_q[4]),
            gripper_state
        ]
        
        msg.points = [point]
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rectangle_traj = RectangleTraj()
    
    try:
        rclpy.spin(rectangle_traj)
    except KeyboardInterrupt:
        pass
    finally:
        rectangle_traj.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()