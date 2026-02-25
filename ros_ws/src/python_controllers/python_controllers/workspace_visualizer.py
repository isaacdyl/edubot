import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

class WorkspaceVisualizer(Node):
    def __init__(self):
        super().__init__('workspace_visualizer')
        self.publisher = self.create_publisher(Marker, 'workspace_points', 10)
        self.timer = self.create_timer(1.0, self.publish_workspace)

    def get_transform(self, x, y, z, roll, pitch, yaw):
        """Creates a 4x4 homogeneous transformation matrix."""
        # Standard translation matrix
        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        
        # Rotation matrices (RPY)
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        
        T[0:3, 0:3] = Rz @ Ry @ Rx
        return T

    def publish_workspace(self):
        marker = Marker()
        marker.header.frame_id = "world" # Reference frame from your table 
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # Point size
        marker.scale.y = 0.005
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # DATA FROM ASSIGNMENT TABLE 
        # Format: (x, y, z, r, p, y)
        T_w_base = self.get_transform(0.0, 0.0, 0.0, 0.0, 0.0, 3.14159)
        
        # Nest loops for joint angles (Task 1.2.2: use -3.14 to 3.14; Task 1.2.3: use measured limits)
        # Reducing step size for performance; decrease 'step' for higher density
        step = 0.5 
        for q1 in np.arange(-3.14, 3.14, step): # Shoulder yaw
            for q2 in np.arange(-1.57, 1.57, step): # Shoulder pitch
                for q3 in np.arange(-1.57, 1.57, step): # Elbow
                    
                    # Chain the transforms using assignment values 
                    T_base_sh = self.get_transform(0.0, -0.0452, 0.0165, 0.0, 0.0, q1)
                    T_sh_ua = self.get_transform(0.0, -0.0306, 0.1025, 0.0, -1.57079 + q2, 0.0)
                    T_ua_la = self.get_transform(0.11257, -0.028, 0, 0.0, 0.0, q3)
                    T_la_wr = self.get_transform(0.0052, -0.1349, 0, 0.0, 0.0, 1.57079)
                    T_wr_gr = self.get_transform(-0.0601, 0, 0, 0.0, -1.57079, 0.0)
                    T_gr_gc = self.get_transform(0.0, 0.0, 0.075, 0.0, 0.0, 0.0)

                    # Total Forward Kinematics
                    T_final = T_w_base @ T_base_sh @ T_sh_ua @ T_ua_la @ T_la_wr @ T_wr_gr @ T_gr_gc
                    
                    p = Point()
                    p.x, p.y, p.z = T_final[0,3], T_final[1,3], T_final[2,3]
                    marker.points.append(p)

        self.publisher.publish(marker)
        self.get_logger().info('Published workspace points')
    
def main():
    rclpy.init()
    node = WorkspaceVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()