import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

# ─── Robot geometric parameters (from assignment table) ───────────────────────
L_SH_Y, L_SH_Z   = -0.0452,  0.0165
L_UA_Y, L_UA_Z   = -0.0306,  0.1025
L_LA_X, L_LA_Y   =  0.11257, -0.028
L_WR_X, L_WR_Y   =  0.0052, -0.1349
L_GR_X            = -0.0601
L_GC_Z            =  0.075

# ─── Joint limits ──────────────────────────────────────────────────────────────
# UNCONSTRAINED: full mechanical range (±π or as large as physically possible)
LIMITS_UNCONSTRAINED = {
    'q1': (-3.14159, 3.14159),
    'q2': (-3.14159, 3.14159),
    'q3': (-3.14159, 3.14159),
    'q4': (-3.14159, 3.14159),
}

# CONSTRAINED: actual joint limits from URDF / datasheet
# !! Verify these against your URDF: grep -r "limit" ros_ws/src --include="*.xacro"
LIMITS_CONSTRAINED = {
    'q1': (-2.0,    2.0   ),   # Shoulder yaw
    'q2': (-1.57,   1.57  ),   # Shoulder pitch
    'q3': (-1.58,   1.58  ),   # Elbow
    'q4': (-1.57,   1.57  ),   # Wrist pitch
}

STEP = 0.15   # rad — reduce to 0.1 for denser cloud (slower)


# ─── Vectorized FK helpers ─────────────────────────────────────────────────────

def rot_z_batch(angles):
    """Return (N,3,3) rotation matrices about Z for an array of angles."""
    c, s = np.cos(angles), np.sin(angles)
    N = len(angles)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] =  c;  R[:, 0, 1] = -s
    R[:, 1, 0] =  s;  R[:, 1, 1] =  c
    R[:, 2, 2] =  1.0
    return R

def make_T(x, y, z, roll, pitch, yaw):
    """Single 4×4 homogeneous transform (constant params)."""
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0             ],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,            1]])
    T[:3, :3] = Rz @ Ry @ Rx
    return T

def batch_fk(limits, step):
    """
    Compute end-effector positions for all joint combinations using
    numpy vectorization — no Python loops over joint angles.

    Returns an (N, 3) array of XYZ positions.
    """
    q1_vals = np.arange(*limits['q1'], step)
    q2_vals = np.arange(*limits['q2'], step)
    q3_vals = np.arange(*limits['q3'], step)
    q4_vals = np.arange(*limits['q4'], step)

    # ── Pre-compute constant transforms ──────────────────────────────────────
    T_w_base = make_T(0.0,   0.0,   0.0,   0.0,  0.0,      3.14159)
    T_wr_gr  = make_T(L_GR_X, 0.0,  0.0,   0.0, -1.57079,  0.0    )
    T_gr_gc  = make_T(0.0,   0.0,   L_GC_Z, 0.0, 0.0,      0.0    )
    T_suffix = T_wr_gr @ T_gr_gc          # constant suffix (2 matrices → 1)

    # ── Build all combinations with meshgrid ──────────────────────────────────
    Q1, Q2, Q3, Q4 = np.meshgrid(q1_vals, q2_vals, q3_vals, q4_vals, indexing='ij')
    Q1 = Q1.ravel()
    Q2 = Q2.ravel()
    Q3 = Q3.ravel()
    Q4 = Q4.ravel()
    N  = len(Q1)

    # ── Vectorized per-joint rotation matrices (N,3,3) ────────────────────────
    # Joint 1: pure Rz(q1) with fixed translation
    R1 = rot_z_batch(Q1)                        # (N,3,3)

    # Joint 2: Ry(-π/2) @ Rz(q2)  — pitch offset baked in
    Ry_offset = make_T(0, L_UA_Y, L_UA_Z, 0, -1.57079, 0)[:3, :3]
    R2_z = rot_z_batch(Q2)                      # (N,3,3)
    R2   = np.einsum('ij,njk->nik', Ry_offset, R2_z)

    # Joint 3: pure Rz(q3)
    R3 = rot_z_batch(Q3)                        # (N,3,3)

    # Joint 4: Rz(π/2 + q4)
    R4 = rot_z_batch(1.57079 + Q4)             # (N,3,3)

    # ── Fixed translations for each joint frame ───────────────────────────────
    t_base_sh = np.array([0.0,   L_SH_Y, L_SH_Z])   # base → shoulder
    t_sh_ua   = np.array([0.0,   L_UA_Y, L_UA_Z])    # shoulder → upper arm
    t_ua_la   = np.array([L_LA_X, L_LA_Y, 0.0  ])    # upper arm → lower arm
    t_la_wr   = np.array([L_WR_X, L_WR_Y, 0.0  ])    # lower arm → wrist

    # ── Forward propagation (vectorized) ─────────────────────────────────────
    # p is the running position vector, shape (N,3)
    # We propagate: p = R_prev @ t_next + p_prev

    # Frame 0 → 1  (shoulder): rotation = Rz(q1), translation = t_base_sh
    p1 = np.einsum('nij,j->ni', R1, t_base_sh)   # (N,3)  position of shoulder origin in world

    # Frame 1 → 2  (upper arm): R1 @ (Ry_off_trans + R2 * 0) + p1
    # Translation t_sh_ua is expressed in frame 1 coords
    p2 = p1 + np.einsum('nij,j->ni', R1, t_sh_ua)

    # Frame 2 → 3  (lower arm)
    R12 = np.einsum('nij,njk->nik', R1, R2)       # combined rotation up to joint 2
    p3  = p2 + np.einsum('nij,j->ni', R12, t_ua_la)

    # Frame 3 → 4  (wrist)
    R3_full = rot_z_batch(Q3)
    R123    = np.einsum('nij,njk->nik', R12, R3_full)
    p4      = p3 + np.einsum('nij,j->ni', R123, t_la_wr)

    # Frame 4 → EE  (gripper, constant suffix T_suffix)
    R4_full = rot_z_batch(1.57079 + Q4)
    R1234   = np.einsum('nij,njk->nik', R123, R4_full)

    t_suffix_pos = T_suffix[:3, 3]                # translation of T_wr_gr @ T_gr_gc
    p_ee = p4 + np.einsum('nij,j->ni', R1234, t_suffix_pos)

    # ── Apply world→base constant transform ──────────────────────────────────
    R_wb = T_w_base[:3, :3]
    t_wb = T_w_base[:3,  3]
    p_world = np.einsum('ij,nj->ni', R_wb, p_ee) + t_wb   # (N,3)

    return p_world


def points_to_marker(xyz, frame_id, marker_id, r, g, b, stamp, point_size=0.005, ns="workspace"):
    """Convert (N,3) numpy array to a ROS Marker POINTS message."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp    = stamp
    marker.ns              = ns        # unique namespace → separate RViz toggle
    marker.id              = marker_id
    marker.type            = Marker.POINTS
    marker.action          = Marker.ADD
    marker.scale.x         = point_size
    marker.scale.y         = point_size
    marker.color.r         = float(r)
    marker.color.g         = float(g)
    marker.color.b         = float(b)
    marker.color.a         = 0.6
    for x, y, z in xyz:
        p = Point()
        p.x, p.y, p.z = float(x), float(y), float(z)
        marker.points.append(p)
    return marker


class WorkspaceVisualizer(Node):

    def __init__(self):
        super().__init__('workspace_visualizer')
        self.publisher = self.create_publisher(MarkerArray, 'workspace_points', 10)

        self.get_logger().info('Computing unconstrained workspace...')
        self._pts_unconstrained = batch_fk(LIMITS_UNCONSTRAINED, STEP)
        self.get_logger().info(f'  → {len(self._pts_unconstrained):,} points')

        self.get_logger().info('Computing constrained workspace...')
        self._pts_constrained = batch_fk(LIMITS_CONSTRAINED, STEP)
        self.get_logger().info(f'  → {len(self._pts_constrained):,} points')

        self.get_logger().info('Done! Publishing every second.')
        self.timer = self.create_timer(1.0, self.publish_workspace)

    def publish_workspace(self):
        stamp = self.get_clock().now().to_msg()
        arr   = MarkerArray()

        # Red = unconstrained (full range)
        arr.markers.append(
            points_to_marker(self._pts_unconstrained, "world", 0,
                             r=1.0, g=0.2, b=0.2, stamp=stamp, point_size=0.004,
                             ns="workspace_unconstrained")
        )

        # Green = constrained (actual joint limits)
        arr.markers.append(
            points_to_marker(self._pts_constrained, "world", 0,
                             r=0.1, g=1.0, b=0.3, stamp=stamp, point_size=0.006,
                             ns="workspace_constrained")
        )

        self.publisher.publish(arr)
        self.get_logger().info('Published workspace markers (red=unconstrained, green=constrained)')


def main():
    rclpy.init()
    node = WorkspaceVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()