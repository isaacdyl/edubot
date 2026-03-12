import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

# ─── Robot geometric parameters ──────────────────────────────────────────────
L_SH_Y, L_SH_Z   = -0.0452,  0.0165
L_UA_Y, L_UA_Z   = -0.0306,  0.1025
L_LA_X, L_LA_Y   =  0.11257, -0.028
L_WR_X, L_WR_Y   =  0.0052, -0.1349
L_GR_X, L_GC_Z   = -0.0601,  0.075

# ─── Joint limits ────────────────────────────────────────────────────────────
LIMITS_UNCONSTRAINED = {
    'q1': (-3.14, 3.14),
    'q2': (-3.14, 3.14),
    'q3': (-3.14, 3.14),
    'q4': (-3.14, 3.14),
    'q5': (-3.14, 3.14),
}

LIMITS_CONSTRAINED = {
    'q1': (-2.0,    2.0   ),
    'q2': (-1.57,   1.57  ),
    'q3': (-1.58,   1.58  ),
    'q4': (-1.57,   1.57  ),
    'q5': (-1.58,   1.58  ),  # Wrist roll doesn't affect workspace boundary
}

STEP = 0.05  # radians between samples in joint space

# ─── Boundary extraction (spherical bins) ─────────────────────────────────────
N_AZ = 240   # azimuth bins  (0..2pi)
N_EL = 120   # elevation bins(-pi/2..pi/2)
ORIGIN = np.array([0.0, 0.0, 0.0])  # bin directions around this point (world frame)

# ─── Kinematics Engine ───────────────────────────────────────────────────────

def rot_z_batch(angles):
    """Returns (N, 3, 3) rotation matrices for an array of angles."""
    c, s = np.cos(angles), np.sin(angles)
    N = len(angles)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0], R[:, 0, 1] = c, -s
    R[:, 1, 0], R[:, 1, 1] = s,  c
    R[:, 2, 2] = 1.0
    return R

def forward_kinematics_batch(q1, q2, q3, q4):
    """
    Computes the XYZ position of the End Effector for N sets of joint angles.
    This follows the physical chain of the robot step-by-step.
    """
    # 1. Rotations at each joint
    R1 = rot_z_batch(q1)                                      # Shoulder Yaw
    R2_local = rot_z_batch(q2)                                # Shoulder Pitch
    R3_local = rot_z_batch(q3)                                # Elbow
    R4_local = rot_z_batch(1.57079 + q4)                      # Wrist (with offset)

    # 2. Fixed Offsets (Translations between frames)
    t_base_sh = np.array([0.0, L_SH_Y, L_SH_Z])               # Base to Shoulder
    t_sh_ua   = np.array([0.0, L_UA_Y, L_UA_Z])               # Shoulder to Upper Arm
    t_ua_la   = np.array([L_LA_X, L_LA_Y, 0.0])               # Upper Arm to Lower Arm
    t_la_wr   = np.array([L_WR_X, L_WR_Y, 0.0])               # Lower Arm to Wrist
    t_wr_ee   = np.array([L_GR_X, 0.0, L_GC_Z])               # Wrist to End Effector (Simplified)

    # 3. Cumulative Rotations & Positions (Chain Multiplication)
    # Joint 1: Shoulder
    p1 = np.einsum('nij,j->ni', R1, t_base_sh)

    # Joint 2: Upper Arm (fixed -90 deg Y offset)
    Ry_offset = np.array([[0, 0, -1],
                          [0, 1,  0],
                          [1, 0,  0]])
    R12 = np.einsum('nij,jk,nkl->nil', R1, Ry_offset, R2_local)
    p2 = p1 + np.einsum('nij,j->ni', R1, t_sh_ua)

    # Joint 3: Lower Arm
    R123 = np.einsum('nij,njk->nik', R12, R3_local)
    p3 = p2 + np.einsum('nij,j->ni', R12, t_ua_la)

    # Joint 4: Wrist
    R1234 = np.einsum('nij,njk->nik', R123, R4_local)
    p4 = p3 + np.einsum('nij,j->ni', R123, t_la_wr)

    # End Effector
    p_ee = p4 + np.einsum('nij,j->ni', R1234, t_wr_ee)

    # 4. Final World Transform (Robot is rotated 180 deg on the table)
    R_world_base = np.array([[-1, 0, 0],
                             [ 0,-1, 0],
                             [ 0, 0, 1]])
    p_world = np.einsum('ij,nj->ni', R_world_base, p_ee)

    return p_world

# ─── Workspace sampling ──────────────────────────────────────────────────────


def print_forward_kinematics():
    """Prints the forward kinematics as a linear combination of transformation matrices."""
    q1 = np.array([0.0])
    q2 = np.array([0.0])
    q3 = np.array([0.0])
    q4 = np.array([0.0])
    
    # Fixed offsets
    t_base_sh = np.array([0.0, L_SH_Y, L_SH_Z])
    t_sh_ua = np.array([0.0, L_UA_Y, L_UA_Z])
    t_ua_la = np.array([L_LA_X, L_LA_Y, 0.0])
    t_la_wr = np.array([L_WR_X, L_WR_Y, 0.0])
    t_wr_ee = np.array([L_GR_X, 0.0, L_GC_Z])
    
    R1 = rot_z_batch(q1)
    R2_local = rot_z_batch(q2)
    R3_local = rot_z_batch(q3)
    R4_local = rot_z_batch(1.57079 + q4)
    
    Ry_offset = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    R12 = np.einsum('nij,jk,nkl->nil', R1, Ry_offset, R2_local)
    R123 = np.einsum('nij,njk->nik', R12, R3_local)
    R1234 = np.einsum('nij,njk->nik', R123, R4_local)
    
    print("FORWARD KINEMATICS AS LINEAR COMBINATION OF TRANSFORMATIONS:\n")
    print("p_ee = T_base_sh + R1*t_sh_ua + R12*t_ua_la + R123*t_la_wr + R1234*t_wr_ee\n")
    
    print(f"T_base_sh: {t_base_sh}")
    print(f"\nR1 (q1={q1[0]}):\n{R1[0]}")
    print(f"→ R1*t_sh_ua = {np.einsum('ij,j->i', R1[0], t_sh_ua)}")
    
    print(f"\nR12 (q1={q1[0]}, q2={q2[0]}):\n{R12[0]}")
    print(f"→ R12*t_ua_la = {np.einsum('ij,j->i', R12[0], t_ua_la)}")
    
    print(f"\nR123 (q1={q1[0]}, q2={q2[0]}, q3={q3[0]}):\n{R123[0]}")
    print(f"→ R123*t_la_wr = {np.einsum('ij,j->i', R123[0], t_la_wr)}")
    
    print(f"\nR1234 (q1={q1[0]}, q2={q2[0]}, q3={q3[0]}, q4={q4[0]}):\n{R1234[0]}")
    print(f"→ R1234*t_wr_ee = {np.einsum('ij,j->i', R1234[0], t_wr_ee)}")
    
    p_world = forward_kinematics_batch(q1, q2, q3, q4)
    print(f"\nFINAL END EFFECTOR POSITION (World Frame):\n{p_world[0]}")

def get_point_cloud(limits, step):
    """Generates the meshgrid and computes all FK positions."""
    q1 = np.arange(*limits['q1'], step)
    q2 = np.arange(*limits['q2'], step)
    q3 = np.arange(*limits['q3'], step)
    q4 = np.arange(*limits['q4'], step)

    Q1, Q2, Q3, Q4 = np.meshgrid(q1, q2, q3, q4, indexing='ij')
    return forward_kinematics_batch(Q1.ravel(), Q2.ravel(), Q3.ravel(), Q4.ravel())

def boundary_by_spherical_binning(points_xyz: np.ndarray,
                                  n_az: int,
                                  n_el: int,
                                  origin: np.ndarray) -> np.ndarray:
    """
    Divide the sphere (directions from origin) into bins and keep only the farthest
    point in each (azimuth,elevation) bin.
    """
    if points_xyz.size == 0:
        return points_xyz

    p = points_xyz - origin[None, :]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x*x + y*y + z*z)

    eps = 1e-12
    r_safe = np.maximum(r, eps)

    phi = np.arctan2(y, x)  # azimuth: [-pi, pi]
    theta = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))  # elevation: [-pi/2, pi/2]

    az = ((phi + np.pi) / (2*np.pi) * n_az).astype(np.int32)
    el = ((theta + np.pi/2) / (np.pi) * n_el).astype(np.int32)

    az = np.clip(az, 0, n_az - 1)
    el = np.clip(el, 0, n_el - 1)

    bin_id = el * n_az + az
    nbins = n_az * n_el

    best_r = -np.ones(nbins, dtype=np.float64)
    best_i = -np.ones(nbins, dtype=np.int32)

    for i in range(points_xyz.shape[0]):
        b = int(bin_id[i])
        if r[i] > best_r[b]:
            best_r[b] = r[i]
            best_i[b] = i

    sel = best_i[best_i >= 0]
    return points_xyz[sel]

# ─── ROS 2 Visualization Logic ───────────────────────────────────────────────

def create_marker(xyz, marker_id, color, stamp, ns, size):
    """Creates a ROS Marker message for a point cloud."""
    m = Marker()
    m.header.frame_id, m.header.stamp = "world", stamp
    m.ns, m.id, m.type, m.action = ns, marker_id, Marker.POINTS, Marker.ADD
    m.scale.x = m.scale.y = size
    m.color.r, m.color.g, m.color.b, m.color.a = color
    for p_val in xyz:
        p = Point()
        p.x, p.y, p.z = float(p_val[0]), float(p_val[1]), float(p_val[2])
        m.points.append(p)
    return m

class WorkspaceVisualizer(Node):
    def __init__(self):
        super().__init__('workspace_visualizer')
        self.pub = self.create_publisher(MarkerArray, 'workspace_points', 10)

        self.get_logger().info("Computing clouds (full)...")
        pts_full = get_point_cloud(LIMITS_UNCONSTRAINED, STEP)
        self.get_logger().info(f"Full cloud computed: {len(pts_full)} points. Extracting boundary...")
        self.pts_full = boundary_by_spherical_binning(pts_full, N_AZ, N_EL, ORIGIN)
        self.get_logger().info(f"Full boundary: {len(self.pts_full)} points.")

        self.get_logger().info("Computing clouds (constrained)...")
        pts_lim = get_point_cloud(LIMITS_CONSTRAINED, STEP)
        self.get_logger().info(f"Constrained cloud computed: {len(pts_lim)} points. Extracting boundary...")
        self.pts_lim = boundary_by_spherical_binning(pts_lim, N_AZ, N_EL, ORIGIN)
        self.get_logger().info(f"Constrained boundary: {len(self.pts_lim)} points.")

        self.timer = self.create_timer(1.0, self.publish)
        self.get_logger().info("Done.")

    def publish(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()
        # Red = Full boundary, Green = Constrained boundary
        ma.markers.append(create_marker(self.pts_full, 0, (1.0, 0.2, 0.2, 0.35), now, "full_boundary", 0.006))
        ma.markers.append(create_marker(self.pts_lim,  1, (0.2, 1.0, 0.4, 0.90), now, "lim_boundary",  0.007))
        self.pub.publish(ma)

def main():
    rclpy.init()
    node = WorkspaceVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    print_forward_kinematics()
    main()