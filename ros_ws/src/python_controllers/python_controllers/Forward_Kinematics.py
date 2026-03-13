import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt

def create_tf_matrix(tx, ty, tz, roll, pitch, yaw):
    Rx = np.array([[1, 0, 0, 0], [0, np.cos(roll), -np.sin(roll), 0], [0, np.sin(roll), np.cos(roll), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return T @ Rz @ Ry @ Rx

def get_joint_rotation(q):
    return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def forward_kinematics(q1, q2, q3, q4, q5=0.0):
    T_world_base     = create_tf_matrix(0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  3.1416)
    T_base_shoulder  = create_tf_matrix(0.0000, -0.0452,  0.0165,  0.0000,  0.0000,  0.0000)
    T_shoulder_upper = create_tf_matrix(0.0000, -0.0306,  0.1025,  0.0000, -1.5708,  0.0000)
    T_upper_lower    = create_tf_matrix(0.1126, -0.0280,  0.0000,  0.0000,  0.0000,  0.0000)
    T_lower_wrist    = create_tf_matrix(0.0052, -0.1349,  0.0000,  0.0000,  0.0000,  1.5708)
    T_wrist_gripper  = create_tf_matrix(-0.0601, 0.0000,  0.0000,  0.0000, -1.5708,  0.0000)
    T_gripper_center = create_tf_matrix(0.0000,  0.0000,  0.0750,  0.0000,  0.0000,  0.0000)

    A1 = T_base_shoulder  @ get_joint_rotation(q1)
    A2 = T_shoulder_upper @ get_joint_rotation(q2)
    A3 = T_upper_lower    @ get_joint_rotation(q3)
    A4 = T_lower_wrist    @ get_joint_rotation(q4)
    A5 = T_wrist_gripper  @ get_joint_rotation(q5)
    
    T_final = T_world_base @ A1 @ A2 @ A3 @ A4 @ A5 @ T_gripper_center
    return T_final[0:3, 3]

# --- Workspace Generation ---

LIMITS_CONSTRAINED = {
    'q1': (-2.0,  2.0),
    'q2': (-1.57, 1.57),
    'q3': (-1.58, 1.58),
    'q4': (-1.57, 1.57),
    'q5': (-1.58, 1.58) 
}

num_samples = 20000  # Increased slightly for better density definition
points = np.zeros((num_samples, 3))

print(f"Generating {num_samples} constrained points. This may take a few seconds...")

for i in range(num_samples):
    q1 = np.random.uniform(*LIMITS_CONSTRAINED['q1'])
    q2 = np.random.uniform(*LIMITS_CONSTRAINED['q2'])
    q3 = np.random.uniform(*LIMITS_CONSTRAINED['q3'])
    q4 = np.random.uniform(*LIMITS_CONSTRAINED['q4'])
    
    pos = forward_kinematics(q1, q2, q3, q4)
    points[i] = pos

# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='plasma', s=1, alpha=0.4)

ax.set_title("SO-ARM101 Constrained Workspace Point Cloud")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")

max_range = np.array([points[:,0].max()-points[:,0].min(), 
                      points[:,1].max()-points[:,1].min(), 
                      points[:,2].max()-points[:,2].min()]).max() / 2.0
mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
mid_z = (points[:,2].max()+points[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Z Height (meters)')

ax.scatter([0], [0], [0], color='red', s=50, label='Base Origin')
ax.legend()

plt.show()
=======


# Robot geometric parameters
L_SH_Y, L_SH_Z = -0.0452, 0.0165
L_UA_Y, L_UA_Z = -0.0306, 0.1025
L_LA_X, L_LA_Y = 0.11257, -0.028
L_WR_X, L_WR_Y = 0.0052, -0.1349
L_GR_X, L_GC_Z = -0.0601, 0.075


def rot_z_batch(angles):
    """Return (N, 3, 3) rotation matrices for an array of angles."""
    c, s = np.cos(angles), np.sin(angles)
    n = len(angles)
    rot = np.zeros((n, 3, 3))
    rot[:, 0, 0], rot[:, 0, 1] = c, -s
    rot[:, 1, 0], rot[:, 1, 1] = s, c
    rot[:, 2, 2] = 1.0
    return rot


def forward_kinematics_batch(q1, q2, q3, q4):
    """
    Compute end-effector XYZ position for batches of joint angles.
    Inputs are arrays of equal length.
    """
    r1 = rot_z_batch(q1)
    r2_local = rot_z_batch(q2)
    r3_local = rot_z_batch(q3)
    r4_local = rot_z_batch(1.57079 + q4)

    t_base_sh = np.array([0.0, L_SH_Y, L_SH_Z])
    t_sh_ua = np.array([0.0, L_UA_Y, L_UA_Z])
    t_ua_la = np.array([L_LA_X, L_LA_Y, 0.0])
    t_la_wr = np.array([L_WR_X, L_WR_Y, 0.0])
    t_wr_ee = np.array([L_GR_X, 0.0, L_GC_Z])

    p1 = np.einsum("nij,j->ni", r1, t_base_sh)

    ry_offset = np.array([[0, 0, -1],
                          [0, 1, 0],
                          [1, 0, 0]])
    r12 = np.einsum("nij,jk,nkl->nil", r1, ry_offset, r2_local)
    p2 = p1 + np.einsum("nij,j->ni", r1, t_sh_ua)

    r123 = np.einsum("nij,njk->nik", r12, r3_local)
    p3 = p2 + np.einsum("nij,j->ni", r12, t_ua_la)

    r1234 = np.einsum("nij,njk->nik", r123, r4_local)
    p4 = p3 + np.einsum("nij,j->ni", r123, t_la_wr)
    p_ee = p4 + np.einsum("nij,j->ni", r1234, t_wr_ee)

    r_world_base = np.array([[-1, 0, 0],
                             [0, -1, 0],
                             [0, 0, 1]])
    return np.einsum("ij,nj->ni", r_world_base, p_ee)


def print_forward_kinematics():
    """Print the FK chain decomposition for the zero configuration."""
    q1 = np.array([0.0])
    q2 = np.array([0.0])
    q3 = np.array([0.0])
    q4 = np.array([0.0])

    t_base_sh = np.array([0.0, L_SH_Y, L_SH_Z])
    t_sh_ua = np.array([0.0, L_UA_Y, L_UA_Z])
    t_ua_la = np.array([L_LA_X, L_LA_Y, 0.0])
    t_la_wr = np.array([L_WR_X, L_WR_Y, 0.0])
    t_wr_ee = np.array([L_GR_X, 0.0, L_GC_Z])

    r1 = rot_z_batch(q1)
    r2_local = rot_z_batch(q2)
    r3_local = rot_z_batch(q3)
    r4_local = rot_z_batch(1.57079 + q4)

    ry_offset = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    r12 = np.einsum("nij,jk,nkl->nil", r1, ry_offset, r2_local)
    r123 = np.einsum("nij,njk->nik", r12, r3_local)
    r1234 = np.einsum("nij,njk->nik", r123, r4_local)

    print("FORWARD KINEMATICS AS LINEAR COMBINATION OF TRANSFORMATIONS:\n")
    print("p_ee = T_base_sh + R1*t_sh_ua + R12*t_ua_la + R123*t_la_wr + R1234*t_wr_ee\n")
    print(f"T_base_sh: {t_base_sh}")
    print(f"\nR1 (q1={q1[0]}):\n{r1[0]}")
    print(f"-> R1*t_sh_ua = {np.einsum('ij,j->i', r1[0], t_sh_ua)}")
    print(f"\nR12 (q1={q1[0]}, q2={q2[0]}):\n{r12[0]}")
    print(f"-> R12*t_ua_la = {np.einsum('ij,j->i', r12[0], t_ua_la)}")
    print(f"\nR123 (q1={q1[0]}, q2={q2[0]}, q3={q3[0]}):\n{r123[0]}")
    print(f"-> R123*t_la_wr = {np.einsum('ij,j->i', r123[0], t_la_wr)}")
    print(f"\nR1234 (q1={q1[0]}, q2={q2[0]}, q3={q3[0]}, q4={q4[0]}):\n{r1234[0]}")
    print(f"-> R1234*t_wr_ee = {np.einsum('ij,j->i', r1234[0], t_wr_ee)}")
    p_world = forward_kinematics_batch(q1, q2, q3, q4)
    print(f"\nFINAL END EFFECTOR POSITION (World Frame):\n{p_world[0]}")
>>>>>>> origin/main
