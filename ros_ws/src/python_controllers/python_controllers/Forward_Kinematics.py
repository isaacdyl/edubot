import numpy as np


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
