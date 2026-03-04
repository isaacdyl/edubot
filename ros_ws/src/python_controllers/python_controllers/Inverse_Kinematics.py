import numpy as np


JOINT_LIMITS = np.array([
    [-2.0, 2.0],
    [-1.57, 1.57],
    [-1.58, 1.58],
    [-1.57, 1.57],
], dtype=float)


def get_transform(x, y, z, roll, pitch, yaw):
    t_matrix = np.eye(4)
    t_matrix[0:3, 3] = [x, y, z]

    r_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll), np.cos(roll)],
    ])
    r_y = np.array([
        [np.cos(pitch), 0.0, np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)],
    ])
    r_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])

    t_matrix[0:3, 0:3] = r_z @ r_y @ r_x
    return t_matrix


def forward_kinematics(joint_angles):
    q1, q2, q3, q4 = np.asarray(joint_angles, dtype=float)

    t_w_base = get_transform(0.0, 0.0, 0.0, 0.0, 0.0, 3.14159)
    t_base_sh = get_transform(0.0, -0.0452, 0.0165, 0.0, 0.0, q1)
    t_sh_ua = get_transform(0.0, -0.0306, 0.1025, 0.0, -1.57079, q2)
    t_ua_la = get_transform(0.11257, -0.028, 0.0, 0.0, 0.0, q3)
    t_la_wr = get_transform(0.0052, -0.1349, 0.0, 0.0, 0.0, 1.57079 + q4)
    t_wr_gr = get_transform(-0.0601, 0.0, 0.0, 0.0, -1.57079, 0.0)
    t_gr_gc = get_transform(0.0, 0.0, 0.075, 0.0, 0.0, 0.0)

    return t_w_base @ t_base_sh @ t_sh_ua @ t_ua_la @ t_la_wr @ t_wr_gr @ t_gr_gc


def gripper_center_position(joint_angles):
    return forward_kinematics(joint_angles)[0:3, 3]


def _in_limits(joint_angles):
    return np.all(joint_angles >= JOINT_LIMITS[:, 0]) and np.all(joint_angles <= JOINT_LIMITS[:, 1])


def _solve_q3_from_z(z_base, wrist_pitch):
    u = 0.0052 + 0.1351 * np.sin(wrist_pitch)
    v = -0.1349 - 0.1351 * np.cos(wrist_pitch)

    d = z_base - 0.0165 - 0.1025 - 0.11257
    radius = np.hypot(u, v)
    if radius < 1e-12:
        return []

    c = np.clip(d / radius, -1.0, 1.0)
    alpha = np.arccos(c)
    phi = np.arctan2(v, u)
    return [alpha - phi, -alpha - phi]


def _solve_q1_q2(x_base, y_base, y2):
    x = x_base
    y = y_base + 0.0452
    k = 0.0306
    m = y2

    if abs(k) < 1e-12 or abs(m) < 1e-12:
        return []

    r = np.hypot(x, y)
    if r < 1e-12:
        return []

    c = (m * m - x * x - y * y - k * k) / (2.0 * k)
    c = np.clip(c / r, -1.0, 1.0)
    alpha = np.arccos(c)
    delta = np.arctan2(x, y)

    q1_candidates = [alpha - delta, -alpha - delta]
    solutions = []
    for q1 in q1_candidates:
        sin_b = -(x - k * np.sin(q1)) / m
        cos_b = (y + k * np.cos(q1)) / m
        b = np.arctan2(sin_b, cos_b)
        q2 = b - q1
        solutions.append((q1, q2))

    return solutions


def inverse_kinematics_from_xyz(
    x,
    y,
    z,
    wrist_pitch=0.0,
    initial_guess=None,
    tolerance=1e-4,
    return_all_solutions=False,
):
    target_position = np.array([x, y, z], dtype=float)

    x_w, y_w, z_w = target_position
    x_b, y_b, z_b = -x_w, -y_w, z_w

    q3_candidates = _solve_q3_from_z(z_b, wrist_pitch)
    solutions = []

    for q3 in q3_candidates:
        u = 0.0052 + 0.1351 * np.sin(wrist_pitch)
        v = -0.1349 - 0.1351 * np.cos(wrist_pitch)
        y2 = -0.028 + np.sin(q3) * u + np.cos(q3) * v

        for q1, q2 in _solve_q1_q2(x_b, y_b, y2):
            q = np.array([q1, q2, q3, wrist_pitch], dtype=float)
            if _in_limits(q):
                err = np.linalg.norm(gripper_center_position(q) - target_position)
                if err <= max(5e-3, tolerance * 10.0):
                    solutions.append((q, err))

    if not solutions:
        if return_all_solutions:
            return [], False
        return np.zeros(4, dtype=float), False

    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=float).reshape(4)
        solutions.sort(key=lambda item: np.linalg.norm(item[0] - initial_guess) + item[1])
    else:
        solutions.sort(key=lambda item: item[1])

    angles = [item[0] for item in solutions]
    if return_all_solutions:
        return angles, True

    return angles[0], True


def inverse_kinematics(target_position, initial_guess=None, tolerance=1e-4, wrist_pitch=0.0):
    target_position = np.asarray(target_position, dtype=float).reshape(3)
    best_angles, ok = inverse_kinematics_from_xyz(
        target_position[0],
        target_position[1],
        target_position[2],
        wrist_pitch=wrist_pitch,
        initial_guess=initial_guess,
        tolerance=tolerance,
        return_all_solutions=False,
    )
    if not ok:
        return np.zeros(4, dtype=float), False, np.inf

    error = np.linalg.norm(gripper_center_position(best_angles) - target_position)
    return best_angles, True, error


def quick_verification_case():
    reference_angles = np.array([0.35, -0.65, 0.55, 0.20], dtype=float)
    target_xyz = gripper_center_position(reference_angles)

    solved_angles, ok = inverse_kinematics_from_xyz(
        target_xyz[0],
        target_xyz[1],
        target_xyz[2],
        wrist_pitch=reference_angles[3],
        initial_guess=reference_angles,
    )

    if not ok:
        print("Quick verification failed: no IK solution found.")
        return False

    rebuilt_xyz = gripper_center_position(solved_angles)
    position_error = np.linalg.norm(rebuilt_xyz - target_xyz)

    print("Quick IK verification")
    print("Target gripper center [x, y, z]:", np.round(target_xyz, 6))
    print("Recovered angles [q1, q2, q3, q4] rad:", np.round(solved_angles, 6))
    print("Position error [m]:", f"{position_error:.6e}")

    return position_error < 1e-4


if __name__ == "__main__":
    passed = quick_verification_case()
    print("Verification status:", "PASS" if passed else "FAIL")