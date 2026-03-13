import numpy as np
from Forward_Kinematics_FINAL import forward_kinematics_full, create_tf_matrix

# --- 1. Your Full Forward Kinematics ---

def create_tf_matrix(tx, ty, tz, roll, pitch, yaw):
    Rx = np.array([[1, 0, 0, 0], [0, np.cos(roll), -np.sin(roll), 0], [0, np.sin(roll), np.cos(roll), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return T @ Rz @ Ry @ Rx

def get_joint_rotation(q):
    return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def forward_kinematics_full(q1, q2, q3, q4, q5):
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
    
    return T_world_base @ A1 @ A2 @ A3 @ A4 @ A5 @ T_gripper_center

BOUNDS_MIN = np.array([-2.0000, -1.5700, -1.5800, -1.5700, -3.1415])
BOUNDS_MAX = np.array([ 2.0000,  1.5700,  1.5800,  1.5700,  3.1415])

def calculate_errors(T_current, T_target):
    """Returns position error, rotation error, and combined weighted error."""
    # Position error (Euclidean distance)
    pos_error = np.linalg.norm(T_target[0:3, 3] - T_current[0:3, 3])
    
    # Orientation error (Frobenius norm difference between rotation matrices)
    rot_error = np.linalg.norm(T_target[0:3, 0:3] - T_current[0:3, 0:3])
    
    # Combined error: weight orientation less so the arm prioritizes reaching the coordinate first
    total_error = pos_error + (0.2 * rot_error)
    
    return pos_error, rot_error, total_error


def calculate_scalar_error(T_current, T_target):
    """Returns combined weighted error for backward compatibility."""
    _, _, total_error = calculate_errors(T_current, T_target)
    return total_error

def ik_coordinate_descent(x, y, z, rot_x, rot_y, rot_z, q_init=None, max_iters=5000, tolerance=1e-6):
    T_target = create_tf_matrix(x, y, z, rot_x, rot_y, rot_z)
    
    q = np.zeros(5) if q_init is None else np.clip(np.array(q_init, dtype=float), BOUNDS_MIN, BOUNDS_MAX)
    step_size = 0.0075      # Start with 0.1 radian nudges
    min_step = 1e-6      # Stop when nudges get this tiny
    
    T_current = forward_kinematics_full(*q)
    pos_err, rot_err, current_error = calculate_errors(T_current, T_target)
    
    for iteration in range(max_iters):
        improved = False
        
        # Test nudging each joint one by one
        for i in range(5):
            for direction in [1.0, -1.0]:
                q_test = q.copy()
                q_test[i] += direction * step_size
                
                # Enforce physical limits
                q_test = np.clip(q_test, BOUNDS_MIN, BOUNDS_MAX)
                
                # Check if this nudge made things better
                T_test = forward_kinematics_full(*q_test)
                _, _, test_error = calculate_errors(T_test, T_target)
                
                if test_error < current_error:
                    current_error = test_error
                    T_current = T_test
                    pos_err, rot_err, _ = calculate_errors(T_current, T_target)
                    q = q_test
                    improved = True
                    break # Move to the next joint immediately if we found an improvement
                    
        # If no joints improved the error, we are stuck. Shrink the step size to refine.
        if not improved:
            step_size *= 0.5
            
        # If we hit the target perfectly, or the step size is microscopic, we are done
        if current_error < tolerance:
            return {
                "q": np.round(q, 2),
                "q_raw": q.copy(),
                "pos_error": np.round(pos_err, 4),
                "pos_error_raw": float(pos_err),
                "rot_error": np.round(rot_err, 4),
                "rot_error_raw": float(rot_err),
                "error": np.round(current_error, 3),
                "error_raw": float(current_error),
                "iters": iteration,
            }
            
    return {
        "q": np.round(q, 2),
        "q_raw": q.copy(),
        "pos_error": np.round(pos_err, 4),
        "pos_error_raw": float(pos_err),
        "rot_error": np.round(rot_err, 4),
        "rot_error_raw": float(rot_err),
        "error": np.round(current_error, 3),
        "error_raw": float(current_error),
        "iters": max_iters,
    }


def generate_initial_guesses(num_random=5, seed=42):
    rng = np.random.default_rng(seed)
    guesses = [
        np.zeros(5),
        BOUNDS_MIN,
        BOUNDS_MAX,
        (BOUNDS_MIN + BOUNDS_MAX) * 0.5,
    ]

    for _ in range(num_random):
        guesses.append(rng.uniform(BOUNDS_MIN, BOUNDS_MAX))

    return guesses


def ik_coordinate_descent_multi_start(
    x,
    y,
    z,
    rot_x,
    rot_y,
    rot_z,
    initial_guesses=None,
    max_iters=5000,
    tolerance=1e-6,
    acceptable_error=0.01,
    unique_decimals=2,
):
    if initial_guesses is None:
        initial_guesses = generate_initial_guesses()

    results = []
    seen = set()

    for q0 in initial_guesses:
        result = ik_coordinate_descent(
            x,
            y,
            z,
            rot_x,
            rot_y,
            rot_z,
            q_init=q0,
            max_iters=max_iters,
            tolerance=tolerance,
        )

        rounded_key = tuple(np.round(result["q_raw"], unique_decimals))
        if rounded_key in seen:
            continue

        seen.add(rounded_key)
        results.append(result)

    results.sort(key=lambda item: item["error_raw"])

    valid_results = [item for item in results if item["error_raw"] <= acceptable_error]
    if valid_results:
        return valid_results

    return results[:1]

# --- 3. Test the Setup ---
poses = {
    "I":    [0.2000, 0.2000, 0.2000,  0.0000,  1.5700,  0.6500],
    "II":   [0.2000, 0.1000, 0.4000,  0.0000,  0.0000, -1.5700],
    "III":  [0.0000, 0.0000, 0.4000,  0.0000, -0.7850,  1.5700],
    "IV_a": [0.0000, 0.0000, 0.0700,  3.1410,  0.0000,  0.0000],
    "IV_b": [0.0000, 0.0452, 0.4500, -0.7850,  0.0000,  3.1410]
}

print("Running Coordinate Descent IK...\n")
for name, pose in poses.items():
    res_list = ik_coordinate_descent_multi_start(*pose)
    print(f"Pose {name}:")
    for idx, res in enumerate(res_list, start=1):
        print(f"  Solution {idx}")
        print(f"    Pos Error: {res['pos_error']:.4f}, Rot Error: {res['rot_error']:.4f}, Total: {res['error']:.4f}")
        print(f"    Angles: {res['q']}")
    print()


