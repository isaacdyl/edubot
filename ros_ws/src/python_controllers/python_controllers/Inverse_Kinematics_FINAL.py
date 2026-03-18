from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:
    from python_controllers.Forward_Kinematics_FINAL import create_tf_matrix, forward_kinematics_full
except ModuleNotFoundError:
    from Forward_Kinematics_FINAL import create_tf_matrix, forward_kinematics_full

NUM_ACTIVE_JOINTS = 5

JOINT_LIMITS = [
    (-2, 2),       # q1
    (-1.57, 1.57), # q2
    (-1.58, 1.58), # q3
    (-1.57, 1.57), # q4
    (-np.pi, np.pi)# q5
]

@dataclass
class IKSolveResult:
    feasible: bool
    q_raw: np.ndarray
    q: np.ndarray
    pos_error_raw: float
    rot_error_raw: float
    error_raw: float


def _ik_objective_function(q, target_matrix):
    """
    This is the function Scipy will try to minimize. 
    It calculates how far the current guess (q) is from the target.
    """
    current_tf = forward_kinematics_full(q[0], q[1], q[2], q[3], q[4])
    
    # Calculate positional error (distance between x,y,z coordinates)
    pos_error = np.linalg.norm(current_tf[:3, 3] - target_matrix[:3, 3])
    
    # Calculate rotational error (difference in the 3x3 rotation matrices)
    rot_error = np.linalg.norm(current_tf[:3, :3] - target_matrix[:3, :3])
    
    # The original ikpy script weighted rotational error by 0.2
    total_error = pos_error + 0.2 * rot_error
    return total_error


def solve_single_pose_scipy(
    x,
    y,
    z,
    roll,
    pitch,
    yaw,
    q_init=None,
    pos_tol=5e-3,
    rot_tol=np.deg2rad(1.0),
    max_iter=200,
):
    target_matrix = create_tf_matrix(x, y, z, roll, pitch, yaw)
    
    # Default starting guess if none is provided (all zeros)
    if q_init is None:
        q_init = np.zeros(NUM_ACTIVE_JOINTS)
    else:
        q_init = np.asarray(q_init, dtype=float)

    # Use Scipy to find the joint angles that minimize the objective function
    # L-BFGS-B is a highly efficient numerical optimization algorithm.
    result = minimize(
        _ik_objective_function,
        q_init,
        args=(target_matrix,),
        method='L-BFGS-B',
        bounds=JOINT_LIMITS,
        options={'maxiter': max_iter, 'disp': False}
    )
    
    q_raw = result.x
    
    # Evaluate the final errors using our final joint angles
    final_tf = forward_kinematics_full(q_raw[0], q_raw[1], q_raw[2], q_raw[3], q_raw[4])
    pos_error = float(np.linalg.norm(final_tf[:3, 3] - target_matrix[:3, 3]))
    rot_error = float(np.linalg.norm(final_tf[:3, :3] - target_matrix[:3, :3]))
    total_error = float(pos_error + 0.2 * rot_error)
    
    return IKSolveResult(
        feasible=bool(pos_error <= pos_tol and rot_error <= rot_tol),
        q_raw=q_raw,
        q=np.round(q_raw, 4),
        pos_error_raw=pos_error,
        rot_error_raw=rot_error,
        error_raw=total_error,
    )


if __name__ == "__main__":
    test_poses = [
        ("I", [0.2000, 0.2000, 0.2000, 0.0000, 1.5700, 0.6500]),
        ("II", [0.2000, 0.1000, 0.4000, 0.0000, 0.0000, -1.5700]),
        ("III", [0.0000, 0.0000, 0.4000, 0.0000, -0.7850, 1.5700]),
        ("IV_a", [0.0000, 0.0000, 0.0700, 3.1410, 0.0000, 0.0000]),
        ("IV_b", [0.0000, 0.0452, 0.4500, -0.7850, 0.0000, 3.1410]),
    ]

    print("Single-pose numerical IK with Scipy\n")

    for label, pose in test_poses:
        x, y, z, roll, pitch, yaw = pose
        result = solve_single_pose_scipy(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw
        )
        print(f"Pose {label}: {pose}")
        print(f"  feasible={result.feasible}")
        print(
            f"  pos_error={result.pos_error_raw:.6f} "
            f"rot_error={result.rot_error_raw:.6f} total={result.error_raw:.6f}"
        )
        print(f"  q={result.q}")