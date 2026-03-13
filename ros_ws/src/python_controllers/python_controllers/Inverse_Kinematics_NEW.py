import numpy as np

# Hardware Constraints
BOUNDS = {
    'q1': (-2.0000,  2.0000),
    'q2': (-1.5700,  1.5700),
    'q3': (-1.5800,  1.5800),
    'q4': (-1.5700,  1.5700),
    'q5': (-3.1415,  3.1415)
}

def create_tf_matrix(tx, ty, tz, roll, pitch, yaw):
    Rx = np.array([[1, 0, 0, 0], [0, np.cos(roll), -np.sin(roll), 0], [0, np.sin(roll), np.cos(roll), 0], [0, 0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return T @ Rz @ Ry @ Rx

def check_limits(q):
    if q is None: return False
    q1, q2, q3, q4, q5 = q
    if not (BOUNDS['q1'][0] <= q1 <= BOUNDS['q1'][1]): return False
    if not (BOUNDS['q2'][0] <= q2 <= BOUNDS['q2'][1]): return False
    if not (BOUNDS['q3'][0] <= q3 <= BOUNDS['q3'][1]): return False
    if not (BOUNDS['q4'][0] <= q4 <= BOUNDS['q4'][1]): return False
    if not (BOUNDS['q5'][0] <= q5 <= BOUNDS['q5'][1]): return False
    return True

def ik_closed_form(x, y, z, rot_x, rot_y, rot_z):
    # 1. Pre-calculated URDF Constants
    L0 = 0.1190    
    L1 = 0.1160    
    L2 = 0.1350    
    L3 = 0.1351    
    A1 = -0.2437   
    A2 = -1.5323   
    
    # 2. Build Target Matrix & Extract Wrist Position
    # Using your world definition, we extract the Z-axis vector (column index 2) of the end effector 
    # as the approach direction, because the gripper's Z-axis points straight out of the hand.
    target_tf = create_tf_matrix(x, y, z, rot_x, rot_y, rot_z)
    approach_vector = target_tf[0:3, 2] 
    
    x_w = x - L3 * approach_vector[0]
    y_w = y - L3 * approach_vector[1]
    z_w = z - L3 * approach_vector[2]
    
    # 3. Base Yaw (q1)
    q1 = np.arctan2(y_w, x_w) - np.pi
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    
    # 4. Map to 2D Plane
    r = np.sqrt(x_w**2 + y_w**2)
    z_prime = z_w - L0
    
    D_squared = r**2 + z_prime**2
    if D_squared > (L1 + L2)**2 or D_squared < (L1 - L2)**2:
        return {"status": "Failed", "reason": "Target exceeds arm length"}

    # 5. Solve Elbow (q3)
    cos_beta = np.clip((D_squared - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    beta_up = np.arccos(cos_beta)
    beta_down = -np.arccos(cos_beta)
    
    q3_up = beta_up - A2 + A1
    q3_down = beta_down - A2 + A1
    
    # 6. Solve Shoulder (q2)
    gamma = np.arctan2(z_prime, r)
    delta_up = np.arctan2(L2 * np.sin(beta_up), L1 + L2 * np.cos(beta_up))
    delta_down = np.arctan2(L2 * np.sin(beta_down), L1 + L2 * np.cos(beta_down))
    
    q2_up = gamma - delta_up - A1
    q2_down = gamma - delta_down - A1
    
    # 7. Solve Wrist Pitch (q4)
    # Project the approach vector onto the r-z plane to find the 2D pitch angle
    r_dir = np.cos(q1 + np.pi) * approach_vector[0] + np.sin(q1 + np.pi) * approach_vector[1]
    pitch_2d = np.arctan2(approach_vector[2], r_dir)
    
    q4_up = pitch_2d - q2_up - q3_up
    q4_down = pitch_2d - q2_down - q3_down
    
    # 8. Solve Wrist Roll (q5)
    q5 = rot_x
    
    # 9. Format Output
    q_up = (np.round(q1, 4), np.round(q2_up, 4), np.round(q3_up, 4), np.round(q4_up, 4), np.round(q5, 4))
    q_down = (np.round(q1, 4), np.round(q2_down, 4), np.round(q3_down, 4), np.round(q4_down, 4), np.round(q5, 4))
    
    valid_up = check_limits(q_up)
    valid_down = check_limits(q_down)
    
    if valid_up: return {"status": "Success", "q": q_up, "config": "Elbow Up"}
    if valid_down: return {"status": "Success", "q": q_down, "config": "Elbow Down"}
    
    return {"status": "Failed", "reason": "Requires angles outside limits"}

# --- Test Execution ---
poses = {
    "I":    [0.2000, 0.2000, 0.2000, 0.0000, 1.5700, 0.6500],
    "II":   [0.2000, 0.1000, 0.4000, 0.0000, 0.0000, -1.5700],
    "III":  [0.0000, 0.0000, 0.4000, 0.0000, -0.7850, 1.5700],
    "IV_a": [0.0000, 0.0000, 0.0700, 3.1410, 0.0000, 0.0000],
    "IV_b": [0.0000, 0.0452, 0.4500, -0.7850, 0.0000, 3.1410]
}

for name, pose in poses.items():
    res = ik_closed_form(*pose)
    if res['status'] == 'Success':
        print(f"Pose {name}: {res['config']} -> {res['q']}")
    else:
        print(f"Pose {name}: Failed -> {res['reason']}")