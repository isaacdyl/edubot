import numpy as np

def inverse_kinematics_closed_form(x, y, z, target_pitch, target_roll):
    """
    Calculates the joint angles for the SO-ARM101 given a target XYZ and orientation.
    Returns a dictionary with 'elbow_up', 'elbow_down', or None if unreachable.
    """
    
    # --- 1. Pre-calculated URDF Constants ---
    L0 = 0.1190    # Base Z height
    L1 = 0.1160    # Upper arm effective length
    L2 = 0.1350    # Lower arm effective length
    L3 = 0.1351    # Tool length (Wrist to Gripper Center)
    A1 = -0.2437   # Upper arm internal angle offset (rad)
    A2 = -1.5323   # Lower arm internal angle offset (rad)
    
    # --- 2. Hardware Constraints ---
    bounds = {
        'q1': (-2.0000,  2.0000),
        'q2': (-1.5700,  1.5700),
        'q3': (-1.5800,  1.5800),
        'q4': (-1.5700,  1.5700),
        'q5': (-3.1415,  3.1415)  # Updated wrist roll limit
    }
    
    def check_limits(q1, q2, q3, q4, q5):
        if not (bounds['q1'][0] <= q1 <= bounds['q1'][1]): return False
        if not (bounds['q2'][0] <= q2 <= bounds['q2'][1]): return False
        if not (bounds['q3'][0] <= q3 <= bounds['q3'][1]): return False
        if not (bounds['q4'][0] <= q4 <= bounds['q4'][1]): return False
        if not (bounds['q5'][0] <= q5 <= bounds['q5'][1]): return False
        return True

    # --- 3. Step 1: Kinematic Decoupling (Find Wrist Position) ---
    yaw = np.arctan2(y, x)
    x_w = x - L3 * np.cos(target_pitch) * np.cos(yaw)
    y_w = y - L3 * np.cos(target_pitch) * np.sin(yaw)
    z_w = z - L3 * np.sin(target_pitch)
    
    # --- 4. Step 2: Solve Base Yaw (q1) ---
    q1 = np.arctan2(y_w, x_w)-np.pi  # Adjust for robot's coordinate system
    
    # Normalize q1 to strictly fall between -pi and pi
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    
    # --- 5. Step 3: Map to 2D Plane ---
    r = np.sqrt(x_w**2 + y_w**2)
    z_prime = z_w - L0
    
    # Check if the target wrist position is physically reachable
    D_squared = r**2 + z_prime**2
    if D_squared > (L1 + L2)**2 or D_squared < (L1 - L2)**2:
        return {"status": "Unreachable", "reason": "Target exceeds physical arm length"}

    # --- 6. Step 4: Solve Elbow Pitch (q3) ---
    cos_beta = (D_squared - L1**2 - L2**2) / (2 * L1 * L2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0) # Prevent float rounding errors
    
    beta_up = np.arccos(cos_beta)
    beta_down = -np.arccos(cos_beta)
    
    q3_up = beta_up - A2 + A1
    q3_down = beta_down - A2 + A1
    
    # --- 7. Step 5: Solve Shoulder Pitch (q2) ---
    gamma = np.arctan2(z_prime, r)
    
    delta_up = np.arctan2(L2 * np.sin(beta_up), L1 + L2 * np.cos(beta_up))
    delta_down = np.arctan2(L2 * np.sin(beta_down), L1 + L2 * np.cos(beta_down))
    
    q2_up = gamma - delta_up - A1
    q2_down = gamma - delta_down - A1
    
    # --- 8. Step 6: Solve Wrist Pitch (q4) and Roll (q5) ---
    q4_up = target_pitch - q2_up - q3_up
    q4_down = target_pitch - q2_down - q3_down
    q5 = target_roll
    
    # --- 9. Format Output and Check Hardware Limits ---
    result = {"status": "Success", "elbow_up": None, "elbow_down": None, "q1": np.round(q1, 4), "q2_up": np.round(q2_up, 4), "q3_up": np.round(q3_up, 4), "q4_up": np.round(q4_up, 4), "q2_down": np.round(q2_down, 4), "q3_down": np.round(q3_down, 4), "q4_down": np.round(q4_down, 4), "q5": np.round(q5, 4)}
    
    if check_limits(q1, q2_up, q3_up, q4_up, q5):
        result["elbow_up"] = (np.round(q1, 4), np.round(q2_up, 4), np.round(q3_up, 4), np.round(q4_up, 4), np.round(q5, 4))
        
    if check_limits(q1, q2_down, q3_down, q4_down, q5):
        result["elbow_down"] = (np.round(q1, 4), np.round(q2_down, 4), np.round(q3_down, 4), np.round(q4_down, 4), np.round(q5, 4))
        
    if not result["elbow_up"] and not result["elbow_down"]:
        return {"status": "Unreachable", "reason": "Target requires angles outside hardware servo limits"}
        
    return result
# --- Test Execution ---
poses = {
    "I":    [0.2, 0.2, 0.2, 0.000, 1.570, 0.650],
    "II":   [0.2, 0.1, 0.4, 0.000, 0.000, -1.570],
    "III":  [0.0, 0.0, 0.4, 0.000, -0.785, 1.570],
    "IV_a": [0.0, 0.0, 0.07, 3.141, 0.000, 0.000],
    "IV_b": [0.0, 0.0452, 0.45, -0.785, 0.000, 3.141]
}

for name, pose in poses.items():
    x, y, z, rot_x, rot_y, rot_z = pose
    print(f"Testing Pose {name}...")
    res = inverse_kinematics_closed_form(x, y, z, rot_y, rot_x)
    print(f"  Result: {res['status']} -> {res.get('reason', 'Valid Angles Calculated')}\n")
    if res['status'] == "Success":
        if res['elbow_up']:
            print(f"  Elbow Up Solution: q1={res['elbow_up'][0]}, q2={res['elbow_up'][1]}, q3={res['elbow_up'][2]}, q4={res['elbow_up'][3]}, q5={res['elbow_up'][4]}")
        else:
            print("  Elbow Up Solution: Not within hardware limits.")
        
        if res['elbow_down']:
            print(f"  Elbow Down Solution: q1={res['elbow_down'][0]}, q2={res['elbow_down'][1]}, q3={res['elbow_down'][2]}, q4={res['elbow_down'][3]}, q5={res['elbow_down'][4]}")
        else:
            print("  Elbow Down Solution: Not within hardware limits.")
    print("\n")