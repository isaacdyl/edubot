import math
import numpy as np
from scipy.optimize import least_squares

<<<<<<< HEAD
LIMITS_CONSTRAINED = {
    'q1': (-2.0,    2.0   ), 
    'q2': (-1.57,   1.57  ),
    'q3': (-1.58,   1.58  ),
    'q4': (-1.57,   1.57  ),
}

L_SH_Y, L_SH_Z   = -0.0452,  0.0165
L_UA_Y, L_UA_Z   = -0.0306,  0.1025
L_LA_X, L_LA_Y   =  0.11257, -0.028
L_WR_X, L_WR_Y   =  0.0052, -0.1349
L_GR_X, L_GC_Z   = -0.0601,  0.075




upper_arm_coordinates=[0, L_SH_Y+L_UA_Y, L_SH_Z+L_UA_Z]



def analytical_ik_closed_form(Y, X, Z, pitch):
=======

def _default_joint_limits():
    return [
        (-2.0, 2.0),      # q1 Shoulder_Rotation
        (-1.57, 1.57),    # q2 Shoulder_Pitch
        (-1.58, 1.58),    # q3 Elbow
        (-1.57, 1.57),    # q4 Wrist_Pitch
    ]


def _limit_violations(q, joint_limits):
    violations = []
    for i, (lo, hi) in enumerate(joint_limits):
        if q[i] < lo or q[i] > hi:
            violations.append(
                {
                    "joint_index": i,
                    "value": q[i],
                    "min": lo,
                    "max": hi,
                }
            )
    return violations


def analytical_ik_closed_form(X, Y, Z, pitch, l2, l3, l4, joint_limits=None, track_invalid=False):
>>>>>>> origin/main
    """
    Analytically solves the closed-form IK equations.

    If track_invalid=False:
      Returns only valid (q1, q2, q3, q4) solutions (possibly empty list).

    If track_invalid=True:
      Returns a dict with:
        - valid_solutions
        - invalid_solutions
        - invalid_reasons
    """
<<<<<<< HEAD
    
    upper_arm_coordinates=[0, L_SH_Y+L_UA_Y, L_SH_Z+L_UA_Z]
    X,Y,Z = X - upper_arm_coordinates[0], Y - upper_arm_coordinates[1], Z - upper_arm_coordinates[2] # Adjust target position to shoulder frame
    l2= (L_LA_X**2+L_LA_Y**2)**0.5
    l3= (L_WR_X**2+L_WR_Y**2)**0.5
    l4= (abs(L_GR_X)+abs(L_GC_Z))
    valid_solutions = []
    X = - X # Invert X to match the robot's coordinate system
=======
    if joint_limits is None:
        joint_limits = _default_joint_limits()

    valid_solutions = []
    invalid_solutions = []
    invalid_reasons = []

>>>>>>> origin/main
    # Step 1: Base angle
    q1 = math.atan2(Y, X)

    # Step 2: Simplify with knowns
    R = math.hypot(X, Y)
    R_prime = R - l4 * math.cos(pitch)
    Z_prime = Z + l4 * math.sin(pitch)

    # Step 3: Set up A*sin(q2) + B*cos(q2) = C
    A = R_prime
    B = Z_prime
    C = (R_prime**2 + Z_prime**2 + l2**2 - l3**2) / (2 * l2)
    
    r = math.hypot(A, B)
    
    # Check if physical reach is impossible
    if abs(C) > r:
        if not track_invalid:
            return []
        invalid_solutions.append(None)
        invalid_reasons.append(
            {
                "code": "geometry_unreachable",
                "message": "wrist center is outside reachable annulus for links l2/l3",
            }
        )
        return {
            "valid_solutions": valid_solutions,
            "invalid_solutions": invalid_solutions,
            "invalid_reasons": invalid_reasons,
        }
        
    phi = math.atan2(B, A)
    s = C / r # The sine of (q2 + phi)
    
    # Two possible solutions for q2
    q2_sol1 = math.asin(s) - phi
    q2_sol2 = math.pi - math.asin(s) - phi

    # Step 4: Solve downstream joints for both poses
    for q2 in [q2_sol1, q2_sol2]:
        # Keep q2 within standard -pi to pi range
        q2 = math.atan2(math.sin(q2), math.cos(q2)) 
        
        cos_alpha = (R_prime - l2 * math.sin(q2)) / l3
        sin_alpha = (Z_prime - l2 * math.cos(q2)) / l3
        
        alpha = math.atan2(sin_alpha, cos_alpha)
        
        q3 = - (alpha + q2)
        q4 = pitch + alpha
<<<<<<< HEAD
        
        # Normalize all angles to [-pi, pi]
        q1_norm = math.atan2(math.sin(q1), math.cos(q1))
        q2_norm = math.atan2(math.sin(q2), math.cos(q2))
        q3_norm = math.atan2(math.sin(q3), math.cos(q3))
        q4_norm = math.atan2(math.sin(q4), math.cos(q4))
        
        # Apply joint constraints
        if all(LIMITS_CONSTRAINED[f'q{i}'][0] <= q <= LIMITS_CONSTRAINED[f'q{i}'][1] for i, q in enumerate([q1_norm, q2_norm, q3_norm, q4_norm], 1)):
            valid_solutions.append((q1_norm, q2_norm, q3_norm, q4_norm))
=======
>>>>>>> origin/main

        q = (q1, q2, q3, q4)
        violations = _limit_violations(q, joint_limits)
        if len(violations) == 0:
            valid_solutions.append(q)
        elif track_invalid:
            invalid_solutions.append(q)
            invalid_reasons.append(
                {
                    "code": "joint_limit_violation",
                    "message": "one or more joints violate limits",
                    "violations": violations,
                }
            )

    if not track_invalid:
        return valid_solutions
    return {
        "valid_solutions": valid_solutions,
        "invalid_solutions": invalid_solutions,
        "invalid_reasons": invalid_reasons,
    }


def check_ik_solution_validity(X, Y, Z, pitch, l2, l3, l4, joint_limits=None):
    """
    Verify IK solutions using joint limits only.
    joint_limits format: [(q1_min,q1_max), (q2_min,q2_max), (q3_min,q3_max), (q4_min,q4_max)]
    """
    out = analytical_ik_closed_form(
        X, Y, Z, pitch, l2, l3, l4,
        joint_limits=joint_limits,
        track_invalid=True,
    )
    valid = out["valid_solutions"]
    invalid = [q for q in out["invalid_solutions"] if q is not None]
    reasons = out["invalid_reasons"]
    checks = []
    for i, q in enumerate(valid):
        checks.append({"index": i, "q": q, "violations": [], "valid": True})
    for i, q in enumerate(invalid):
        reason = reasons[i] if i < len(reasons) else {}
        checks.append(
            {
                "index": len(valid) + i,
                "q": q,
                "violations": reason.get("violations", []),
                "valid": False,
                "reason_code": reason.get("code", "unknown"),
            }
        )

    return {
        "target": (X, Y, Z, pitch),
        "num_solutions": len(valid) + len(invalid),
        "any_valid": len(valid) > 0,
        "checks": checks,
    }


def ik_feasibility_reason(X, Y, Z, pitch, l2, l3, l4, joint_limits=None):
    """Return a short reason based on geometry and joint-limit feasibility checks."""
    out = analytical_ik_closed_form(
        X, Y, Z, pitch, l2, l3, l4,
        joint_limits=joint_limits,
        track_invalid=True,
    )
    if len(out["valid_solutions"]) > 0:
        return "feasible (at least one IK branch satisfies all joint limits)"

    for reason in out["invalid_reasons"]:
        if reason.get("code") == "geometry_unreachable":
            return reason.get("message", "geometrically unreachable target")
    if len(out["invalid_solutions"]) > 0:
        return "IK branches exist, but all violate joint limits"

    R = math.hypot(X, Y)
    R_prime = R - l4 * math.cos(pitch)
    Z_prime = Z + l4 * math.sin(pitch)

    A = R_prime
    B = Z_prime
    C = (R_prime**2 + Z_prime**2 + l2**2 - l3**2) / (2 * l2)
    r = math.hypot(A, B)

    if r == 0.0:
        return "degenerate wrist-center geometry (r=0)"
    if abs(C) > r:
        return "wrist center is outside reachable annulus for links l2/l3"
    return "no solution for this model after branch evaluation"

def numerical_ik_solver(X_target, Y_target, Z_target, pitch_target):
    """
    Numerically solves the IK system using SciPy's least_squares.
    
    Args:
        X_target, Y_target, Z_target: The 3D coordinates to reach.
        pitch_target: The known angle combination target in radians.
        l2, l3, l4: Link lengths.
        initial_guess: List or tuple of [q1, q2, q3, q4] guesses in radians.
    
    Returns:
        Array of [q1, q2, q3, q4] in radians if successful.
    """
    # 1. Format the bounds for SciPy
    lower_bounds = [
        LIMITS_CONSTRAINED['q1'][0], 
        LIMITS_CONSTRAINED['q2'][0], 
        LIMITS_CONSTRAINED['q3'][0], 
        LIMITS_CONSTRAINED['q4'][0]
    ]
    upper_bounds = [
        LIMITS_CONSTRAINED['q1'][1], 
        LIMITS_CONSTRAINED['q2'][1], 
        LIMITS_CONSTRAINED['q3'][1], 
        LIMITS_CONSTRAINED['q4'][1]
    ]
    
    # 2. Calculate the safest initial guess (the exact midpoint of the bounds)
    safe_guess = [(low + high) / 2.0 for low, high in zip(lower_bounds, upper_bounds)]
    X_target,Y_target,Z_target = X_target - upper_arm_coordinates[0], Y_target - upper_arm_coordinates[1], Z_target - upper_arm_coordinates[2] # Adjust target position to shoulder frame
    l2= (L_LA_X**2+L_LA_Y**2)**0.5
    l3= (L_WR_X**2+L_WR_Y**2)**0.5
    l4= (abs(L_GR_X)+abs(L_GC_Z))
    
    def error_function(vars):
        """
        This function calculates the difference between where the arm IS 
        based on the current guess, and where you WANT it to be. 
        The solver minimizes this error to 0.
        """
        
        q1, q2, q3, q4 = vars
        
        # Note: I have included the -q3 substitution here. 
        # If your pitch constraint equation differs, change the signs here!
        angle_elbow = -q3 - q2
        angle_wrist = q2 + q4 + q3 # Assuming pitch was q2+q4-q3, substituting -q3 makes it +q3
        
        # Calculate current position based on guesses
        Y_current = (l2 * np.sin(q2) + l3 * np.cos(angle_elbow) + l4 * np.cos(angle_wrist)) * np.sin(q1)
        Z_current = (l2 * np.cos(q2) + l3 * np.sin(angle_elbow) - l4 * np.sin(angle_wrist))
        X_current = -(l2 * np.sin(q2) + l3 * np.cos(angle_elbow) + l4 * np.cos(angle_wrist)) * np.cos(q1)
        
        # Calculate current orientation constraint
        pitch_current = angle_wrist
        
        # Return the errors (Target - Current)
        return [
            X_current - X_target,
            Y_current - Y_target,
            Z_current - Z_target,
            pitch_current - pitch_target
        ]

    # Run the Levenberg-Marquardt algorithm ('lm'), excellent for unconstrained IK
    result = least_squares(
        error_function, 
        x0=safe_guess, 
        bounds=(lower_bounds, upper_bounds),  
        method='trf'                          
        )
    
    # Check if the solver successfully converged to a solution
    if result.success and result.cost < 1e-2:
        return result.x
    else:
        print(f"Solver stopped. Message: {result.message}")
        return None

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
<<<<<<< HEAD
    example_poses = [
    [0.2, 0.2, 0.2, 1.57, 0.00],
    [0.2, 0.1, 0.4, 0.0, 1.57],
    [0.0, 0.3, 0.45, 0.785, 0.785],
    [0.0, 0.0, 0.07, 3.141, 0.0],
    [0.0, 0.0452, 0.45, 0.785, 3.141]
     ]
    
    for i in example_poses:
        X, Y, Z, pitch = i[0], i[1], i[2], i[3] #Converting meters to mm
        print(f"Testing pose: X={X}, Y={Y}, Z={Z}, pitch={pitch} rad")
        #solutions = analytical_ik_closed_form(X=X, Y=Y, Z=Z, pitch=pitch)
        solutions = numerical_ik_solver(X_target=X, Y_target=Y, Z_target=Z, pitch_target=pitch)
        print(solutions)
=======
    # Link lengths in meters (consistent with shape_follower defaults)
    L2, L3, L4 = 0.11167, 0.16000, 0.15000

    print("Assignment Pose Feasibility Check")
    print("Note: This IK model solves [x, y, z, pitch]. Roll is not modeled.\n")

    assignment_poses = [
        ("I",   [0.2, 0.2,   0.2,   1.57,  0.0]),
        ("II",  [0.2, 0.1,   0.4,   0.0,   1.57]),
        ("III", [0.0, 0.0,   0.45,  0.785, 0.785]),
        ("IV",  [0.0, 0.0,   0.07,  3.141, 0.0]),
        ("V",   [0.0, 0.0452, 0.45, 0.785, 3.141]),
    ]

    print("a) IK solutions (YES/NO):")
    for label, pose in assignment_poses:
        x, y, z, pitch, roll = pose
        out = analytical_ik_closed_form(x, y, z, pitch, L2, L3, L4, track_invalid=True)
        total_branches = len(out["valid_solutions"]) + sum(
            1 for q in out["invalid_solutions"] if q is not None
        )
        yes_no = "YES" if len(out["valid_solutions"]) > 0 else "NO"
        print(f"  {label}. {pose} -> {yes_no} ({total_branches} IK branch(es))")

    print("\nb) Why NO for unsolved poses:")
    for label, pose in assignment_poses:
        x, y, z, pitch, roll = pose
        out = analytical_ik_closed_form(x, y, z, pitch, L2, L3, L4, track_invalid=True)
        if len(out["valid_solutions"]) > 0:
            continue
        reason = ik_feasibility_reason(x, y, z, pitch, L2, L3, L4)
        print(f"  {label}. {pose} -> {reason}")
        for i, q in enumerate(out["invalid_solutions"]):
            reason_i = out["invalid_reasons"][i] if i < len(out["invalid_reasons"]) else {}
            if reason_i.get("code") != "joint_limit_violation":
                continue
            violation_txt = ", ".join(
                f"q{v['joint_index']+1}={v['value']:.3f} not in [{v['min']:.3f}, {v['max']:.3f}]"
                for v in reason_i.get("violations", [])
            )
            print(f"     branch {i}: {violation_txt}")
>>>>>>> origin/main
