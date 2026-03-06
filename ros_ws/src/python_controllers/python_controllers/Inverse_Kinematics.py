import math
import numpy as np
from scipy.optimize import least_squares

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
    """
    Analytically solves the closed-form IK equations.
    Returns a list of all valid (q1, q2, q3, q4) configurations in radians.
    """
    
    upper_arm_coordinates=[0, L_SH_Y+L_UA_Y, L_SH_Z+L_UA_Z]
    X,Y,Z = X - upper_arm_coordinates[0], Y - upper_arm_coordinates[1], Z - upper_arm_coordinates[2] # Adjust target position to shoulder frame
    l2= (L_LA_X**2+L_LA_Y**2)**0.5
    l3= (L_WR_X**2+L_WR_Y**2)**0.5
    l4= (abs(L_GR_X)+abs(L_GC_Z))
    valid_solutions = []
    X = - X # Invert X to match the robot's coordinate system
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
        return [] 
        
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
        
        # Normalize all angles to [-pi, pi]
        q1_norm = math.atan2(math.sin(q1), math.cos(q1))
        q2_norm = math.atan2(math.sin(q2), math.cos(q2))
        q3_norm = math.atan2(math.sin(q3), math.cos(q3))
        q4_norm = math.atan2(math.sin(q4), math.cos(q4))
        
        # Apply joint constraints
        if all(LIMITS_CONSTRAINED[f'q{i}'][0] <= q <= LIMITS_CONSTRAINED[f'q{i}'][1] for i, q in enumerate([q1_norm, q2_norm, q3_norm, q4_norm], 1)):
            valid_solutions.append((q1_norm, q2_norm, q3_norm, q4_norm))

    return valid_solutions

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