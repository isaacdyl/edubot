import math

def analytical_ik_closed_form(X, Y, Z, pitch, l2, l3, l4):
    """
    Analytically solves the closed-form IK equations.
    Returns a list of all valid (q1, q2, q3, q4) configurations in radians.
    """
    valid_solutions = []

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
        
        valid_solutions.append((q1, q2, q3, q4))

    return valid_solutions

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    X, Y, Z = 200.0, 50.0, 100.0
    pitch = math.radians(0.0) 
    L2, L3, L4 = 111.67, 160.0, 150.0
    
    solutions = analytical_ik_closed_form(X, Y, Z, pitch, L2, L3, L4)
    
    if not solutions:
        print("Target is out of reach.")
    else:
        print(f"Found {len(solutions)} valid poses:\n")
        for i, (q1, q2, q3, q4) in enumerate(solutions):
            print(f"Pose {i+1}:")
            print(f"  q1: {q1:.4f} [rad]")
            print(f"  q2: {q2:.4f} [rad]")
            print(f"  q3: {q3:.4f} [rad]")
            print(f"  q4: {q4:.4f} [rad] \n")