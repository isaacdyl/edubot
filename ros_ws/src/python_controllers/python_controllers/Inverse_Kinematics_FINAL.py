import numpy as np

try:
    from python_controllers.Forward_Kinematics_FINAL import (
        create_tf_matrix,
        forward_kinematics_full,
    )
except ModuleNotFoundError:
    from Forward_Kinematics_FINAL import create_tf_matrix, forward_kinematics_full


BOUNDS_MIN = np.array([-2.0000, -1.5700, -1.5800, -1.5700, -3.1415], dtype=float)
BOUNDS_MAX = np.array([2.0000, 1.5700, 1.5800, 1.5700, 3.1415], dtype=float)


def calculate_errors(T_current, T_target):
    """Return position, rotation, and weighted total error."""
    pos_error = np.linalg.norm(T_target[0:3, 3] - T_current[0:3, 3])
    rot_error = np.linalg.norm(T_target[0:3, 0:3] - T_current[0:3, 0:3])
    total_error = pos_error + 0.2 * rot_error
    return float(pos_error), float(rot_error), float(total_error)


def _build_result(q, pos_error, rot_error, total_error, iters, pos_tol, rot_tol):
    success = bool(pos_error <= pos_tol and rot_error <= rot_tol)
    return {
        "success": success,
        "q_raw": q.copy(),
        "q": np.round(q, 3),
        "pos_error_raw": float(pos_error),
        "rot_error_raw": float(rot_error),
        "error_raw": float(total_error),
        "pos_error": round(float(pos_error), 4),
        "rot_error": round(float(rot_error), 4),
        "error": round(float(total_error), 4),
        "iters": int(iters),
    }


def ik_coordinate_descent(
    x,
    y,
    z,
    rot_x,
    rot_y,
    rot_z,
    q_init=None,
    max_iters=5000,
    pos_tol=5e-3,
    rot_tol=0.15,
    init_step_size=0.0075,
    min_step=1e-6,
):
    T_target = create_tf_matrix(x, y, z, rot_x, rot_y, rot_z)

    if q_init is None:
        q = np.zeros(5, dtype=float)
    else:
        q = np.clip(np.asarray(q_init, dtype=float), BOUNDS_MIN, BOUNDS_MAX)

    step_size = float(init_step_size)
    T_current = forward_kinematics_full(*q)
    pos_err, rot_err, current_error = calculate_errors(T_current, T_target)

    for iteration in range(max_iters):
        improved = False

        for i in range(5):
            for direction in (1.0, -1.0):
                q_test = q.copy()
                q_test[i] += direction * step_size
                q_test = np.clip(q_test, BOUNDS_MIN, BOUNDS_MAX)

                T_test = forward_kinematics_full(*q_test)
                pos_test, rot_test, total_test = calculate_errors(T_test, T_target)

                if total_test < current_error:
                    q = q_test
                    T_current = T_test
                    pos_err, rot_err, current_error = pos_test, rot_test, total_test
                    improved = True
                    break
            if improved:
                continue

        if not improved:
            step_size *= 0.5
            if step_size < min_step:
                break

        if pos_err <= pos_tol and rot_err <= rot_tol:
            return _build_result(
                q, pos_err, rot_err, current_error, iteration, pos_tol, rot_tol
            )

    return _build_result(q, pos_err, rot_err, current_error, max_iters, pos_tol, rot_tol)


def generate_initial_guesses(num_random=5, seed=42, q_prev=None):
    rng = np.random.default_rng(seed)
    guesses = []
    if q_prev is not None:
        guesses.append(np.clip(np.asarray(q_prev, dtype=float), BOUNDS_MIN, BOUNDS_MAX))

    guesses.extend(
        [
            np.zeros(5, dtype=float),
            BOUNDS_MIN.copy(),
            BOUNDS_MAX.copy(),
            0.5 * (BOUNDS_MIN + BOUNDS_MAX),
        ]
    )

    for _ in range(num_random):
        guesses.append(rng.uniform(BOUNDS_MIN, BOUNDS_MAX))
    return guesses


def _is_unique_solution(q, results, unique_eps):
    for result in results:
        if np.linalg.norm(q - result["q_raw"]) < unique_eps:
            return False
    return True


def ik_coordinate_descent_multi_start(
    x,
    y,
    z,
    rot_x,
    rot_y,
    rot_z,
    initial_guesses=None,
    q_prev=None,
    max_iters=5000,
    pos_tol=5e-3,
    rot_tol=0.15,
    unique_eps=0.05,
    num_random=5,
    seed=42,
):
    if initial_guesses is None:
        initial_guesses = generate_initial_guesses(
            num_random=num_random, seed=seed, q_prev=q_prev
        )

    all_results = []
    valid_results = []

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
            pos_tol=pos_tol,
            rot_tol=rot_tol,
        )
        if not _is_unique_solution(result["q_raw"], all_results, unique_eps):
            continue
        all_results.append(result)
        if result["success"]:
            valid_results.append(result)

    all_results.sort(key=lambda item: item["error_raw"])
    valid_results.sort(key=lambda item: item["error_raw"])

    return {
        "valid_solutions": valid_results,
        "all_solutions": all_results,
    }


if __name__ == "__main__":
    poses = {
        "I": [0.2000, 0.2000, 0.2000, 0.0000, 1.5700, 0.6500],
        "II": [0.2000, 0.1000, 0.4000, 0.0000, 0.0000, -1.5700],
        "III": [0.0000, 0.0000, 0.4000, 0.0000, -0.7850, 1.5700],
        "IV_a": [0.0000, 0.0000, 0.0700, 3.1410, 0.0000, 0.0000],
        "IV_b": [0.0000, 0.0452, 0.4500, -0.7850, 0.0000, 3.1410],
    }

    print("Running multi-start numerical IK...\n")
    for name, pose in poses.items():
        out = ik_coordinate_descent_multi_start(*pose)
        print(f"Pose {name}:")
        if not out["all_solutions"]:
            print("  no solutions returned")
            continue
        for idx, res in enumerate(out["all_solutions"], start=1):
            status = "valid" if res["success"] else "approx"
            print(
                f"  Solution {idx} ({status}) "
                f"pos={res['pos_error']:.4f} rot={res['rot_error']:.4f} total={res['error']:.4f}"
            )
            print(f"    q={res['q']}")
        print()
