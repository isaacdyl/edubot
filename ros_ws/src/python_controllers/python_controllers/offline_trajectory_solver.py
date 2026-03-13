import csv
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from python_controllers.Inverse_Kinematics_FINAL import ik_coordinate_descent_multi_start
    from python_controllers.trajectory_generator import generate_trajectory
except ModuleNotFoundError:
    from Inverse_Kinematics_FINAL import ik_coordinate_descent_multi_start
    from trajectory_generator import generate_trajectory


@dataclass
class OfflineTrajectoryConfig:
    shape: str = "square"
    plane: str = "yz"
    size: float = 0.04
    duration_s: float = 12.0
    rate_hz: float = 20.0
    center_x: float = 0.05
    center_y: float = 0.10
    center_z: float = 0.15
    roll: float = 0.0
    pitch: float = -3.14159 / 2
    yaw: float = 0.0
    pos_tol: float = 5e-3
    rot_tol: float = 0.15
    unique_eps: float = 0.05
    max_joint_jump: float = 0.30
    output_csv: str = "/tmp/offline_joint_trajectory.csv"


def _closest_to_previous(valid_solutions: List[dict], q_prev):
    if not valid_solutions:
        return None
    if q_prev is None:
        return valid_solutions[0]
    return min(
        valid_solutions,
        key=lambda item: float(np.linalg.norm(item["q_raw"] - q_prev)),
    )


def solve_offline_trajectory(cfg: OfflineTrajectoryConfig):
    trajectory = generate_trajectory(
        shape_name=cfg.shape,
        center_xyz=(cfg.center_x, cfg.center_y, cfg.center_z),
        plane=cfg.plane,
        size=cfg.size,
        duration_s=cfg.duration_s,
        rate_hz=cfg.rate_hz,
    )
    print(
        "Generated {} trajectory points to solve for shape='{}' plane='{}'".format(
            len(trajectory), cfg.shape, cfg.plane
        )
    )

    solved_rows = []
    q_prev = None

    for point in trajectory:
        out = ik_coordinate_descent_multi_start(
            point.x,
            point.y,
            point.z,
            cfg.roll,
            cfg.pitch,
            cfg.yaw,
            q_prev=q_prev,
            pos_tol=cfg.pos_tol,
            rot_tol=cfg.rot_tol,
            unique_eps=cfg.unique_eps,
        )
        chosen = _closest_to_previous(out["valid_solutions"], q_prev)
        if chosen is None:
            raise RuntimeError(
                "failed to solve trajectory point at t={:.3f} xyz=({:.3f}, {:.3f}, {:.3f})".format(
                    point.t, point.x, point.y, point.z
                )
            )

        if q_prev is not None:
            jump = float(np.max(np.abs(chosen["q_raw"] - q_prev)))
            if jump > cfg.max_joint_jump:
                raise RuntimeError(
                    "joint jump {:.3f} exceeds max_joint_jump at t={:.3f}".format(
                        jump, point.t
                    )
                )

        q_prev = chosen["q_raw"].copy()
        solved_rows.append(
            {
                "t": point.t,
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "q1": q_prev[0],
                "q2": q_prev[1],
                "q3": q_prev[2],
                "q4": q_prev[3],
                "q5": q_prev[4],
                "pos_error": chosen["pos_error_raw"],
                "rot_error": chosen["rot_error_raw"],
                "error": chosen["error_raw"],
            }
        )
    return solved_rows


def write_trajectory_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "x",
                "y",
                "z",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "pos_error",
                "rot_error",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = OfflineTrajectoryConfig()
    rows = solve_offline_trajectory(cfg)
    write_trajectory_csv(cfg.output_csv, rows)
    print(f"Solved {len(rows)} trajectory points")
    print(f"Wrote {cfg.output_csv}")


if __name__ == "__main__":
    main()
