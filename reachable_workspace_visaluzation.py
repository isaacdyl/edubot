#!/usr/bin/env python3

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RevoluteJoint:
	origin_xyz: np.ndarray
	origin_rpy: np.ndarray
	axis: np.ndarray


def rot_x(angle: float) -> np.ndarray:
	c, s = np.cos(angle), np.sin(angle)
	return np.array(
		[
			[1.0, 0.0, 0.0],
			[0.0, c, -s],
			[0.0, s, c],
		]
	)


def rot_y(angle: float) -> np.ndarray:
	c, s = np.cos(angle), np.sin(angle)
	return np.array(
		[
			[c, 0.0, s],
			[0.0, 1.0, 0.0],
			[-s, 0.0, c],
		]
	)


def rot_z(angle: float) -> np.ndarray:
	c, s = np.cos(angle), np.sin(angle)
	return np.array(
		[
			[c, -s, 0.0],
			[s, c, 0.0],
			[0.0, 0.0, 1.0],
		]
	)


def rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
	roll, pitch, yaw = rpy
	return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
	axis = axis / np.linalg.norm(axis)
	x, y, z = axis
	c = np.cos(angle)
	s = np.sin(angle)
	one_minus_c = 1.0 - c
	return np.array(
		[
			[c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
			[y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
			[z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
		]
	)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
	transform = np.eye(4)
	transform[:3, :3] = rotation
	transform[:3, 3] = translation
	return transform


# Joint definitions copied from ros_ws/src/lerobot/urdf/lerobot.urdf
JOINTS = [
	RevoluteJoint(np.array([0.0, -0.0452, 0.0165]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
	RevoluteJoint(np.array([0.0, -0.0306, 0.1025]), np.array([0.0, -1.57079, 0.0]), np.array([0.0, 0.0, 1.0])),
	RevoluteJoint(np.array([0.11257, -0.028, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
	RevoluteJoint(np.array([0.0052, -0.1349, 0.0]), np.array([0.0, 0.0, 1.57079]), np.array([0.0, 0.0, 1.0])),
	RevoluteJoint(np.array([-0.0601, 0.0, 0.0]), np.array([0.0, -1.57079, 0.0]), np.array([0.0, 0.0, 1.0])),
]

# Fixed transform from gripper to gripper_center
TOOL_FIXED_TRANSLATION = np.array([0.0, 0.0, 0.075])
TOOL_FIXED_RPY = np.array([0.0, 0.0, 0.0])


def forward_kinematics(joint_angles: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
	if joint_angles.shape != (5,):
		raise ValueError("joint_angles must have shape (5,)")

	transform = np.eye(4)
	points = [transform[:3, 3].copy()]

	for angle, joint in zip(joint_angles, JOINTS):
		joint_origin = make_transform(rpy_to_rotation(joint.origin_rpy), joint.origin_xyz)
		transform = transform @ joint_origin
		points.append(transform[:3, 3].copy())
		transform = transform @ make_transform(axis_angle_to_rotation(joint.axis, float(angle)), np.zeros(3))

	tool_transform = make_transform(rpy_to_rotation(TOOL_FIXED_RPY), TOOL_FIXED_TRANSLATION)
	transform = transform @ tool_transform
	points.append(transform[:3, 3].copy())
	tool_position = transform[:3, 3].copy()
	return tool_position, points


def sample_workspace(num_samples: int, seed: int) -> np.ndarray:
	rng = np.random.default_rng(seed)
	samples = rng.uniform(-np.pi, np.pi, size=(num_samples, 5))
	points = np.zeros((num_samples, 3))
	for index, q in enumerate(samples):
		points[index], _ = forward_kinematics(q)
	return points


def set_equal_axes(ax: plt.Axes, xyz: np.ndarray) -> None:
	mins = xyz.min(axis=0)
	maxs = xyz.max(axis=0)
	centers = (mins + maxs) / 2.0
	radius = np.max(maxs - mins) / 2.0
	for axis_idx, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
		setter(centers[axis_idx] - radius, centers[axis_idx] + radius)


def visualize_workspace_with_sketch(num_samples: int, seed: int, save_path: str | None = None) -> None:
	workspace_points = sample_workspace(num_samples=num_samples, seed=seed)

	sketch_angles = np.array([0.0, -0.5, 1.0, -0.4, 0.6])
	tool_point, chain_points = forward_kinematics(sketch_angles)
	chain_xyz = np.vstack(chain_points)

	figure = plt.figure(figsize=(10, 8))
	axis = figure.add_subplot(111, projection="3d")

	axis.scatter(
		workspace_points[:, 0],
		workspace_points[:, 1],
		workspace_points[:, 2],
		s=1,
		alpha=0.20,
		c="#3b82f6",
		label=f"Reachable workspace ({num_samples:,} unconstrained samples)",
	)

	axis.plot(
		chain_xyz[:, 0],
		chain_xyz[:, 1],
		chain_xyz[:, 2],
		"-o",
		lw=2.0,
		ms=5.0,
		c="#111827",
		label="EduBot sketch",
	)

	axis.scatter(tool_point[0], tool_point[1], tool_point[2], c="#dc2626", s=45, label="t (tool center)")
	axis.text(tool_point[0], tool_point[1], tool_point[2], "  t", color="#dc2626", fontsize=10)

	axis.set_title("EduBot 3D Reachable Workspace (No Joint Constraints)")
	axis.set_xlabel("X [m]")
	axis.set_ylabel("Y [m]")
	axis.set_zlabel("Z [m]")
	axis.legend(loc="upper left")
	axis.grid(True, alpha=0.3)

	all_points = np.vstack((workspace_points, chain_xyz))
	set_equal_axes(axis, all_points)

	plt.tight_layout()
	if save_path:
		figure.savefig(save_path, dpi=250)
		print(f"Saved figure to: {save_path}")
	plt.show()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Visualize EduBot reachable workspace and include a robot sketch with tool point t."
	)
	parser.add_argument("--samples", type=int, default=25000, help="Number of random joint samples to draw.")
	parser.add_argument("--seed", type=int, default=7, help="Random seed.")
	parser.add_argument("--save", type=str, default=None, help="Optional output image path (PNG, PDF, etc.).")
	args = parser.parse_args()

	visualize_workspace_with_sketch(num_samples=args.samples, seed=args.seed, save_path=args.save)


if __name__ == "__main__":
	main()
