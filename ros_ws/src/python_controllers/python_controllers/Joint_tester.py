import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import time

# ─── Joint names (from URDF) ──────────────────────────────────────────────────
JOINT_NAMES = [
    'Shoulder_Rotation',  # q1 — yaw,   limit: -2.0  to  2.0
    'Shoulder_Pitch',     # q2 — pitch,  limit: -1.57 to  1.57
    'Elbow',              # q3 — elbow,  limit: -1.58 to  1.58
    'Wrist_Pitch',        # q4 — wrist,  limit: -1.57 to  1.57
]

# Wrist_Roll is a separate revolute joint (-π to π) not used in FK workspace,
# but you can add it to poses if you want to test roll.
# Gripper jaw joint is 'Gripper': -0.2 to 2.0

# ─── Test poses ───────────────────────────────────────────────────────────────
# Format: [Shoulder_Rotation, Shoulder_Pitch, Elbow, Wrist_Pitch]
TEST_POSES = {
    # Safe upright home — arm points straight up
    'home':             [ 0.0,   0.0,   0.0,    0.0  ],

    # Reach out horizontally in front at safe height
    'reach_forward':    [ 0.0,   0.4,     0.8,    0.3  ],

    # Sweep left and right at safe height
    'reach_left':       [ 1.57,  0.4,     0.8,    0.3  ],
    'reach_right':      [-1.57,  0.4,     0.8,    0.3  ],

    # Diagonal reaches
    'reach_fwd_left':   [ 0.8,   0.4,     0.8,    0.3  ],
    'reach_fwd_right':  [-0.8,   0.4,     0.8,    0.3  ],

    # Arm raised high — full extension upward
    'reach_high':       [ 0.0,   1.2,     0.5,    0.0  ],

    # Elbow bent, arm compact
    'elbow_bent':       [ 0.0,   0.8,     1.2,    0.0  ],

    # Wrist pitch test — arm safely raised first
    'wrist_test':       [ 0.0,   0.6,     0.5,    1.2  ],

    # Shoulder rotation sweep — arm held high
    'rotate_left':      [ 1.8,   0.785,   0.0,    0.0  ],
    'rotate_right':     [-1.8,   0.785,   0.0,    0.0  ],
}

def get_q(pose_name):
    """Return joint angles for a named pose."""
    if pose_name not in TEST_POSES:
        print(f"Unknown pose '{pose_name}'. Available: {list(TEST_POSES.keys())}")
        return None
    q = TEST_POSES[pose_name]
    print(f"[{pose_name}]  Shoulder_Rotation={q[0]:.3f}  Shoulder_Pitch={q[1]:.3f}  Elbow={q[2]:.3f}  Wrist_Pitch={q[3]:.3f}")
    return q


def get_q_sweep(joint_index, n=5, use_constrained=True):
    """
    Return a list of poses sweeping one joint through its range
    while keeping all others at zero.

    joint_index: 0=Shoulder_Rotation, 1=Shoulder_Pitch, 2=Elbow, 3=Wrist_Pitch
    """
    # Limits taken directly from URDF <limit> tags
    LIMITS = {
        0: (-2.0,  2.0 ),   # Shoulder_Rotation
        1: (-1.57, 1.57),   # Shoulder_Pitch
        2: (-1.58, 1.58),   # Elbow
        3: (-1.57, 1.57),   # Wrist_Pitch
    }
    lo, hi = LIMITS[joint_index]
    sweep = []
    for val in np.linspace(lo, hi, n):
        q = [0.0, 0.0, 0.0, 0.0]
        q[joint_index] = float(val)
        sweep.append(q)
    print(f"Sweep for joint {joint_index+1}: {[round(q[joint_index],3) for q in sweep]}")
    return sweep


# ─── ROS2 Node ────────────────────────────────────────────────────────────────

class JointTester(Node):
    """
    Sends joint angle commands for testing.

    SIM:   publishes to /joint_trajectory_controller/joint_trajectory
    ROBOT: same topic — hardware interface picks it up automatically
    
    Usage modes:
      1. Single pose:   node.send_pose('reach_forward')
      2. Sweep:         node.run_sweep(joint_index=1, n=7)
      3. Sequence:      node.run_sequence(['home', 'elbow_up', 'reach_forward', 'home'])
    """

    def __init__(self, move_time=2.0):
        super().__init__('joint_tester')
        self.move_time = move_time

        # Topic name — verify with:
        # ros2 topic list | grep trajectory
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_cmds',
            10
        )
        self.state_pub = self.create_publisher(JointState, '/joint_states_cmd', 10)
        time.sleep(0.5)
        self.get_logger().info('JointTester ready. Joints: ' + str(JOINT_NAMES))

    def _make_trajectory(self, q, move_time=None):
        """Build a JointTrajectory for a single target pose.
        q = [Shoulder_Rotation, Shoulder_Pitch, Elbow, Wrist_Pitch]
        Wrist_Roll is held at 0.0 unless you extend q to length 5.
        """
        t = move_time or self.move_time

        # Pad Wrist_Roll to 0.0 if not supplied
        all_names = JOINT_NAMES + ['Wrist_Roll']
        all_q     = list(q) + [0.0] if len(q) == 4 else list(q)

        msg = JointTrajectory()
        msg.joint_names = all_names
        msg.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in all_q]
        pt.velocities = [0.0] * len(all_names)
        pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        msg.points = [pt]
        return msg

    def send_pose(self, pose_name):
        """Send a single named pose."""
        q = get_q(pose_name)
        if q is None:
            return
        self.traj_pub.publish(self._make_trajectory(q))
        self.get_logger().info(f'Sent pose: {pose_name}')

    def send_q(self, q, label='custom'):
        """Send a raw list of joint angles [q1, q2, q3, q4]."""
        self.traj_pub.publish(self._make_trajectory(q))
        self.get_logger().info(f'Sent {label}: {[round(v,3) for v in q]}')

    def run_sweep(self, joint_index, n=7, pause=2.5):
        """
        Sweep one joint through its full range.
        All other joints stay at zero.
        """
        poses = get_q_sweep(joint_index, n)
        self.get_logger().info(
            f'Sweeping joint {joint_index+1} through {n} positions...'
        )
        for i, q in enumerate(poses):
            self.get_logger().info(f'  Step {i+1}/{n}: q{joint_index+1}={q[joint_index]:.3f}')
            self.send_q(q, label=f'sweep_j{joint_index+1}_{i+1}')
            time.sleep(pause)

        # Return home after sweep
        self.send_pose('home')

    def run_sequence(self, pose_names, pause=3.0):
        """
        Execute a sequence of named poses with a pause between each.
        Example: run_sequence(['home', 'elbow_up', 'reach_forward', 'home'])
        """
        self.get_logger().info(f'Running sequence: {pose_names}')
        for name in pose_names:
            self.send_pose(name)
            time.sleep(pause)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = JointTester(move_time=2.0)

    # ── Choose what to run ──────────────────────────────────────────────────
    # Option A: send a single named pose
    # node.send_pose('reach_forward')

    # Option B: sweep one joint (0=yaw, 1=pitch, 2=elbow, 3=wrist)
    # node.run_sweep(joint_index=1, n=7, pause=2.5)

    # Option C: run a full sequence
    node.run_sequence(
        ['home', 'reach_forward', 'reach_left', 'reach_right', 'elbow_up', 'home'],
        pause=3.0
    )

    # Option D: send raw angles directly
    # node.send_q([0.5, -0.8, -1.0, 0.3], label='my_test')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()