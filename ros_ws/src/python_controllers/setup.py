from setuptools import find_packages, setup

package_name = 'python_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'ikpy'],
    zip_safe=True,
    maintainer='anton',
    maintainer_email='a.bredenbeck@tudelft.nl',
    description='Example Python ROS 2 trajectory controllers for EduBot/LeRobot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'example_pos_traj = python_controllers.example_pos_traj:main',
            'example_vel_traj = python_controllers.example_vel_traj:main',
            'DUMMY_FILE_Trajectory = python_controllers.DUMMY_FILE_Trajectory:main',
            'Position_Trajectory_Final = python_controllers.Position_Trajectory_Final:main',
            'workspace_visualizer = python_controllers.workspace_visualizer:main',
            'joint_tester = python_controllers.Joint_tester:main',
            'shape_follower = python_controllers.shape_follower:main',
            'constant_velocity_follower = python_controllers.constant_velocity_follower:main',
            'trajectory_precheck = python_controllers.trajectory_precheck:main',
            'offline_trajectory_solver = python_controllers.offline_trajectory_solver:main',
            'export_feasible_trajectories = python_controllers.export_feasible_trajectories:main',
            'joint_trajectory_replayer = python_controllers.joint_trajectory_replayer:main',
            'pick_place_open_loop = python_controllers.pick_place_open_loop:main',
        ],
    },
)
