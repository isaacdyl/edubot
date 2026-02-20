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
    install_requires=['setuptools', 'numpy'],
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
        ],
    },
)
