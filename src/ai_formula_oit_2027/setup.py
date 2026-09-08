import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'ai_formula_oit_2027'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py') + glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rosuser',
    maintainer_email='rosuser@todo.todo',
    description='Vision-Only BEV Navigation and Pure Pursuit Control for AI Formula OIT 2027',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_lane_tracker = ai_formula_oit_2027.bev_lane_tracker_node:main',
            'pure_pursuit_controller = ai_formula_oit_2027.pure_pursuit_controller:main',
            'traffic_light_detector = ai_formula_oit_2027.traffic_light_detector_node:main',
        ],
    },
)


