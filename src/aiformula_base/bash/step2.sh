cd workspace/ros2_ws
source install/setup.bash
#ros2 launch launchers all_nodes.launch.py 
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap /cmd_vel:=/aiformula_control/twist_mux/cmd_vel
