cd workspace/ros2_ws
source install/setup.bash
./src/ai_formula_oit_2026/launchers/sample_launchers/shellscript/init_sensors.sh

sleep 5
ros2 launch oit_navigation oit_nodes_sim.launch.py 
