# Usage

First create the scene: 
```bash
python create_mj_scene.py
```

test if it is successfully created: 
```bash
python test_mj_scene.py
```

`robot_mj_interface.py` contains a class `RobotMjInterface`. It has 2 threads, one is for simulation step, another is for viewer updating. 

`robot_ctrl.py` contains a class `RobotCtrl`. It has a thread for maintaining low-level control, which use inverse dynamics and PD controller to control the joint. This class also has a ik (inverse kinematics) function to calculate the joint angles for a given end-effector pose.

Please refer to the code in `if __name__ == "__main__"` in `robot_ctrl.py` for how to use the `RobotMjInterface` and the `RobotCtrl` class.