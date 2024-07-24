import numpy as np
import time 
import threading
import copy
import cv2

import mujoco
import mujoco.viewer

from utils.transformations import *


class RobotMjInterface:
    """A class to interact with Mujoco simulation.

        It would maintain a thread to run the simulation.
        Current data can be accessed by the outside.
        Torque control is supported.
        
    """
    def __init__(self, sim_dt = 0.002, viewer_dt = 0.02) -> None:
        
        # load model and data
        xml_path = "ur5e_with_robotiq_2f85/ur5e_with_robotiq_2f85.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        self.data_lock = threading.Lock()
        
        # some parameters
        self.model.opt.timestep = sim_dt
        self.viewer_dt = viewer_dt
        self.joint_num = 6
        self.base_site_name = 'fixed_base_site'
        self.ee_site_name = 'ur5e/robotiq_2f85/end_effector'
        
        height = 480
        width = 640
        self.renderer = mujoco.Renderer(self.model, height, width)
        
        # start a new thread to run the simulation
        try:
            viewer_thread = threading.Thread(target=self.viewer_thread)
            sim_thread = threading.Thread(target=self.simulation_thread)
            
            viewer_thread.start()
            sim_thread.start()
        except Exception as e:
            print(e)
    
    def get_joint_pos(self):
        """
            get the joint position of the robot arm
        """
        return self.data.sensordata[:self.joint_num]
    
    def get_joint_vel(self):
        """
            get the joint velocity of the robot arm
        """
        return self.data.sensordata[self.joint_num:2*self.joint_num]
    
    def set_joint_torque(self, torque : np.ndarray) -> None:
        """set the joint torque of the robot arm

        Args:
            torque (np.ndarray): desired torque for each joint
        """
        assert len(torque) == self.joint_num
        with self.data_lock:
            self.data.ctrl[:self.joint_num] = torque
    
    def get_base_to_ee(self) -> np.ndarray:
        """get the homogeneous transformation from robot base to end effector
        
        (in fact we should use forward kinematics to calculate, but here we cheat by directly reading)

        Returns:
            np.ndarray: homogeneous transformation from robot base to end effector
        """
        with self.data_lock:
            ee_pos_in_world = self.data.site(self.ee_site_name).xpos
            base_pos_in_world = self.data.site(self.base_site_name).xpos
            
            ee_rotm = self.data.site(self.ee_site_name).xmat.reshape(3,3)
            base_rotm = self.data.site(self.base_site_name).xmat.reshape(3,3)
        
        ee_homo = numpy.identity(4)
        ee_homo[:3, :3] = ee_rotm
        ee_homo[:3, 3] = ee_pos_in_world
        
        base_homo = numpy.identity(4)
        base_homo[:3, :3] = base_rotm
        base_homo[:3, 3] = base_pos_in_world
        
        base_to_ee = np.linalg.solve(base_homo, ee_homo)
        
        return base_to_ee
    
    
    def get_gripper_pos(self):
        with self.data_lock:
            return self.data.ctrl[self.joint_num]
    
    def set_gripper_pos(self, desired_pos):
        with self.data_lock:
            self.data.ctrl[self.joint_num] = desired_pos
    
    def get_camera_image(self, camera_name = 'sideview'):
        with self.data_lock:
            self.renderer.update_scene(self.data, camera_name)
        frame = self.renderer.render()
        return frame
    
    def ok(self):
        return self.viewer.is_running()
    
    def simulation_thread(self):
        
        # load the pre-defined keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        while self.viewer.is_running():
            step_start = time.time()
            with self.data_lock:
                mujoco.mj_step(self.model, self.data)
            time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            # else:
            #     print(f"[Warning] simulation is too slow, {time_until_next_step}")
    
    def viewer_thread(self):
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        self.viewer.cam.distance = 1.5
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = -40
        self.viewer.cam.lookat = (0.4, 0, .3)
        
        while self.viewer.is_running():
            with self.data_lock:
                self.viewer.sync()
            time.sleep(self.viewer_dt)

                    
if __name__ == "__main__":
    robot = RobotMjInterface()
    
    robot.set_joint_torque(np.zeros(6))
    
    while robot.ok():
        # print(robot.get_base_to_ee())
        print(robot.get_joint_pos())
        time.sleep(1)