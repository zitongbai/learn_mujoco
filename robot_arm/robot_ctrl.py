import numpy as np
from copy import deepcopy
import time
import threading

import mujoco
from mujoco import minimize

from robot_mj_interface import RobotMjInterface

from utils.transformations import *


class RobotCtrl:
    def __init__(self, interface:RobotMjInterface) -> None:
        self.interface = interface
        
        # we only use model and data for calculation
        # we get current states from outside
        # so here is deepcopy
        with interface.data_lock:
            self.model = interface.model
            self.data = deepcopy(interface.data)
        
        self.joint_num = interface.joint_num
        self.ee_site_name = interface.ee_site_name
        
        # parameters for joint pd control
        self.kp = np.array([50]*self.joint_num, dtype=np.float64)
        self.kd = np.array([10]*self.joint_num, dtype=np.float64)
        
        # create a thread to run the low level control loop
        self.control_freq = 100
        self.thread_low_level_ctrl = threading.Thread(target=self.low_level_ctrl_loop)
        self.thread_low_level_ctrl.start()
    
    def ok(self):
        return self.thread_low_level_ctrl.is_alive()
    
    def inverse_dynamics_control(self, 
                                 desired_joint_pos:np.ndarray, 
                                 desired_joint_vel:np.ndarray = None) -> np.ndarray:
        """inverse dynamics control
        
            Given the desired joint position, calculate the desired joint torque
            Desired joint torque = M(q) * (kp * (q_des - q) - kd * (q_dot_des - q_dot)) + C(q, q_dot)
            
            https://github.com/google-deepmind/mujoco/discussions/424 

        Args:
            desired_joint_pos (np.ndarray): desired joint position
            desired_joint_vel (np.ndarray): desired joint velocity

        Returns:
            np.ndarray: desired joint torque (control input to the robot in mujoco)
        """
        
        assert len(desired_joint_pos) == self.joint_num
        if desired_joint_vel is None: 
            desired_joint_vel = np.zeros(self.joint_num)
        else: 
            assert len(desired_joint_vel) == self.joint_num
        

        self.data.qpos[:self.joint_num] = self.interface.get_joint_pos()
        self.data.qvel[:self.joint_num] = self.interface.get_joint_vel()
        
        mujoco.mj_forward(self.model, self.data)
        
        # inertia matrix
        M = np.zeros([self.model.nv, self.model.nv])
        mujoco.mj_fullM(self.model, M, self.data.qM)
        # coriolis and gravity
        C = self.data.qfrc_bias[:self.joint_num]
        
        pd_ctrl = self.kp * (desired_joint_pos - self.interface.get_joint_pos()) + self.kd * (desired_joint_vel - self.interface.get_joint_vel())
        desired_joint_torque = M[:self.joint_num, :self.joint_num] @ pd_ctrl + C
        
        return desired_joint_torque
    
    def set_joint_pos_vel(self, desired_joint_pos:np.ndarray, desired_joint_vel:np.ndarray=None) -> None:
        self.desired_joint_pos = desired_joint_pos
        self.desired_joint_vel = desired_joint_vel
    
    def low_level_ctrl_loop(self):
        
        self.desired_joint_pos = self.interface.get_joint_pos()
        self.desired_joint_vel = None
        
        while self.interface.ok():
            start_time = time.time()
            try:
                desired_joint_torque = self.inverse_dynamics_control(self.desired_joint_pos, self.desired_joint_vel)
                self.interface.set_joint_torque(desired_joint_torque)
            except Exception as e:
                print(f"Error in control loop: {e}")
                break
            
            # sleep to control the loop frequency
            time_until_next_step = 1/self.control_freq - (time.time() - start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            else: 
                print("Control loop is too slow")

    def ik(self, target_ee_pos, target_ee_quat):
        current_joint_pos = self.interface.get_joint_pos()
        self.target_ee_pos = target_ee_pos
        self.target_ee_quat = target_ee_quat
        ik_joint_pos, _ = minimize.least_squares(
            current_joint_pos, 
            self.ik_res, 
            None, 
            jacobian = self.ik_jac,
            verbose = 0
        )
        return ik_joint_pos
    
    def ik_res(self, 
               joint_pos, 
               radius = 0.04, 
               reg = 1e-3, 
               reg_target = None):
        # for vectorized calculation
        res = []    # residual 
        for i in range(joint_pos.shape[1]):
            # compute forward kinematics
            self.data.qpos[:self.joint_num] = joint_pos[:, i]
            mujoco.mj_kinematics(self.model, self.data)
            
            # position error (residual)
            res_pos = self.data.site(self.ee_site_name).xpos - self.target_ee_pos
            
            # current ee quaternion
            ee_quat = np.empty(4)
            mujoco.mju_mat2Quat(ee_quat, self.data.site(self.ee_site_name).xmat)
            # orientation residual
            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, self.target_ee_quat, ee_quat)
            res_quat *= radius
            
            # regularization
            reg_target = np.zeros(self.joint_num) if reg_target is None else reg_target
            res_reg = reg * (joint_pos[:, i] - reg_target)
            
            # concatenate them together
            res_i = np.hstack([res_pos, res_quat, res_reg])
            res.append(np.atleast_2d(res_i).T)
            
        return np.hstack(res)
    
    def ik_jac(self, 
               joint_pos, 
               res,
               radius = 0.04,
               reg = 1e-3):
        del res
        
        # prepare for jacobian
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # get end effector site Jacobian
        jac_pos = np.empty((3, self.model.nv))
        jac_quat = np.empty((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_quat, self.data.site(self.ee_site_name).id)
        jac_pos = jac_pos[:, :self.joint_num]
        jac_quat = jac_quat[:, :self.joint_num]
        
        # calculate the rotation part of jacobian in residual
        ee_quat = np.empty(4)
        mujoco.mju_mat2Quat(ee_quat, self.data.site(self.ee_site_name).xmat)
        Deffector = np.empty((3,3))
        mujoco.mjd_subQuat(self.target_ee_quat, ee_quat, None, Deffector)
        
        # Rotate into target frame, multiply by subQuat Jacobian, scale by radius.
        target_mat = np.empty(9)
        mujoco.mju_quat2Mat(target_mat, self.target_ee_quat)
        target_mat = target_mat.reshape(3,3)
        mat = radius * Deffector.T @ target_mat.T
        jac_quat = mat @ jac_quat
        
        # regularization jac
        jac_reg = reg * np.eye(self.joint_num)
        
        return np.vstack([jac_pos, jac_quat, jac_reg])
    
    def move_delta_ee(self, d_pos, d_rpy=[0, 0, 0], axes='sxyz') -> None:
        """Given the delta position and rotation of end effector, move the robot

        Args:
            d_pos (_type_): delta position in the end effector frame
            d_rpy (_type_): delta rotation of the end effector
        """
        
        assert len(d_pos) == 3
        assert len(d_rpy) == 3
        d_quat = quaternion_from_euler(d_rpy[0], d_rpy[1], d_rpy[2], axes=axes)
        
        # homogeneous transformation from base to end effector
        homo_base_to_ee = self.interface.get_base_to_ee()
        # current end effector pos in base frame
        current_ee_pos = homo_base_to_ee[:3, 3]
        # current rotation matrix from base to end effector
        current_ee_rotm = np.identity(4)
        current_ee_rotm[:3,:3] = homo_base_to_ee[:3,:3]
        current_ee_quat = quaternion_from_matrix(current_ee_rotm) # xyzw
        
        # next end effector position (in base frame)
        # next_ee_pos = current_ee_pos + current_ee_rotm[:3,:3] @ np.array(d_pos)   # d_pos in ee frame
        next_ee_pos = current_ee_pos + np.array(d_pos)  # d_pos in base frame
        # next end effector quaternion (from base to ee)
        next_ee_quat = quaternion_multiply(current_ee_quat, d_quat)
        # xyzw -> wxyz
        next_ee_quat = np.array([next_ee_quat[3], next_ee_quat[0], next_ee_quat[1], next_ee_quat[2]])
        
        ik_target_joint_pos = self.ik(next_ee_pos, next_ee_quat)
        self.set_joint_pos_vel(ik_target_joint_pos)


if __name__ == "__main__":
    
    import cv2
    
    robot = RobotMjInterface()
    robot_ctrl = RobotCtrl(robot)
    
    while robot_ctrl.ok():
        t = time.time()
        d_pos = [-0.03*np.sin(t), 0.03*np.cos(t), 0]
        robot_ctrl.move_delta_ee(d_pos)
        
        # print(robot.get_base_to_ee())
        
        image = robot.get_camera_image('ur5e/robotiq_2f85/ee_view')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("image", image)
        cv2.waitKey(1)
        
        time.sleep(0.1)