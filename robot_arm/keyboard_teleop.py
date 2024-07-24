import mujoco
import pygame 
import time
import numpy as np

from robot_mj_interface import RobotMjInterface
from robot_ctrl import RobotCtrl

import cv2

# mujoco and controller setup
robot = RobotMjInterface()
robot_ctrl = RobotCtrl(robot)

# pygame setup
pygame.init()
screen = pygame.display.set_mode((640, 360))
clock = pygame.time.Clock()
running_pygame = True

dt = 0

while running_pygame and robot_ctrl.ok():
    
    for event in pygame.event.get():    # non-blocking
        if event.type == pygame.QUIT:
            running_pygame = False
    
    d_pos = np.zeros(3)
    d_rpy = np.zeros(3)
    d_grasp = 0.0
    
    move_scale = 0.4
    grasp_scale = 100
    
    keys = pygame.key.get_pressed() # non-blocking
    # move end effector
    if keys[pygame.K_w]:
        d_pos[0] += dt * move_scale
    if keys[pygame.K_s]:
        d_pos[0] -= dt * move_scale
    if keys[pygame.K_a]:
        d_pos[1] += dt * move_scale
    if keys[pygame.K_d]:
        d_pos[1] -= dt * move_scale
    if keys[pygame.K_UP]:
        d_pos[2] += dt * move_scale
    if keys[pygame.K_DOWN]:
        d_pos[2] -= dt * move_scale
    # move gripper
    if keys[pygame.K_j]:
        d_grasp += dt * grasp_scale
    if keys[pygame.K_k]:
        d_grasp -= dt * grasp_scale
    
    robot_ctrl.move_delta_ee(d_pos)
    current_gripper_pos = robot.get_gripper_pos()
    robot.set_gripper_pos(current_gripper_pos + d_grasp)
    
    # collect data
    image = robot.get_camera_image()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", image)
    cv2.waitKey(1)
    # d_pos
    # d_rpy
    
    dt = clock.tick(60) / 1000
    
cv2.destroyAllWindows()