#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkyVortex 六旋翼无人机控制器
基于MuJoCo物理仿真的Python控制器

作者: AI Assistant
日期: 2024
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
from typing import Tuple, List, Optional


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵
    Args:
        quat: 四元数 [w, x, y, z]  
    Returns:
        3x3 旋转矩阵
    """
    w, x, y, z = quat
    
    # 计算旋转矩阵的各个元素
    R11 = 1 - 2 * (y * y + z * z)
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (x * z + w * y)
    
    R21 = 2 * (x * y + w * z)
    R22 = 1 - 2 * (x * x + z * z)
    R23 = 2 * (y * z - w * x)
    
    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)
    R33 = 1 - 2 * (x * x + y * y)
    
    # 构造旋转矩阵
    rotation_matrix = np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33]
    ])
    
    return rotation_matrix

def main():
    """
    主函数 - 演示控制器使用
    """
    print("=== SkyVortex 控制器演示 ===")
    """
    计算六维力/力矩到六个电机推力的分配矩阵
    
    Returns:
        6x6 分配矩阵
    """
        #<site name="rotor_site_0" pos="0.44207 0.2778 0.033856" zaxis="-0.249999 0.433009 0.866028"/>
        #<site name="rotor_site_1" pos="0.019547 0.52175 0.033856" zaxis="0.499998 0 0.866027"/>
        #<site name="rotor_site_2" pos="-0.46162 0.24395 0.033856" zaxis="-0.249999 -0.433009 0.866028"/>
        #<site name="rotor_site_3" pos="-0.46162 -0.24395 0.033856" zaxis="-0.249999 0.433009 0.866028"/>
        #<site name="rotor_site_4" pos="0.019547 -0.52175 0.033856" zaxis="0.499998 0 0.866027"/>
        #<site name="rotor_site_5" pos="0.442074 -0.277802 0.0338557" zaxis="-0.25 -0.433013 0.866025"/>
    # 定义电机位置 [x, y, z]
    rotor_positions = [
        np.array([0.44207, 0.2778, 0.033856]),
        np.array([0.019547, 0.52175, 0.033856]),
        np.array([-0.46162, 0.24395, 0.033856]),
        np.array([-0.46162, -0.24395, 0.033856]),
        np.array([0.019547, -0.52175, 0.033856]),
        np.array([0.442074, -0.277802, 0.0338557])
    ]
    
    # 定义电机四元数姿态 [w, x, y, z]
    rotor_quaternions = [
        np.array([0.922577, -0.190485, -0.184513, -0.280218]),
        np.array([0.683013, 0.183012, 0.183012, 0.683013]),
        np.array([0.922493, 0.190542, -0.184455, 0.280495]),
        np.array([0.922577, -0.190485, -0.184513, -0.280218]),
        np.array([0.683013, 0.183012, 0.183012, 0.683013]),
        np.array([0.922493, 0.190542, -0.184455, 0.280495])
    ]
    
    

    # 定义电机推力方向在旋转矩阵中的z轴
    thrust_direction = np.array([0, 0, 1])
    torque_coeff = 0.06  # 扭矩系数
    rotation_signs = [1, -1, 1, -1, 1, -1]  # 电机旋转方向符号
    
    # 初始化分配矩阵
    allocation_matrix = np.zeros((6, 6))
    
    for i in range(6):
        # 获取当前电机的旋转矩阵
        R_rotor = _quat_to_rotation_matrix(rotor_quaternions[i])
        
        # 计算电机的推力方向在世界坐标系中的方向
        thrust_world = R_rotor @ thrust_direction
        thrust_world /= np.linalg.norm(thrust_world)  # 归一化
        
        # 计算力矩分配
        # 力矩 M = r x F，其中 r 是电机位置向量，F 是电机推力方向向量
        r = rotor_positions[i]
        M = np.cross(r, thrust_world)
        
        # 4. 添加反扭矩（绕Z轴）
        reaction_torque = torque_coeff * rotation_signs[i]
        M[2] += reaction_torque  # 添加到Mz

        # 构造分配矩阵的第 i 行
        # [Fx, Fy, Fz, Mx, My, Mz]
        allocation_matrix[i, :3] = thrust_world
        allocation_matrix[i, 3:] = M

    print(f"分配矩阵: {allocation_matrix}")


    # 验证悬停状态
    hover_test = np.array([0, 0, 6 * 9.8, 0, 0, 0])  # 总推力=6*重力
    thrusts = np.linalg.pinv(allocation_matrix) @ hover_test
    print("悬停状态推力分布:", thrusts)  # 应接近平均分布
    

if __name__ == "__main__":
    main()