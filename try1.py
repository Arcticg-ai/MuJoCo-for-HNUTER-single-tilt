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
    
    # 更精确的旋转矩阵计算
    rotation_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return rotation_matrix

def calculate_allocation_matrix():
    """
    计算六维力/力矩到六个电机推力的分配矩阵
    
    Returns:
        6x6 分配矩阵
    """
    # 定义电机位置 [x, y, z]（机体坐标系）
    rotor_positions = [
        np.array([0.44207, 0.2778, 0.033856]),
        np.array([0.019547, 0.52175, 0.033856]),
        np.array([-0.46162, 0.24395, 0.033856]),
        np.array([-0.46162, -0.24395, 0.033856]),
        np.array([0.019547, -0.52175, 0.033856]),
        np.array([0.442074, -0.277802, 0.0338557])
    ]
    
    # 重心位置（从XML获取）
    com = np.array([0, -0.002137, 0.004638])
    
    # 计算相对于重心的位置向量
    rotor_positions_relative = [pos - com for pos in rotor_positions]
    
    # 电机推力方向（机体坐标系，单位向量）[7,8](@ref)
    thrust_directions = [
        np.array([-0.249999, 0.433009, 0.866028]),  # 旋翼0
        np.array([0.499998, 0, 0.866027]),          # 旋翼1
        np.array([-0.249999, -0.433009, 0.866028]), # 旋翼2
        np.array([-0.249999, 0.433009, 0.866028]),  # 旋翼3
        np.array([0.499998, 0, 0.866027]),          # 旋翼4
        np.array([-0.25, -0.433013, 0.866025])      # 旋翼5
    ]
    
    # 归一化推力方向（确保单位向量）
    thrust_directions = [v / np.linalg.norm(v) for v in thrust_directions]
    
    # 电机旋转方向（反扭矩符号）[6](@ref)
    rotation_signs = [1, -1, 1, -1, 1, -1]  # 0,2,4逆时针；1,3,5顺时针
    torque_coeff = 0.06  # 扭矩系数（来自XML gear属性）
    
    # 初始化分配矩阵（6行 x 6列）
    # 行：Fx, Fy, Fz, Mx, My, Mz
    # 列：每个电机的贡献
    B = np.zeros((6, 6))
    
    for i in range(6):
        # 获取当前电机的参数
        F = thrust_directions[i]          # 推力方向（单位向量）
        r = rotor_positions_relative[i]   # 相对于重心的位置
        
        # 计算力矩（r × F）[4](@ref)
        M = np.cross(r, F)
        
        # 添加反扭矩（只影响Mz分量）[1](@ref)
        M[2] += rotation_signs[i] * torque_coeff
        
        # 填入分配矩阵（第i列）
        B[:3, i] = F  # 力分量
        B[3:, i] = M  # 力矩分量
    
    return B

def main():
    """
    主函数 - 演示控制器使用
    """
    print("=== SkyVortex 控制器演示 ===")
    
    # 计算分配矩阵
    allocation_matrix = calculate_allocation_matrix()
    
    print("分配矩阵:")
    for i in range(6):
        print(f"电机{i}: {allocation_matrix[:, i]}")
    
    # 验证分配矩阵性质
    cond_num = np.linalg.cond(allocation_matrix)
    print(f"\n分配矩阵条件数: {cond_num:.2e}")
    
    if cond_num > 1000:
        print("警告：条件数过高，可能导致数值不稳定")
    
    # 验证悬停状态[3](@ref)
    mass = 7.1077  # 无人机质量 (kg)
    gravity = 9.81
    total_thrust = mass * gravity  # 总推力需求
    
    hover_test = np.array([0, 0, total_thrust, 0, 0, 0])  # 悬停状态力/力矩向量
    thrusts = np.linalg.pinv(allocation_matrix) @ hover_test
    
    print("\n悬停状态推力分布:")
    for i, thrust in enumerate(thrusts):
        print(f"电机{i}: {thrust:.2f} N")
    
    # 验证纯偏航状态
    yaw_test = np.array([0, 0, total_thrust, 0, 0, 5])  # 偏航力矩
    yaw_thrusts = np.linalg.pinv(allocation_matrix) @ yaw_test
    print("\n偏航状态推力分布:")

    rotation_signs = [1, -1, 1, -1, 1, -1]  # 0,2,4逆时针；1,3,5顺时针
    for i, thrust in enumerate(yaw_thrusts):
        print(f"电机{i}: {thrust:.2f} N (旋转方向: {'逆时针' if rotation_signs[i] > 0 else '顺时针'})")

if __name__ == "__main__":
    main()