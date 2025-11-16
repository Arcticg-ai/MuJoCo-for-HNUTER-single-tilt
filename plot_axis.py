import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QuadcopterDynamics:
    def __init__(self):
        # 无人机参数
        self.mass = 1.0  # kg
        self.g = 9.81  # m/s²
        
        # 无人机尺寸 (基于图片描述)
        self.arm_length = 0.3  # 机臂长度 (m)
        self.body_length = 0.2  # 机身长度 (m)
        
        # 螺旋桨位置 (机体坐标系 {B})
        # 前右(1), 前左(2), 后左(3), 后右(4)
        self.prop_positions = np.array([
            [self.arm_length, self.arm_length, 0],    # 前右
            [self.arm_length, -self.arm_length, 0],   # 前左
            [-self.arm_length, -self.arm_length, 0],  # 后左
            [-self.arm_length, self.arm_length, 0]    # 后右
        ])
        
        # 推力系数和扭矩系数
        self.k_thrust = 8e-6  # 推力系数 N/(rad/s)²
        self.k_torque = 1e-7  # 扭矩系数 N·m/(rad/s)²
        
        # 螺旋桨旋转方向 (基于图片描述: 橙色顺时针, 蓝色逆时针)
        self.prop_directions = np.array([1, -1, 1, -1])  # 1: 顺时针, -1: 逆时针
        
    def rotation_matrix(self, roll, pitch, yaw):
        """计算从机体坐标系{B}到惯性坐标系{I}的旋转矩阵"""
        # 绕x轴旋转 (滚转)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # 绕y轴旋转 (俯仰)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # 绕z轴旋转 (偏航)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵: R = Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def calculate_propeller_forces(self, omega_sq):
        """计算每个螺旋桨产生的力和力矩"""
        # 每个螺旋桨的推力 (沿机体z轴负方向)
        thrusts = self.k_thrust * omega_sq
        
        # 每个螺旋桨的反扭矩 (沿机体z轴)
        torques = self.k_torque * omega_sq * self.prop_directions
        
        return thrusts, torques
    
    def calculate_total_force_moment(self, omega_sq, roll=0, pitch=0, yaw=0):
        """计算无人机整体受到的力和力矩"""
        # 计算每个螺旋桨的推力和反扭矩
        thrusts, torques = self.calculate_propeller_forces(omega_sq)
        
        # 总推力 (在机体坐标系中)
        total_thrust_body = np.array([0, 0, -np.sum(thrusts)])
        
        # 总反扭矩 (在机体坐标系中)
        total_reaction_torque = np.array([0, 0, np.sum(torques)])
        
        # 计算每个螺旋桨推力产生的力矩
        moment_from_thrust = np.zeros(3)
        for i, (thrust, pos) in enumerate(zip(thrusts, self.prop_positions)):
            # 力向量 (沿机体z轴负方向)
            force_vec = np.array([0, 0, -thrust])
            
            # 力矩 = 位置向量 × 力向量
            moment_from_thrust += np.cross(pos, force_vec)
        
        # 总力矩 = 推力产生的力矩 + 反扭矩
        total_moment_body = moment_from_thrust + total_reaction_torque
        
        # 转换到惯性坐标系
        R_BI = self.rotation_matrix(roll, pitch, yaw)
        total_force_inertial = R_BI @ total_thrust_body + np.array([0, 0, self.mass * self.g])
        
        return total_force_inertial, total_moment_body, thrusts, torques
    
    def allocation_matrix(self):
        """计算分配矩阵"""
        # 分配矩阵将螺旋桨转速平方映射到力和力矩
        # [总推力, 滚转力矩, 俯仰力矩, 偏航力矩]^T = M * [ω1², ω2², ω3², ω4²]^T
        
        k_f = self.k_thrust
        k_m = self.k_torque
        L = self.arm_length
        
        # 分配矩阵
        M = np.array([
            [k_f, k_f, k_f, k_f],           # 总推力
            [L*k_f/np.sqrt(2), -L*k_f/np.sqrt(2), -L*k_f/np.sqrt(2), L*k_f/np.sqrt(2)],  # 滚转力矩
            [L*k_f/np.sqrt(2), L*k_f/np.sqrt(2), -L*k_f/np.sqrt(2), -L*k_f/np.sqrt(2)],  # 俯仰力矩
            [k_m, -k_m, k_m, -k_m]          # 偏航力矩
        ])
        
        return M
    
    def plot_forces(self, omega_sq, roll=0, pitch=0, yaw=0):
        """绘制力的分解示意图"""
        total_force, total_moment, thrusts, torques = self.calculate_total_force_moment(omega_sq, roll, pitch, yaw)
        R_BI = self.rotation_matrix(roll, pitch, yaw)
        
        fig = plt.figure(figsize=(15, 5))
        
        # 1. 机体坐标系中的力
        ax1 = fig.add_subplot(131, projection='3d')
        self._plot_body_forces(ax1, thrusts, torques)
        
        # 2. 惯性坐标系中的力
        ax2 = fig.add_subplot(132, projection='3d')
        self._plot_inertial_forces(ax2, total_force, R_BI)
        
        # 3. 力矩示意图
        ax3 = fig.add_subplot(133, projection='3d')
        self._plot_moments(ax3, total_moment)
        
        plt.tight_layout()
        plt.show()
        
        return total_force, total_moment
    
    def _plot_body_forces(self, ax, thrusts, torques):
        """绘制机体坐标系中的力"""
        # 绘制螺旋桨位置和力
        colors = ['orange', 'cyan', 'orange', 'cyan']  # 根据图片中的颜色
        
        for i, (pos, thrust, torque, color) in enumerate(zip(self.prop_positions, thrusts, torques, colors)):
            # 推力向量 (向下)
            ax.quiver(pos[0], pos[1], 0, 0, 0, -thrust/100, 
                     color=color, linewidth=2, arrow_length_ratio=0.1, label=f'推力{i+1}')
            
            # 扭矩向量
            ax.quiver(pos[0], pos[1], 0, 0, 0, torque/1000, 
                     color='red', linewidth=2, arrow_length_ratio=0.1, linestyle='--')
        
        # 坐标系
        ax.quiver(0, 0, 0, 0.2, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0.2, 0, color='g', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 0.2, color='b', linewidth=2, arrow_length_ratio=0.1)
        
        ax.text(0.2, 0, 0, 'X', color='r', fontsize=12)
        ax.text(0, 0.2, 0, 'Y', color='g', fontsize=12)
        ax.text(0, 0, 0.2, 'Z', color='b', fontsize=12)
        ax.text(0, 0, 0, '机体坐标系 {B}', color='k', fontsize=10)
        
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('机体坐标系中的力分布')
        
    def _plot_inertial_forces(self, ax, total_force, R_BI):
        """绘制惯性坐标系中的力"""
        # 重力
        gravity = np.array([0, 0, -self.mass * self.g])
        ax.quiver(0, 0, 0, 0, 0, gravity[2]/100, 
                 color='purple', linewidth=3, arrow_length_ratio=0.1, label='重力')
        
        # 总推力在惯性系中的分量
        thrust_inertial = total_force - gravity
        ax.quiver(0, 0, 0, thrust_inertial[0]/100, thrust_inertial[1]/100, thrust_inertial[2]/100,
                 color='blue', linewidth=3, arrow_length_ratio=0.1, label='总推力')
        
        # 总力
        ax.quiver(0, 0, 0, total_force[0]/100, total_force[1]/100, total_force[2]/100,
                 color='red', linewidth=3, arrow_length_ratio=0.1, label='总力')
        
        # 坐标系
        ax.quiver(0, 0, 0, 0.2, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0.2, 0, color='g', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 0.2, color='b', linewidth=2, arrow_length_ratio=0.1)
        
        ax.text(0.2, 0, 0, 'X', color='r', fontsize=12)
        ax.text(0, 0.2, 0, 'Y', color='g', fontsize=12)
        ax.text(0, 0, 0.2, 'Z', color='b', fontsize=12)
        ax.text(0, 0, 0, '惯性坐标系 {I}', color='k', fontsize=10)
        
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('惯性坐标系中的力分解')
        
    def _plot_moments(self, ax, total_moment):
        """绘制力矩示意图"""
        # 力矩向量
        ax.quiver(0, 0, 0, total_moment[0]/1000, total_moment[1]/1000, total_moment[2]/1000,
                 color='green', linewidth=3, arrow_length_ratio=0.1, label='总力矩')
        
        # 力矩分量
        ax.quiver(0, 0, 0, total_moment[0]/1000, 0, 0,
                 color='r', linewidth=2, arrow_length_ratio=0.1, linestyle='--', label='滚转力矩')
        ax.quiver(0, 0, 0, 0, total_moment[1]/1000, 0,
                 color='g', linewidth=2, arrow_length_ratio=0.1, linestyle='--', label='俯仰力矩')
        ax.quiver(0, 0, 0, 0, 0, total_moment[2]/1000,
                 color='b', linewidth=2, arrow_length_ratio=0.1, linestyle='--', label='偏航力矩')
        
        # 坐标系
        ax.quiver(0, 0, 0, 0.2, 0, 0, color='r', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0.2, 0, color='g', linewidth=2, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 0.2, color='b', linewidth=2, arrow_length_ratio=0.1)
        
        ax.text(0.2, 0, 0, 'X', color='r', fontsize=12)
        ax.text(0, 0.2, 0, 'Y', color='g', fontsize=12)
        ax.text(0, 0, 0.2, 'Z', color='b', fontsize=12)
        ax.text(0, 0, 0, '机体坐标系 {B}', color='k', fontsize=10)
        
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('力矩分解')
        ax.legend()

# 使用示例
if __name__ == "__main__":
    # 创建无人机对象
    drone = QuadcopterDynamics()
    
    # 计算分配矩阵
    M = drone.allocation_matrix()
    print("分配矩阵 M:")
    print(M)
    print("\n分配矩阵的逆 (用于控制):")
    print(np.linalg.pinv(M))
    
    # 示例1: 悬停状态 (所有螺旋桨转速相同)
    omega_sq_hover = np.array([600**2, 600**2, 600**2, 600**2])  # rad/s 平方
    
    print("\n=== 悬停状态 ===")
    force, moment, thrusts, torques = drone.calculate_total_force_moment(omega_sq_hover)
    print(f"总力 (惯性系): {force} N")
    print(f"总力矩 (机体系): {moment} N·m")
    print(f"各螺旋桨推力: {thrusts} N")
    print(f"各螺旋桨扭矩: {torques} N·m")
    
    # 示例2: 前飞状态 (后螺旋桨转速增加)
    omega_sq_forward = np.array([550**2, 550**2, 650**2, 650**2])
    
    print("\n=== 前飞状态 ===")
    force, moment, thrusts, torques = drone.calculate_total_force_moment(omega_sq_forward, pitch=0.2)
    print(f"总力 (惯性系): {force} N")
    print(f"总力矩 (机体系): {moment} N·m")
    
    # 绘制力的分解图
    print("\n绘制力的分解图...")
    drone.plot_forces(omega_sq_hover)
    
    # 验证分配矩阵
    print("\n=== 验证分配矩阵 ===")
    # 使用分配矩阵计算总力和力矩
    forces_moments = M @ omega_sq_hover
    print(f"通过分配矩阵计算的结果: {forces_moments}")
    print(f"推力: {forces_moments[0]:.2f} N")
    print(f"滚转力矩: {forces_moments[1]:.4f} N·m")
    print(f"俯仰力矩: {forces_moments[2]:.4f} N·m")
    print(f"偏航力矩: {forces_moments[3]:.4f} N·m")
    