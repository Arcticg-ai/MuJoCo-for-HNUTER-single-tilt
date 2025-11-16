import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional
from datetime import datetime

class HnuterController:
    def __init__(self, model_path: str = "scene.xml"):
        # 加载MuJoCo模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # 物理参数
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        self.mass = 1.7 + 0.3  # 主机身质量 + 旋翼机构质量
        self.J = np.diag([0.08, 0.12, 0.1])  # 惯量矩阵
        
        # 旋翼布局参数
        self.l1 = 0.326  # 前旋翼组Y向距离(m)
        self.l2 = 0.5  # 尾部推进器X向距离(m)
        self.k_d = 8.1e-16  # 尾部反扭矩系数
        
        # 控制器增益
        self.Kp = np.diag([5, 5, 5])  # 位置增益
        self.Dp = np.diag([1.5, 1.5, 2.0])  # 速度阻尼
        self.KR = np.diag([1.2, 0.8, 0.5])   # 姿态增益
        self.Domega = np.diag([2.5, 1.5, 1.2])  # 角速度阻尼
        
        # 添加几何控制器参数
        self.geometric_params = {
            'position': {'kp': 10.0, 'kv': 15, 'ki': 0.2},
            'attitude': {'kr': 15.0 * np.array([1.5, 1.0, 1.0]),  # roll增益增加50%
                        'kw': 10.5 * np.array([1.8, 1.0, 1.0])}, # roll阻尼增加80%
            'altitude': {'kp': 5.0, 'ki': 0.2, 'kd': 1.0}
        }

        # 控制量
        self.f_c_body = np.zeros(3)  # 力
        self.f_c_world = np.zeros(3)  # 速度
        self.tau_c = np.zeros(3)  # 力矩
        self.u = np.zeros(5)  # 控制输入

        # 分配矩阵 (5x5)
        self.A = np.array([
            [1,  0, -1,  0,  0],   # X力分配 
            [0, -1,  0, -1, -1],   # Z力分配
            [0, self.l1,  0, -self.l1, 0],   # 滚转力矩
            [0,  0,  0,  0, -self.l2],  # 俯仰力矩
            [self.l1, 0,  self.l1, 0, -self.k_d]  # 偏航力矩
        ])
        
        # 增加分配矩阵的伪逆计算（应对奇异情况）
        self.A_pinv = np.linalg.pinv(self.A)

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 0.3])  # 初始目标高度1m
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.target_attitude_rate = np.array([0.0, 0.0, 0.0])
        
        # 倾转状态
        self.alpha0 = 0.0  # 前右倾角
        self.alpha1 = 0.0  # 前左倾角
        self.T12 = 0.0  # 前左旋翼组推力
        self.T34 = 0.0  # 前右旋翼组推力
        self.T5 = 0.0   # 尾部推进器推力
        
        # 执行器名称映射
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # 创建日志文件
        self._create_log_file()
        
        print("倾转旋翼控制器初始化完成")
    
    def _create_log_file(self):
        """创建日志文件并写入表头"""
        # 确保logs目录存在
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # 创建带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f'logs/drone_log_{timestamp}.csv'
        
        # 写入CSV表头
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'pos_x', 'pos_y', 'pos_z', 
                'target_x', 'target_y', 'target_z',
                'roll', 'pitch', 'yaw',
                'vel_x', 'vel_y', 'vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                'accel_x', 'accel_y', 'accel_z',  # 新增加速度字段[1](@ref)
                'f_world_x', 'f_world_y', 'f_world_z',
                'f_body_x', 'f_body_y', 'f_body_z',
                'tau_x', 'tau_y', 'tau_z',
                'u1', 'u2', 'u3', 'u4', 'u5',
                'T12', 'T34', 'T5',
                'alpha0', 'alpha1'
            ])
        
        print(f"已创建日志文件: {self.log_file}")
    
    def log_status(self, state: dict):
        """记录状态到日志文件"""
        timestamp = time.time()
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                state['position'][0], state['position'][1], state['position'][2],
                self.target_position[0], self.target_position[1], self.target_position[2],
                state['euler'][0], state['euler'][1], state['euler'][2],
                state['velocity'][0], state['velocity'][1], state['velocity'][2],
                state['angular_velocity'][0], state['angular_velocity'][1], state['angular_velocity'][2],
                state['acceleration'][0], state['acceleration'][1], state['acceleration'][2],  # 新增加速度值[1](@ref)
                self.f_c_world[0], self.f_c_world[1], self.f_c_world[2],
                self.f_c_body[0], self.f_c_body[1], self.f_c_body[2],
                self.tau_c[0], self.tau_c[1], self.tau_c[2],
                self.u[0], self.u[1], self.u[2], self.u[3], self.u[4],
                self.T12, self.T34, self.T5,
                self.alpha0, self.alpha1
            ])
    
    def _get_actuator_ids(self):
        """获取执行器ID"""
        self.actuator_ids = {}
        
        # 倾转执行器
        self.actuator_ids['tilt_right_joint'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_right')
        self.actuator_ids['tilt_left_joint'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_left')
        
        # 推力执行器
        self.thrust_actuators = ['thrust_rb', 'thrust_rt', 'thrust_lb', 'thrust_lt', 'thrust_tail']
        for name in self.thrust_actuators:
            self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    
    def _get_sensor_ids(self):
        """获取传感器ID"""
        self.sensor_ids = {}
        sensor_names = ['body_gyro', 'body_acc', 'body_quat', 'body_pos', 'body_vel']
        
        for name in sensor_names:
            self.sensor_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
    
    def get_state(self) -> dict:
        """获取无人机当前状态"""
        # 位置
        pos_sensor_id = self.sensor_ids['body_pos']
        position = self.data.sensordata[self.model.sensor_adr[pos_sensor_id]:self.model.sensor_adr[pos_sensor_id]+3]
        
        # 四元数姿态
        quat_sensor_id = self.sensor_ids['body_quat']
        quaternion = self.data.sensordata[self.model.sensor_adr[quat_sensor_id]:self.model.sensor_adr[quat_sensor_id]+4]
        
        # 转换为旋转矩阵
        rotation_matrix = self._quat_to_rotation_matrix(quaternion)
        
        # 速度
        vel_sensor_id = self.sensor_ids['body_vel']
        velocity = self.data.sensordata[self.model.sensor_adr[vel_sensor_id]:self.model.sensor_adr[vel_sensor_id]+3]
        
        # 角速度
        gyro_sensor_id = self.sensor_ids['body_gyro']
        angular_velocity = self.data.sensordata[self.model.sensor_adr[gyro_sensor_id]:self.model.sensor_adr[gyro_sensor_id]+3]
        
        # 欧拉角
        euler = self._quat_to_euler(quaternion) # roll, pitch, yaw
        
        # 加速度
        accel_sensor_id = self.sensor_ids['body_acc']
        acceleration = self.data.sensordata[self.model.sensor_adr[accel_sensor_id]:self.model.sensor_adr[accel_sensor_id]+3]

        return {
            'position': position.copy(),
            'quaternion': quaternion.copy(),
            'rotation_matrix': rotation_matrix,
            'velocity': velocity.copy(),
            'angular_velocity': angular_velocity.copy(),
            'acceleration': acceleration.copy(),  # 包含加速度数据
            'euler': euler.copy()
        }
    
    def set_target_position(self, x: float, y: float, z: float):
        """设置目标位置"""
        self.target_position = np.array([x, y, z])
        print(f"目标位置设置为: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def set_target_attitude(self, roll: float, pitch: float, yaw: float):
        """设置目标姿态"""
        self.target_attitude = np.array([roll, pitch, yaw])
        print(f"目标姿态设置为: Roll={math.degrees(roll):.1f}°, Pitch={math.degrees(pitch):.1f}°, Yaw={math.degrees(yaw):.1f}°")
    
    def takeoff(self, target_height: float):
        """起飞到指定高度"""
        self.set_target_position(0.0, 0.0, 1.0)
        self.set_target_attitude(0.0, 0.0, 0.0)
        print(f"开始起飞到高度 {target_height:.1f}m")
    
    def land(self):
        """降落"""
        current_pos = self.get_state()['position']
        self.set_target_position(current_pos[0], current_pos[1], 0.35)
        print("开始降落")
    
    def hover(self):
        """悬停在当前位置"""
        current_pos = self.get_state()['position']
        self.set_target_position(current_pos[0], current_pos[1], current_pos[2])
        print(f"悬停在位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
    
    def move_to(self, x: float, y: float, z: float):
        """移动到指定位置"""
        self.set_target_position(x, y, z)
    
    def skew(self, v):
        """向量到反对称矩阵"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def vee(self, S):
        """反对称矩阵到向量"""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])
    
    def euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        欧拉角转旋转矩阵
        (roll, pitch, yaw) -> 3x3旋转矩阵
        """
        roll, pitch, yaw = euler
        
        # 计算每个轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        
        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        
        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # 组合旋转矩阵 (Z-Y-X顺序)
        return R_z.dot(R_y.dot(R_x))

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转旋转矩阵
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

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转欧拉角 (roll, pitch, yaw)
        """
        w, x, y, z = quat
        
        # Roll (x轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y轴旋转)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z轴旋转)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

    def geometric_attitude_control(self, state: dict) -> np.ndarray:
        """
        基于几何控制理论的姿态控制器（修复版）
        """
        # 获取当前状态
        R = state['rotation_matrix']  # 当前旋转矩阵
        omega = state['angular_velocity']  # 当前角速度
        target_rot = self.euler_to_rotation_matrix(self.target_attitude)
        
        # 修复1：使用四元数计算姿态误差（更稳定）
        current_quat = state['quaternion']
        target_quat = self._euler_to_quaternion(self.target_attitude)
        q_error = self._quat_mult(self._quat_conjugate(target_quat), current_quat)
        
        # 转换为旋转向量（轴角表示）
        angle = 2 * math.acos(max(min(q_error[0], 1), -1))  # 限制范围防止数值错误
        if angle < 1e-6:
            axis = np.array([0, 0, 1])
        else:
            axis = q_error[1:4] / math.sin(angle/2)
        
        e_R_vec = axis * angle  # 姿态误差向量
        
        # 修复2：使用目标角速度（通常为零）
        e_omega = omega - self.target_attitude_rate
        
        # 修复3：调整增益（增加roll通道增益）
        kr = self.geometric_params['attitude']['kr'] * np.array([1.5, 1.0, 1.0])  # roll增益增强
        kw = self.geometric_params['attitude']['kw'] * np.array([1.8, 1.0, 1.0])  # roll阻尼增强
        
        # 修复4：使用实际惯量矩阵（非对角）
        inertia = self.J  # 使用类中定义的惯量矩阵
        
        # 力矩计算公式（包含科里奥利项补偿）
        torque = -kr * e_R_vec - kw * e_omega + np.cross(omega, inertia.dot(omega))
        
        return torque

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数"""
        roll, pitch, yaw = euler
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """四元数共轭"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_mult(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算控制力和力矩
        """
        p = state['position']
        v = state['velocity']
        R = state['rotation_matrix']
        
        # 位置误差（世界坐标系）
        e_p = self.target_position - p
        e_v = self.target_velocity - v
        f_c_world = self.mass *(self.Kp @ e_p + self.Dp @ e_v + np.array([0, 0, self.gravity]))
        self.f_c_world = f_c_world
        f_c_body = R.T @ f_c_world
        tau_c = self.geometric_attitude_control(state)
        self.f_c_body = f_c_body
        self.tau_c = tau_c
        return f_c_body, tau_c
    
    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray):
        """分配执行器命令"""
        # 构造控制向量b
        b = np.array([
            f_c_body[0],    # X力
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
        ])
        # 求解分配矩阵方程
        # 使用伪逆求解（更稳定）
        try:
            u = np.linalg.solve(self.A, b)
        except np.linalg.LinAlgError:
            u = self.A_pinv @ b  # 奇异时使用伪逆
        self.u = u

        # 计算推力
        T12 = np.sqrt(u[0]**2 + u[1]**2)  # 前左组总推力
        T34 = np.sqrt(u[2]**2 + u[3]**2)  # 前右组总推力
        T5 = u[4]                        # 尾部推进器推力
        
        # 计算倾角
        alpha1 = np.arctan2(u[0], -u[1])  # 前左倾角
        alpha0 = np.arctan2(-u[2], -u[3])  # 前右倾角
        
        # 推力限制
        T_max = 30
        T12 = np.clip(T12, 0, T_max)
        T34 = np.clip(T34, 0, T_max)
        T5 = np.clip(T5, -30, 30)
        
        # 更新状态
        self.T12 = T12
        self.T34 = T34
        self.T5 = T5
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
        return T12, T34, T5, alpha0, alpha1
    
    def apply_controls(self, T12: float, T34: float, T5: float, alpha0: float, alpha1: float):
        """应用控制命令到执行器"""
        # 设置倾转角度
        tilt_right_id = self.actuator_ids['tilt_right_joint']
        tilt_left_id = self.actuator_ids['tilt_left_joint']

        self.data.ctrl[tilt_right_id] = alpha0  # 右侧倾角
        self.data.ctrl[tilt_left_id] = alpha1  # 左侧倾角
        
        # 设置推力
        # 右侧两个螺旋桨（每个推力为总推力的一半）
        thrust_rt_id = self.actuator_ids['thrust_rt']
        thrust_rb_id = self.actuator_ids['thrust_rb']
        self.data.ctrl[thrust_rt_id] = T34 / 2
        self.data.ctrl[thrust_rb_id] = T34 / 2
        # 左侧两个螺旋桨
        thrust_lt_id = self.actuator_ids['thrust_lt']
        thrust_lb_id = self.actuator_ids['thrust_lb']
        self.data.ctrl[thrust_lt_id] = T12 / 2
        self.data.ctrl[thrust_lb_id] = T12 / 2
        
        # 尾部推进器
        thrust_tail_id = self.actuator_ids['thrust_tail']
        self.data.ctrl[thrust_tail_id] = T5  
    
    def update_control(self):
        """更新控制命令"""
        # 获取当前状态
        state = self.get_state()
        
        # 记录状态到日志
        self.log_status(state)
        
        # 计算控制力和力矩
        f_c_body, tau_c = self.compute_control_wrench(state)
        
        # 分配执行器命令
        T12, T34, T5, alpha0, alpha1 = self.allocate_actuators(f_c_body, tau_c)
        # 应用控制
        self.apply_controls(T12, T34, T5, alpha0, alpha1)
    
    def print_status(self):
        """打印当前状态信息（不再打印全部信息，只显示关键状态）"""
        state = self.get_state()
        pos = state['position']
        print(f"位置: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}) | 目标: ({self.target_position[0]:6.2f}, {self.target_position[1]:6.2f}, {self.target_position[2]:6.2f})")

def setup_plots():
    """设置实时绘图"""
    plt.ion()  # 开启交互模式
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    
    # 设置图间距
    plt.subplots_adjust(hspace=0.5)
    
    # 位置图
    pos_ax = axs[0]
    pos_ax.set_title('Position (XYZ)')
    pos_ax.set_ylabel('Meters')
    pos_ax.grid(True)
    pos_lines = {
        'x': pos_ax.plot([], [], 'r-', label='X')[0],
        'y': pos_ax.plot([], [], 'g-', label='Y')[0],
        'z': pos_ax.plot([], [], 'b-', label='Z')[0],
        'target': pos_ax.plot([], [], 'k--', label='Target Z')[0]
    }
    pos_ax.legend(loc='upper right')
    
    # 姿态图
    att_ax = axs[1]
    att_ax.set_title('Attitude (Roll, Pitch, Yaw)')
    att_ax.set_ylabel('Degrees')
    att_ax.grid(True)
    att_lines = {
        'roll': att_ax.plot([], [], 'r-', label='Roll')[0],
        'pitch': att_ax.plot([], [], 'g-', label='Pitch')[0],
        'yaw': att_ax.plot([], [], 'b-', label='Yaw')[0]
    }
    att_ax.legend(loc='upper right')
    
    # 推力图
    thrust_ax = axs[2]
    thrust_ax.set_title('Thrust')
    thrust_ax.set_ylabel('Newtons')
    thrust_ax.grid(True)
    thrust_lines = {
        'T12': thrust_ax.plot([], [], 'r-', label='T12')[0],
        'T34': thrust_ax.plot([], [], 'g-', label='T34')[0],
        'T5': thrust_ax.plot([], [], 'b-', label='T5')[0]
    }
    thrust_ax.legend(loc='upper right')
    
    # 控制力图
    force_ax = axs[3]
    force_ax.set_title('Control Forces (Body Frame)')
    force_ax.set_ylabel('Newtons')
    force_ax.grid(True)
    force_lines = {
        'x': force_ax.plot([], [], 'r-', label='X')[0],
        'y': force_ax.plot([], [], 'g-', label='Y')[0],
        'z': force_ax.plot([], [], 'b-', label='Z')[0]
    }
    force_ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 存储数据
    plot_data = {
        'time': [],
        'pos_x': [], 'pos_y': [], 'pos_z': [],
        'target_z': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'T12': [], 'T34': [], 'T5': [],
        'f_x': [], 'f_y': [], 'f_z': []
    }
    
    return fig, axs, plot_data, {
        'pos': pos_lines,
        'att': att_lines,
        'thrust': thrust_lines,
        'force': force_lines
    }

def update_plot(fig, plot_data, plot_lines, time_val, state, controller):
    """更新实时图表"""
    # 更新数据
    plot_data['time'].append(time_val)
    
    # 位置数据
    plot_data['pos_x'].append(state['position'][0])
    plot_data['pos_y'].append(state['position'][1])
    plot_data['pos_z'].append(state['position'][2])
    plot_data['target_z'].append(controller.target_position[2])
    
    # 姿态数据（转换为度数）
    plot_data['roll'].append(np.degrees(state['euler'][0]))
    plot_data['pitch'].append(np.degrees(state['euler'][1]))
    plot_data['yaw'].append(np.degrees(state['euler'][2]))
    
    # 推力数据
    plot_data['T12'].append(controller.T12)
    plot_data['T34'].append(controller.T34)
    plot_data['T5'].append(controller.T5)
    
    # 控制力数据
    plot_data['f_x'].append(controller.f_c_body[0])
    plot_data['f_y'].append(controller.f_c_body[1])
    plot_data['f_z'].append(controller.f_c_body[2])
    
    # 只保留最近100个数据点
    max_points = 100
    for key in plot_data:
        plot_data[key] = plot_data[key][-max_points:]
    
    # 更新位置图
    plot_lines['pos']['x'].set_data(plot_data['time'], plot_data['pos_x'])
    plot_lines['pos']['y'].set_data(plot_data['time'], plot_data['pos_y'])
    plot_lines['pos']['z'].set_data(plot_data['time'], plot_data['pos_z'])
    plot_lines['pos']['target'].set_data(plot_data['time'], plot_data['target_z'])
    plot_lines['pos']['target'].axes.set_xlim(min(plot_data['time']), max(plot_data['time']))
    y_min = min(min(plot_data['pos_x']), min(plot_data['pos_y']), min(plot_data['pos_z']))
    y_max = max(max(plot_data['pos_x']), max(plot_data['pos_y']), max(plot_data['pos_z']))
    plot_lines['pos']['x'].axes.set_ylim(y_min - 0.1, y_max + 0.1)
    
    # 更新姿态图
    plot_lines['att']['roll'].set_data(plot_data['time'], plot_data['roll'])
    plot_lines['att']['pitch'].set_data(plot_data['time'], plot_data['pitch'])
    plot_lines['att']['yaw'].set_data(plot_data['time'], plot_data['yaw'])
    plot_lines['att']['roll'].axes.set_xlim(min(plot_data['time']), max(plot_data['time']))
    y_min = min(min(plot_data['roll']), min(plot_data['pitch']), min(plot_data['yaw']))
    y_max = max(max(plot_data['roll']), max(plot_data['pitch']), max(plot_data['yaw']))
    plot_lines['att']['roll'].axes.set_ylim(y_min - 5, y_max + 5)
    
    # 更新推力图
    plot_lines['thrust']['T12'].set_data(plot_data['time'], plot_data['T12'])
    plot_lines['thrust']['T34'].set_data(plot_data['time'], plot_data['T34'])
    plot_lines['thrust']['T5'].set_data(plot_data['time'], plot_data['T5'])
    plot_lines['thrust']['T12'].axes.set_xlim(min(plot_data['time']), max(plot_data['time']))
    y_min = min(min(plot_data['T12']), min(plot_data['T34']), min(plot_data['T5']))
    y_max = max(max(plot_data['T12']), max(plot_data['T34']), max(plot_data['T5']))
    plot_lines['thrust']['T12'].axes.set_ylim(y_min - 2, y_max + 2)
    
    # 更新控制力图
    plot_lines['force']['x'].set_data(plot_data['time'], plot_data['f_x'])
    plot_lines['force']['y'].set_data(plot_data['time'], plot_data['f_y'])
    plot_lines['force']['z'].set_data(plot_data['time'], plot_data['f_z'])
    plot_lines['force']['x'].axes.set_xlim(min(plot_data['time']), max(plot_data['time']))
    y_min = min(min(plot_data['f_x']), min(plot_data['f_y']), min(plot_data['f_z']))
    y_max = max(max(plot_data['f_x']), max(plot_data['f_y']), max(plot_data['f_z']))
    plot_lines['force']['x'].axes.set_ylim(y_min - 2, y_max + 2)
    
    # 重绘图表
    fig.canvas.draw()
    fig.canvas.flush_events()

def compute_arrow_rotation_matrix(vec):
    """从方向向量生成箭头的旋转矩阵（3x3）"""
    z = vec / (np.linalg.norm(vec) + 1e-8)
    x = np.cross(np.array([0, 0, 1]), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([10.0, 0.0, 0.0])
    else:
        x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])

def user_render_callback(scene):
    """修正后的渲染回调"""
    global controller
    model = controller.model
    data = controller.data
    
    # 获取控制力和位置
    force = controller.f_c_world.copy()
    scale = 0.05
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hnuter")
    pos = data.xpos[body_id].copy()
    
    # 计算箭头端点
    end = pos + force * scale
    vec = end - pos
    
    # 添加箭头到场景
    if scene.ngeom < scene.maxgeom:
        arrow = scene.geoms[scene.ngeom]
        arrow.type = mj.mjtGeom.mjGEOM_ARROW
        arrow.objtype = mj.mjtObj.mjOBJ_UNKNOWN
        arrow.category = mj.mjtCatBit.mjCAT_DECOR
        arrow.pos[:] = pos
        arrow.size[:] = [0.01, 0.02, np.linalg.norm(vec)]  # 箭头尺寸
        arrow.mat[:] = compute_arrow_rotation_matrix(vec).flatten()
        arrow.rgba[:] = [1, 0, 0, 1]  # 红色
        scene.ngeom += 1
    print(f"最终位置:6****************************************************************6")

def main():
    """主函数 - 启动仿真并可视化受力箭头"""
    global controller
    print("=== 倾转旋翼无人机控制器演示（含箭头可视化） ===")
    
    # 设置图表（可选）
    fig, axs, plot_data, plot_lines = setup_plots()

    # 初始化控制器
    controller = HnuterController("scene.xml")
    count = 0
    # 启动 Viewer 并注册回调
    with viewer.launch_passive(controller.model, controller.data) as v:
        v.scene_callback = user_render_callback  # 注册箭头渲染回调
        
        print("\n仿真启动：")
        print("按 Ctrl+C 终止仿真")
        print(f"日志文件路径: {controller.log_file}")
        
        start_time = time.time()
        last_plot_update = 0
        plot_update_interval = 0.1
        try:
            while v.is_running:
                current_time = time.time() - start_time
                
                # 飞行任务：起飞到2.0m
                controller.takeoff(2.0)
                
                # 控制更新
                controller.update_control()
                count = count + 1
                if count % 1 == 0:
                # 仿真步进
                    mj.mj_step(controller.model, controller.data)
                # mj.mj_step(controller.model, controller.data)
                # v.render()
                v.sync()
                
                # 状态打印
                if int(current_time * 10) % 5 == 0:
                    controller.print_status()

                # 可选：绘图更新
                state = controller.get_state()
                if current_time - last_plot_update > plot_update_interval:
                    update_plot(fig, plot_data, plot_lines, current_time, state, controller)
                    last_plot_update = current_time

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n仿真中断")
        
        print("仿真结束")
        plt.savefig(controller.log_file.replace('.csv', '.png'))
        plt.close()
        print(f"图表保存至: {controller.log_file.replace('.csv', '.png')}")

        final_state = controller.get_state()
        print(f"最终位置: ({final_state['position'][0]:.2f}, {final_state['position'][1]:.2f}, {final_state['position'][2]:.2f})")

if __name__ == "__main__":
    main()