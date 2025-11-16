import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
import csv
import os
import matplotlib.pyplot as plt
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
        self.mass = 1.7 + 0.3 + 0.36 + 0.02 # 主机身质量 + 旋翼机构质量
        self.J = np.diag([0.08, 0.12, 0.1])  # 惯量矩阵
        
        # 旋翼布局参数
        self.l1 = 0.3  # 前旋翼组Y向距离(m)
        self.l2 = 0.5  # 尾部推进器X向距离(m)
        self.k_d = 8.1e-8  # 尾部反扭矩系数
        
        # 控制器增益 (根据论文设置)
        self.Kp = np.diag([1, 3, 3])  # 位置增益
        self.Dp = np.diag([1.5, 1.5, 2.0])  # 速度阻尼
        self.KR = np.array([5.0, 1.0, 1.0])   # 姿态增益
        self.Domega = np.array([1.5, 2.5, 2.0])  # 角速度阻尼

        # 控制量
        self.f_c_body = np.zeros(3)  # 机体坐标系下的控制力
        self.f_c_world = np.zeros(3)  # 世界坐标系下的控制力
        self.tau_c = np.zeros(3)     # 控制力矩
        self.u = np.zeros(5)         # 控制输入向量

        #=====================时来运转系===========================
        self.A = np.array([
            [1, 0, -1, 0, 0],   # X力分配 
            [0, 1,  0, 1, 1],   # Z力分配
            [0, -self.l1,  0, self.l1, 0],   # 滚转力矩
            [0,  0,  0,  0, -self.l2],  # 俯仰力矩
            [self.l1, 0,  self.l1, 0, self.k_d]  # 偏航力矩
        ])
        
        # 分配矩阵的伪逆 (用于奇异情况)
        self.A_pinv = np.linalg.pinv(self.A)

        # 目标状态
        self.target_position = np.array([0.0, 0.0, 0.3])  # 初始目标高度
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.target_attitude_rate = np.array([0.0, 0.0, 0.0])
        
        # 倾转状态
        self.alpha0 = 0.0  # 前右倾角
        self.alpha1 = 0.0  # 前左倾角
        self.T12 = 0.0  # 前左旋翼组推力
        self.T34 = 0.0  # 前右旋翼组推力
        self.T5 = 0.0   # 尾部推进器推力
        
        # 添加角度连续性处理参数
        self.last_alpha0 = 0
        self.last_alpha1 = 0

        # 执行器名称映射
        self._get_actuator_ids()
        self._get_sensor_ids()
        
        # 创建日志文件
        self._create_log_file()
              
        print("倾转旋翼控制器初始化完成（含几何控制器）")
    
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
                'target_roll', 'target_pitch', 'target_yaw'
                'vel_x', 'vel_y', 'vel_z',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                'accel_x', 'accel_y', 'accel_z',
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
                self.target_attitude[0],self.target_attitude[1]/2, self.target_attitude[2],
                state['velocity'][0], state['velocity'][1], state['velocity'][2],
                state['angular_velocity'][0], state['angular_velocity'][1], state['angular_velocity'][2],
                state['acceleration'][0], state['acceleration'][1], state['acceleration'][2],
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
        
        # 加速度
        accel_sensor_id = self.sensor_ids['body_acc']
        acceleration = self.data.sensordata[self.model.sensor_adr[accel_sensor_id]:self.model.sensor_adr[accel_sensor_id]+3]
        
        # 欧拉角
        euler = self._quat_to_euler(quaternion) # roll, pitch, yaw
        
        return {
            'position': position.copy(),
            'quaternion': quaternion.copy(),
            'rotation_matrix': rotation_matrix,
            'velocity': velocity.copy(),
            'angular_velocity': angular_velocity.copy(),
            'acceleration': acceleration.copy(),
            'euler': euler.copy()
        }
    
    def set_target_position(self, x: float, y: float, z: float):
        """设置目标位置"""
        self.target_position = np.array([-x, -y, z])
    
    def set_target_velocity(self, x: float, y: float, z: float):
        """设置目标位置"""
        self.target_velocity = np.array([x, y, z])

    def set_target_attitude(self, roll: float, pitch: float, yaw: float):
        """设置目标姿态"""
        self.target_attitude = np.array([roll, pitch, yaw])
    
    def takeoff(self, target_height: float):
        """起飞到指定高度"""
        self.set_target_position(0.0, 0.0, target_height)
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
        """欧拉角转旋转矩阵(roll, pitch, yaw) -> 3x3旋转矩阵"""
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
        """四元数转旋转矩阵"""
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
        """四元数转欧拉角 (roll, pitch, yaw)"""
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
    
    def vee_map(self, S: np.ndarray) -> np.ndarray:
        """反对称矩阵的vee映射"""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    def compute_target_rotation(self, f_c_world: np.ndarray) -> np.ndarray:
        """计算目标旋转矩阵（改进版：优先保证俯仰角跟踪目标值）"""
        # 计算期望的z轴方向（推力方向）
        f_norm = np.linalg.norm(f_c_world)
        if f_norm < 1e-6:
            z_d = np.array([0, 0, 1])
        else:
            z_d = f_c_world / f_norm
        
        # 直接使用目标俯仰角构造参考x轴（而非仅依赖偏航角）
        pitch = self.target_attitude[1]  # 目标俯仰角
        yaw = self.target_attitude[2]    # 目标偏航角
        
        # 构造参考x轴：先绕Y轴旋转俯仰角，再绕Z轴旋转偏航角
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        x_ref = R_z @ R_y @ np.array([1, 0, 0])  # 世界坐标系下的参考x轴
        
        # 计算正交坐标系（确保y轴与z轴、x_ref垂直）
        y_d = np.cross(z_d, x_ref)
        y_norm = np.linalg.norm(y_d)
        if y_norm < 1e-6:
            # 退化情况处理：使用默认y轴（避免除零）
            y_d = np.array([0, 1, 0])
        else:
            y_d /= y_norm

        # 构造目标旋转矩阵（无需SVD，直接返回正交矩阵）
        return np.column_stack([x_ref, y_d, z_d])

    def compute_attitude_errors(self, state: dict, R_WB_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算姿态误差(论文公式5)"""
        R_WB = state['rotation_matrix']
        omega = state['angular_velocity']
        
        # 计算姿态误差矩阵
        R_error = R_WB_d.T @ R_WB - R_WB.T @ R_WB_d
        
        # vee映射（反对称矩阵→向量）
        e_R = np.array([R_error[2, 1], R_error[0, 2], R_error[1, 0]])
        
        # 角速度误差
        omega_ref = self.target_attitude_rate
        e_omega = omega - R_WB @ omega_ref
        
        return e_R, e_omega

    def compute_control_torque(self, e_R: np.ndarray, e_omega: np.ndarray, state: dict, R_WB_d: np.ndarray) -> np.ndarray:
        """计算控制力矩（论文公式6）"""
        omega = state['angular_velocity']
        J = self.J
        
        # 科里奥利项
        coriolis = np.cross(omega, J @ omega)
        
        # 角加速度前馈项（简化版）
        # 实际实现可能需要数值微分
        d_omega_ref = np.zeros(3)  # 假设角加速度为0
        
        # 前馈补偿项
        R_WB = state['rotation_matrix']
        term1 = np.cross(omega, R_WB.T @ R_WB_d @ self.target_attitude_rate)
        term2 = R_WB.T @ R_WB_d @ d_omega_ref
        feedforward = -J @ (term1 - term2)
        
        # 总控制力矩
        tau = -self.KR * e_R - self.Domega * e_omega + coriolis + feedforward
        
        return tau

    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """计算控制力和力矩(论文公式3)"""
        p = state['position']
        v = state['velocity']
        R = state['rotation_matrix']
        
        # 位置误差（世界坐标系）
        e_p = self.target_position - p
        # print(e_p)
        e_v = self.target_velocity - v
        # 位置控制律（论文公式3）
        f_c_world = self.mass * (self.Kp @ e_p + self.Dp @ e_v + np.array([0, 0, self.gravity]))
        f_c_body = R.T @ f_c_world
        # 姿态控制律
        # 计算目标旋转矩阵（公式4）
        R_WB_d = self.compute_target_rotation(f_c_world)
        # 计算姿态误差（公式5）
        e_R, e_omega = self.compute_attitude_errors(state, R_WB_d)
        # 计算控制力矩（公式6）
        tau_c = self.compute_control_torque(e_R, e_omega, state, R_WB_d)
        # 存储控制量用于日志记录
        self.f_c_world = f_c_world
        self.f_c_body = f_c_body
        self.tau_c = tau_c
        
        return f_c_body, tau_c

    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        """
        分配执行器命令（论文公式7-9）
        """
        # 构造控制向量b（论文公式7）
        b = np.array([
            f_c_body[0],    # X力
            f_c_body[2],    # Z力
            tau_c[0],       # 滚转力矩
            tau_c[1],       # 俯仰力矩
            tau_c[2]        # 偏航力矩
        ])
        
        # 求解分配矩阵方程（论文公式8）
        try:
            u = np.linalg.solve(self.A, b)
        except np.linalg.LinAlgError:
            u = self.A_pinv @ b  # 奇异时使用伪逆
        # 计算推力（论文公式9）
        T12 = np.sqrt(u[0]**2 + u[1]**2)  # 前左组推力
        T34 = np.sqrt(u[2]**2 + u[3]**2)  # 前右组推力
        T5 = u[4]                        # 尾部推进器推力

        # 计算倾角（使用平滑函数）
        alpha1 = np.arctan2(u[0], u[1])  # 前左倾角
        alpha0 = np.arctan2(u[2], u[3])  # 前右倾角
        # 更新角度历史
        self.last_alpha0 = alpha0
        self.last_alpha1 = alpha1
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
        self.u = u  # 存储控制输入向量
        
        return T12, T34, T5, alpha0, alpha1
    
    def set_actuators(self, T12: float, T34: float, T5: float, alpha0: float, alpha1: float):
        """应用控制命令到执行器"""
        # 设置倾转角度
        tilt_right_id = self.actuator_ids['tilt_right_joint']
        tilt_left_id = self.actuator_ids['tilt_left_joint']
        
        self.alpha0 = alpha0
        self.alpha1 = alpha1

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
        """更新控制命令（整合几何控制器）"""
        # 获取当前状态
        state = self.get_state()

        # 记录状态到日志
        self.log_status(state)
        
        # 计算控制力和力矩
        f_c_body, tau_c = self.compute_control_wrench(state)
        
        # 分配执行器命令
        T12, T34, T5, alpha0, alpha1 = self.allocate_actuators(f_c_body, tau_c, state)
        
        # 应用控制
        self.set_actuators(T12, T34, T5, alpha0, alpha1)
    
    def print_status(self):
        """打印当前状态信息（包含控制信息）"""
        state = self.get_state()
        pos = state['position']
        vel = state['velocity']
        accel = state['acceleration']
        euler_deg = np.degrees(state['euler'])
        target_euler_deg = np.degrees(self.target_attitude)
        print(f"位置: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")
        print(f"目标位置: X={self.target_position[0]:.2f}m, Y={self.target_position[1]:.2f}m, Z={self.target_position[2]:.2f}m")
        print(f"姿态: Roll={euler_deg[0]:.1f}°, Pitch={euler_deg[1]:.1f}°, Yaw={euler_deg[2]:.1f}°")  
        print(f"姿态: tar_Roll={target_euler_deg[0]:.1f}°, tar_Pitch={target_euler_deg[1]/2:.1f}°, tar_Yaw={target_euler_deg[2]:.1f}°") 
        print(f"速度: X={vel[0]:.2f}m/s, Y={vel[1]:.2f}m/s, Z={vel[2]:.2f}m/s")
        print(f"加速度: X={accel[0]:.2f}m/s², Y={accel[1]:.2f}m/s², Z={accel[2]:.2f}m/s²")
        print(f"控制力: X={self.f_c_body[0]:.2f}N, Y={self.f_c_body[1]:.2f}N, Z={self.f_c_body[2]:.2f}N")
        print(f"控制力矩: X={self.tau_c[0]:.2f}Nm, Y={self.tau_c[1]:.2f}Nm, Z={self.tau_c[2]:.2f}Nm")
        print(f"执行器状态: T12={self.T12:.2f}N, T34={self.T34:.2f}N, T5={self.T5:.2f}N, α0={math.degrees(self.alpha0):.1f}°, α1={math.degrees(self.alpha1):.1f}°")
        print("--------------------------------------------------")

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
    
        # === 倾转角度图（替换原加速度图）===
    tilt_ax = axs[3]
    tilt_ax.set_title('Tilt Angles')
    tilt_ax.set_ylabel('Degrees')
    tilt_ax.grid(True)
    tilt_lines = {
        'alpha0': tilt_ax.plot([], [], 'r-', label='Right Tilt (α0)')[0],
        'alpha1': tilt_ax.plot([], [], 'g-', label='Left Tilt (α1)')[0]
    }
    tilt_ax.legend(loc='upper right')
    tilt_ax.axhline(y=85, color='r', linestyle='--', alpha=0.5)
    tilt_ax.axhline(y=-85, color='r', linestyle='--', alpha=0.5)
    tilt_ax.text(0.05, 0.9, 'SAFE LIMIT ±85°', transform=tilt_ax.transAxes, color='r')
    # ================================

    plt.tight_layout()
    
    # 存储数据
    plot_data = {
        'time': [],
        'pos_x': [], 'pos_y': [], 'pos_z': [],
        'target_z': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'T12': [], 'T34': [], 'T5': [],
        # 'accel_x': [], 'accel_y': [], 'accel_z': []
        'alpha0': [], 'alpha1': []  # 添加倾转角存储
    }
    
    return fig, axs, plot_data, {
        'pos': pos_lines,
        'att': att_lines,
        'thrust': thrust_lines,
        # 'accel': accel_lines
        'tilt': tilt_lines  # 更新为倾转角度图
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
    
    # 加速度数据
    # plot_data['accel_x'].append(state['acceleration'][0])
    # plot_data['accel_y'].append(state['acceleration'][1])
    # plot_data['accel_z'].append(state['acceleration'][2])

    # 倾转角度数据
    plot_data['alpha0'].append(np.degrees(controller.alpha0))
    plot_data['alpha1'].append(np.degrees(controller.alpha1))
    
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
    
    # 更新加速度图
    # plot_lines['accel']['x'].set_data(plot_data['time'], plot_data['accel_x'])
    # plot_lines['accel']['y'].set_data(plot_data['time'], plot_data['accel_y'])
    # plot_lines['accel']['z'].set_data(plot_data['time'], plot_data['accel_z'])
    # plot_lines['accel']['x'].axes.set_xlim(min(plot_data['time']), max(plot_data['time']))
    # y_min = min(min(plot_data['accel_x']), min(plot_data['accel_y']), min(plot_data['accel_z']))
    # y_max = max(max(plot_data['accel_x']), max(plot_data['accel_y']), max(plot_data['accel_z']))
    # plot_lines['accel']['x'].axes.set_ylim(y_min - 2, y_max + 2)
    
    # =====更新倾转角度图 =====
    plot_lines['tilt']['alpha0'].set_data(plot_data['time'], plot_data['alpha0'])
    plot_lines['tilt']['alpha1'].set_data(plot_data['time'], plot_data['alpha1'])
    plot_lines['tilt']['alpha0'].axes.set_xlim(min(plot_data['time']), max(plot_data['time']))
    y_min = min(min(plot_data['alpha0']), min(plot_data['alpha1']))
    y_max = max(max(plot_data['alpha0']), max(plot_data['alpha1']))
    plot_lines['tilt']['alpha0'].axes.set_ylim(min(y_min-5, -90), max(y_max+5, 90))

    # 重绘图表
    fig.canvas.draw()
    fig.canvas.flush_events()

def mobius_trajectory(t, period=20.0, radius=2.0, height=2.5):
    """生成莫比乌斯环轨迹参数[7,8](@ref)"""
    u = 2 * np.pi * t / period
    
    # 参数方程
    x = radius * (1 + 0.5 * np.cos(u/2)) * np.cos(u)
    y = radius * (1 + 0.5 * np.cos(u/2)) * np.sin(u)
    z = 0.5 * radius * np.sin(u/2) + height
    
    # 导数（速度）
    dx_du = radius * (-(1+0.5*np.cos(u/2))*np.sin(u) - 0.25*np.sin(u/2)*np.cos(u))
    dy_du = radius * ((1+0.5*np.cos(u/2))*np.cos(u) - 0.25*np.sin(u/2)*np.sin(u))
    dz_du = 0.25 * radius * np.cos(u/2)
    
    # 速度 = 导数 * (2π/period)
    speed_factor = 2 * np.pi / period
    vx = dx_du * speed_factor
    vy = dy_du * speed_factor
    vz = dz_du * speed_factor
    
    # 目标偏航角（速度方向）
    yaw = np.arctan2(vy, vx) if np.sqrt(vx**2 + vy**2) > 0.1 else 0
    pitch = np.arctan2(vz, np.sqrt(vx**2+vy**2))

    return np.array([x, y, z]), np.array([vx, vy, vz]), yaw, pitch

def main():
    """主函数 - 启动仿真"""
    print("=== 倾转旋翼无人机状态监控系统 ===")
    
    # 设置图表
    fig, axs, plot_data, plot_lines = setup_plots()

    # 初始化控制器
    controller = HnuterController("mission_scene.xml")
    
    pitch = 0
    yaw = 0
    tag = 0
    pos_x = 0
    pos_y = 0
    pos_z = 2.5

    # 启动 Viewer
    with viewer.launch_passive(controller.model, controller.data) as v:
        print("\n仿真启动：")
        print("按 Ctrl+C 终止仿真")
        print(f"日志文件路径: {controller.log_file}")
        
        start_time = time.time()
        last_print_time = 0
        last_plot_update = 0

        print_interval = 0.5  # 打印间隔 (秒)
        plot_update_interval = 0.1  # 绘图更新间隔
        
        try:
            while v.is_running:
                current_time = time.time() - start_time
                

                target_pos, target_vel, target_yaw , target_pitch = mobius_trajectory(current_time)
            
                # 设置目标状态
                # controller.set_target_position(*target_pos)
                # controller.set_target_velocity(*target_vel)
                # controller.set_target_attitude(0, target_pitch, target_yaw)
                
                target_position = [-3.0, 3.0, 2.25]
                target_attitude = [-1.0472, 0.0, 0.0]  # roll, pitch, yaw
                controller.set_target_position(0.0, 0.0, 1.25)
                controller.set_target_attitude(0.0, 1.4, 0)
                # 获取当前状态
                state = controller.get_state()
                
                # 执行控制算法
                controller.update_control()
                
                # 仿真步进
                mj.mj_step(controller.model, controller.data)
                
                # 同步可视化
                v.sync()
                
                # 获取当前状态
                state = controller.get_state()
                
                # 记录状态
                controller.log_status(state)
                
                # 定期打印状态
                if current_time - last_print_time > print_interval:
                    controller.print_status()
                    last_print_time = current_time
                
                # # 定期更新绘图
                # if current_time - last_plot_update > plot_update_interval:
                #     update_plot(fig, plot_data, plot_lines, current_time, state, controller)
                #     last_plot_update = current_time

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