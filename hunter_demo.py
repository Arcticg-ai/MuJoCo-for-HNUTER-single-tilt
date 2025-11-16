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
        self.mass = 2.36  # 主机身质量 + 旋翼机构质量
        self.J = np.diag([0.08, 0.12, 0.1])  # 惯量矩阵
        
        # 旋翼布局参数
        self.l1 = 0.326  # 前旋翼组Y向距离(m)
        self.l2 = 0.5  # 尾部推进器X向距离(m)
        self.k_d = 8.1e-16  # 尾部反扭矩系数
        
        # 目标状态
        self.target_position = np.array([0.0, 0.0, 0.3])  # 初始目标高度
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        
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
        
        print("倾转旋翼控制器初始化完成（无控制算法）")
    
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
                'accel_x', 'accel_y', 'accel_z',
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
                state['acceleration'][0], state['acceleration'][1], state['acceleration'][2],
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
        """获取无人机当前状态（核心功能）"""
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
        self.target_position = np.array([x, y, z])
        print(f"目标位置设置为: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def set_target_attitude(self, roll: float, pitch: float, yaw: float):
        """设置目标姿态"""
        self.target_attitude = np.array([roll, pitch, yaw])
        print(f"目标姿态设置为: Roll={math.degrees(roll):.1f}°, Pitch={math.degrees(pitch):.1f}°, Yaw={math.degrees(yaw):.1f}°")
    
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
    
    def set_actuators(self, T12: float, T34: float, T5: float, alpha0: float, alpha1: float):
        """设置执行器参数（核心接口）"""
        # 更新状态
        self.T12 = T12
        self.T34 = T34
        self.T5 = T5
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
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
    
    def print_status(self):
        """打印当前状态信息"""
        state = self.get_state()
        pos = state['position']
        vel = state['velocity']
        accel = state['acceleration']
        euler_deg = np.degrees(state['euler'])
        
        print(f"位置: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")
        print(f"目标位置: X={self.target_position[0]:.2f}m, Y={self.target_position[1]:.2f}m, Z={self.target_position[2]:.2f}m")
        print(f"姿态: Roll={euler_deg[0]:.1f}°, Pitch={euler_deg[1]:.1f}°, Yaw={euler_deg[2]:.1f}°")
        print(f"速度: X={vel[0]:.2f}m/s, Y={vel[1]:.2f}m/s, Z={vel[2]:.2f}m/s")
        print(f"加速度: X={accel[0]:.2f}m/s², Y={accel[1]:.2f}m/s², Z={accel[2]:.2f}m/s²")
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
    
    # 加速度图
    accel_ax = axs[3]
    accel_ax.set_title('Acceleration')
    accel_ax.set_ylabel('m/s²')
    accel_ax.grid(True)
    accel_lines = {
        'x': accel_ax.plot([], [], 'r-', label='X')[0],
        'y': accel_ax.plot([], [], 'g-', label='Y')[0],
        'z': accel_ax.plot([], [], 'b-', label='Z')[0]
    }
    accel_ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 存储数据
    plot_data = {
        'time': [],
        'pos_x': [], 'pos_y': [], 'pos_z': [],
        'target_z': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'T12': [], 'T34': [], 'T5': [],
        'accel_x': [], 'accel_y': [], 'accel_z': []
    }
    
    return fig, axs, plot_data, {
        'pos': pos_lines,
        'att': att_lines,
        'thrust': thrust_lines,
        'accel': accel_lines
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
    plot_data['accel_x'].append(state['acceleration'][0])
    plot_data['accel_y'].append(state['acceleration'][1])
    plot_data['accel_z'].append(state['acceleration'][2])
    
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
    plot_lines['accel']['x'].set_data(plot_data['time'], plot_data['accel_x'])
    plot_lines['accel']['y'].set_data(plot_data['time'], plot_data['accel_y'])
    plot_lines['accel']['z'].set_data(plot_data['time'], plot_data['accel_z'])
    plot_lines['accel']['x'].axes.set_xlim(min(plot_data['time']), max(plot_data['time']))
    y_min = min(min(plot_data['accel_x']), min(plot_data['accel_y']), min(plot_data['accel_z']))
    y_max = max(max(plot_data['accel_x']), max(plot_data['accel_y']), max(plot_data['accel_z']))
    plot_lines['accel']['x'].axes.set_ylim(y_min - 2, y_max + 2)
    
    # 重绘图表
    fig.canvas.draw()
    fig.canvas.flush_events()

def main():
    """主函数 - 启动仿真"""
    print("=== 倾转旋翼无人机状态监控系统 ===")
    
    # 设置图表
    fig, axs, plot_data, plot_lines = setup_plots()

    # 初始化控制器
    controller = HnuterController("scene.xml")
    
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
        
        # 设置初始执行器参数
        tilt_angle = 0.0  # 初始倾角为0（垂直）
        thrust_value = 10.0  # 初始推力值
        
        try:
            while v.is_running:
                current_time = time.time() - start_time
                
                # 应用执行器参数（可在此处修改）
                controller.set_actuators(
                    T12=thrust_value,
                    T34=thrust_value,
                    T5=0.0,
                    alpha0=np.pi,
                    alpha1=np.pi
                )
                
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
                
                # 定期更新绘图
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