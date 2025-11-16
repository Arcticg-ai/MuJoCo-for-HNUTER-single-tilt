import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation

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
        self.k_d = 8.1e-6  # 尾部反扭矩系数
        
        # 控制器增益
        self.Kp = np.diag([30, 30, 10])  # 位置增益
        self.Dp = np.diag([20, 20, 25])  # 速度阻尼
        self.KR = np.diag([5, 5, 5])   # 姿态增益
        self.Domega = np.diag([13, 15, 12])  # 角速度阻尼
        
        # 添加几何控制器参数
        self.geometric_params = {
            'position': {'kp': 10.0, 'kv': 15, 'ki': 0.2},
            'attitude': {'kr': 15.0, 'kw': 10.5},
            'altitude': {'kp': 5.0, 'ki': 0.2, 'kd': 1.0}
        }

        # 控制量
        self.f_c_body = np.zeros(3)  # 力矩
        self.tau_c = np.zeros(3)  # 力矩

        # 分配矩阵 (5x5)
        self.A = np.array([
            [1,  0, -1,  0,  0],   # X力分配 
            [0, -1,  0, -1, -1],   # Z力分配
            [0, self.l1,  0, -self.l1, 0],   # 滚转力矩
            [0,  0,  0,  0, -self.l2],  # 俯仰力矩
            [self.l1, 0,  self.l1, 0, -self.k_d]  # 偏航力矩
        ])
        
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
        
        print("倾转旋翼控制器初始化完成")
    
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
        # 欧拉角转四元数
        quaternion = self.data.sensordata[self.model.sensor_adr[quat_sensor_id]:self.model.sensor_adr[quat_sensor_id]+4]
        
        # 转换为旋转矩阵
        rot = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        rotation_matrix = rot.as_matrix()
        
        # 速度
        vel_sensor_id = self.sensor_ids['body_vel']
        velocity = self.data.sensordata[self.model.sensor_adr[vel_sensor_id]:self.model.sensor_adr[vel_sensor_id]+3]
        
        # 角速度
        gyro_sensor_id = self.sensor_ids['body_gyro']
        angular_velocity = self.data.sensordata[self.model.sensor_adr[gyro_sensor_id]:self.model.sensor_adr[gyro_sensor_id]+3]
        
        # 欧拉角
        euler = rot.as_euler('xyz')  # roll, pitch, yaw
        
        return {
            'position': position.copy(),
            'quaternion': quaternion.copy(),
            'rotation_matrix': rotation_matrix,
            'velocity': velocity.copy(),
            'angular_velocity': angular_velocity.copy(),
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
    
    def takeoff(self, target_height: float = 2.0):
        """起飞到指定高度"""
        current_pos = self.get_state()['position']
        # self.set_target_position(current_pos[0], current_pos[1], target_height)
        self.set_target_position(0.0, 0.0, target_height)
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
        
        Args:
            euler: [roll, pitch, yaw] 弧度
            
        Returns:
            3x3 旋转矩阵
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

    def geometric_attitude_control(self, state: dict) -> np.ndarray:
        """
        基于几何控制理论的姿态控制器
        
        Args:
            state: 当前状态字典
            dt: 时间步长
            
        Returns:
            3维力矩向量 [Mx, My, Mz]
        """
        # 获取当前状态
        R = state['rotation_matrix']  # 当前旋转矩阵
        omega = state['angular_velocity']  # 当前角速度
        # print(f"角速度: ({omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f})")
        # 目标姿态转换为旋转矩阵
        target_rot = self.euler_to_rotation_matrix(self.target_attitude)
        
        # 计算姿态误差 e_R = 0.5*(R_desᵀR - RᵀR_des) (对数映射)
        R_error = 0.5 * (target_rot.T.dot(R) - R.T.dot(target_rot))
        # 提取误差向量的反对称部分 (vee映射)
        e_R_vec = np.array([R_error[2, 1], R_error[0, 2], R_error[1, 0]])
        # print(f"姿态误差: ({e_R_vec[0]:.2f}, {e_R_vec[1]:.2f}, {e_R_vec[2]:.2f})")
        # 计算角速度误差 e_ω = ω - RᵀR_desω_des
        # (假设目标角速度ω_des=0，即稳定飞行)
        e_omega = omega
        
        # 几何控制律计算力矩
        kr = self.geometric_params['attitude']['kr']
        kw = self.geometric_params['attitude']['kw']
        inertia = np.diag([0.08, 0.12, 0.1])  # 无人机惯量
        
        # 力矩计算公式: τ = -kᵣeᵣ - kᵥeᵥ + ω×(Jω)
        torque = -kr * e_R_vec - kw * e_omega + np.cross(omega, inertia.dot(omega))
        # print(f"力矩: ({torque[0]:.2f}, {torque[1]:.2f}, {torque[2]:.2f})")
        return torque

    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算控制力和力矩
        返回: (f_c_body, tau_c)
        """
        p = state['position']
        v = state['velocity']
        R = state['rotation_matrix']
        omega = state['angular_velocity']
        
        # 位置误差（世界坐标系）
        e_p = self.target_position - p
        e_v = self.target_velocity - v
        # print(f"位置误差: {e_p} 速度误差: {e_v}")

        # # 科里奥利项
        # coriolis_term = self.mass * np.cross(omega, R.T @ v)
        
        # # 位置控制律
        # f_c_body = (
        #     self.Kp @ e_p_body +
        #     self.Dp @ e_v_body -
        #     self.mass * R.T @ np.array([0, 0, self.gravity]) +
        #     coriolis_term
        # )
        # 位置控制律
        f_c_world = ( self.mass *(self.Kp @ e_p + self.Dp @ e_v + np.array([0, 0, self.gravity]))
        )

        f_c_body = R.T @ f_c_world

        # 期望姿态矩阵
        rot_ref = Rotation.from_euler('xyz', self.target_attitude)
        R_ref = rot_ref.as_matrix()
        
        # 期望z轴（去除x分量）
        f_c_bar = np.array([0, f_c_world[1], f_c_world[2]])  # 去除x分量
        W_fc = R @ f_c_bar
        R_d = R_ref
        if np.linalg.norm(W_fc) < 1e-5:
            # R_d = R_ref
            R_d = np.eye(3)
        else:
            B_z_d = W_fc / np.linalg.norm(W_fc)
            W_x_ref = R_ref[:, 0]  # 参考x轴
            B_y_d = np.cross(B_z_d, W_x_ref)
            
            if np.linalg.norm(B_y_d) < 1e-5:
                R_d = R_ref
            else:
                B_y_d = B_y_d / np.linalg.norm(B_y_d)
                B_z_d_actual = np.cross(W_x_ref, B_y_d)
                B_z_d_actual = B_z_d_actual / np.linalg.norm(B_z_d_actual)
                R_d = np.column_stack((W_x_ref, B_y_d, B_z_d_actual))
        
        # 计算姿态误差
        e_R_mat = 0.5 * (R_d.T @ R - R.T @ R_d)
        e_R = self.vee(e_R_mat)
        
        # 角速度误差
        e_omega = omega - R.T @ R_d @ self.target_attitude_rate
        

        # 姿态控制律
        # tau_c = (
        #     self.J @ (-self.KR @ e_R - self.Domega @ e_omega) +
        #     np.cross(omega, self.J @ omega)
        # )
        tau_c = (
            -self.KR @ e_R -  self.Domega @ e_omega + np.cross(omega, self.J @ omega) -
            self.J @ ( np.cross(omega, R.T @ R_d @ self.target_attitude_rate) # - R.T @ R_d @ self.target_attitude_accel
            )
        )
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
        # print(f"控制向量b: {b}")
        # 求解分配矩阵方程
        u = np.linalg.solve(self.A, b)
        
        # 计算推力
        T12 = np.sqrt(u[0]**2 + u[1]**2)  # 前左组总推力
        T34 = np.sqrt(u[2]**2 + u[3]**2)  # 前右组总推力
        T5 = u[4]                        # 尾部推进器推力
        
        # 计算倾角
        alpha1 = np.arctan2(u[0], u[1])  # 前左倾角
        alpha0 = np.arctan2(u[2], u[3])  # 前右倾角
        
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
        # self.data.ctrl[thrust_rt_id] = T34 / 2
        # self.data.ctrl[thrust_rb_id] = T34 / 2
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
        #=====================test=============================#
        # tilt_right_id = self.actuator_ids['tilt_right_joint']
        # tilt_left_id = self.actuator_ids['tilt_left_joint']

        # self.data.ctrl[tilt_right_id] = 0.0  # 右侧倾角
        # self.data.ctrl[tilt_left_id] = 0.0  # 左侧倾角
        
        # # 设置推力
        # # 右侧两个螺旋桨（每个推力为总推力的一半）
        # thrust_rt_id = self.actuator_ids['thrust_rt']
        # thrust_rb_id = self.actuator_ids['thrust_rb']
        # # self.data.ctrl[thrust_rt_id] = T34 / 2
        # # self.data.ctrl[thrust_rb_id] = T34 / 2
        # self.data.ctrl[thrust_rt_id] = 0
        # self.data.ctrl[thrust_rb_id] = 0
        # # 左侧两个螺旋桨
        # thrust_lt_id = self.actuator_ids['thrust_lt']
        # thrust_lb_id = self.actuator_ids['thrust_lb']
        # self.data.ctrl[thrust_lt_id] = 0
        # self.data.ctrl[thrust_lb_id] = 0
        
        # # 尾部推进器
        # thrust_tail_id = self.actuator_ids['thrust_tail']
        # self.data.ctrl[thrust_tail_id] = 0
    
    def update_control(self):
        """更新控制命令"""
        # 获取当前状态
        state = self.get_state()
        
        # 计算控制力和力矩
        f_c_body, tau_c = self.compute_control_wrench(state)
        
        # 分配执行器命令
        T12, T34, T5, alpha0, alpha1 = self.allocate_actuators(f_c_body, tau_c)
        # 应用控制
        self.apply_controls(T12, T34, T5, alpha0, alpha1)
    
    def print_status(self):
        """打印当前状态信息"""
        state = self.get_state()
        pos = state['position']
        euler = np.degrees(state['euler'])
        thrust_total = self.T12 + self.T34  # 总升力
        f_c_body = self.f_c_body
        tau_c = self.tau_c
        print(f"位置: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}) | "
              f"姿态: R={euler[0]:6.1f}° P={euler[1]:6.1f}° Y={euler[2]:6.1f}° | "
              f"控制力: {f_c_body[0]:5.2f}N {f_c_body[1]:5.2f}N {f_c_body[2]:5.2f}N | "
              f"力矩: {tau_c[0]:5.2f}N·m {tau_c[1]:5.2f}N·m {tau_c[2]:5.2f}N·m | "
              f"推力分配: T12={self.T12:.2f}N T34={self.T34:.2f}N T5={self.T5:.2f}N | "
              f"倾角: R={np.degrees(self.alpha0):5.1f}° L={np.degrees(self.alpha1):5.1f}°")

def main():
    """主函数 - 演示控制器使用"""
    print("=== 倾转旋翼无人机控制器演示 ===")
    
    # 创建控制器
    controller = HnuterController("scene.xml")
    
    # 启动viewer
    with viewer.launch_passive(controller.model, controller.data) as v:
        print("\n可视化窗口已启动")
        print("控制说明:")
        print("- 无人机将自动执行起飞、悬停、降落任务")
        print("- 按 Ctrl+C 可以停止仿真")
        
        # 飞行任务状态
        flight_state = "INIT"
        start_time = time.time()
        
        count = 0

        try:
            while v.is_running:
                current_time = time.time() - start_time
                
                # 状态机控制飞行任务
                if flight_state == "INIT":
                    controller.takeoff()
                    flight_state = "TAKEOFF"
                    print("起飞命令已发送")
                
                elif flight_state == "TAKEOFF":
                    state = controller.get_state()
                    # 高度误差小于0.1m时进入悬停
                    if abs(state['position'][2] - 1.0) < 0.1:
                        flight_state = "HOVER"
                        hover_start_time = time.time()
                        controller.hover()
                        print("进入悬停状态")
                
                elif flight_state == "HOVER":
                    # 悬停5秒后降落
                    if current_time - hover_start_time > 5.0:
                        flight_state = "LAND"
                        controller.land()
                        print("开始降落")
                
                elif flight_state == "LAND":
                    state = controller.get_state()
                    # 高度低于0.15m时结束
                    if state['position'][2] < 0.15:
                        flight_state = "COMPLETE"
                        print("降落完成")
                
                # 更新控制
                controller.update_control()
                
                # 仿真步进
                # mj.mj_step(controller.model, controller.data)
                count = count + 1
                if count % 5 == 0:
                    # 仿真步进
                    mj.mj_step(controller.model, controller.data)
                
                # 更新viewer
                v.sync()
                
                # 打印状态 (每0.5秒)
                if int(current_time * 10) % 5 == 0:
                    controller.print_status()
                
                # 简单延时
                time.sleep(0.001)
                
                # 检查飞行结束
                if flight_state == "COMPLETE":
                    print("\n飞行任务完成")
                    break
        
        except KeyboardInterrupt:
            print("\n仿真被用户中断")
        
        print("\n仿真结束")
    
    # 最终状态
    final_state = controller.get_state()
    print(f"最终位置: ({final_state['position'][0]:.2f}, {final_state['position'][1]:.2f}, {final_state['position'][2]:.2f})")

if __name__ == "__main__":
    main()