import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math

class DroneSimulationTest:
    def __init__(self, model_path: str = "scene.xml"):
        # 加载MuJoCo模型
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # 物理参数
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        
        # 执行器名称映射
        self._get_actuator_ids()
        
        # 状态变量
        self.prev_velocity = np.zeros(3)  # 上一时刻速度
        self.acceleration = np.zeros(3)    # 当前加速度
        self.total_force = np.zeros(3)     # 总外力
        self.total_thrust = 0.0            # 总推力值
        
        # Z方向位置控制器参数
        self.target_z = 1.5  # 初始目标高度
        self.Kp_z = 0.5     # 位置比例增益 (提高)
        self.Kd_z = 0.01      # 速度微分增益 (提高)
        self.Ki_z = 0.01      # 积分增益 (降低)
        self.integral_error = 0.0  # 积分误差
        
        print("无人机测试初始化完成")
    
    def _get_actuator_ids(self):
        """获取执行器ID"""
        self.actuator_ids = {}
        
        # 倾转执行器
        self.actuator_ids['tilt_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_right')
        self.actuator_ids['tilt_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_left')
        
        # 推力执行器
        thrust_names = ['thrust_rb', 'thrust_rt', 'thrust_lb', 'thrust_lt', 'thrust_tail']
        for name in thrust_names:
            self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
    
    def get_body_position(self):
        """获取无人机位置"""
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hnuter")
        return self.data.xpos[body_id].copy()
    
    def get_body_quaternion(self):
        """获取无人机四元数姿态"""
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hnuter")
        return self.data.xquat[body_id].copy()
    
    def get_body_velocity(self):
        """获取无人机速度"""
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hnuter")
        return self.data.cvel[body_id][:3].copy()
    
    def get_body_acceleration(self):
        """获取无人机加速度"""
        return self.acceleration.copy()
    
    def get_total_force(self):
        """获取总外力"""
        return self.total_force.copy()
    
    def get_total_thrust(self):
        """获取总推力"""
        return self.total_thrust
    
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
    
    def set_target_height(self, target_z: float):
        """设置目标高度"""
        self.target_z = target_z
        self.integral_error = 0.0  # 重置积分项
        print(f"目标高度设置为: {target_z:.2f}m")
    
    def z_position_control(self):
        """Z方向位置控制器"""
        # 获取当前高度和速度
        current_pos = self.get_body_position()
        current_vel = self.get_body_velocity()
        
        # 计算高度误差
        error_z = self.target_z - current_pos[2]
        
        # 计算速度误差 (目标速度为0)
        error_vel_z = 0 - current_vel[2]
        
        # 更新积分误差 (带抗饱和限制)
        self.integral_error += error_z * self.dt
        self.integral_error = np.clip(self.integral_error, -5.0, 5.0)  # 限制积分项范围
        
        # PID控制律
        thrust_control = (self.Kp_z * error_z + 
                          self.Kd_z * error_vel_z + 
                          self.Ki_z * self.integral_error)
        
        # 重力补偿 (无人机质量 * 重力加速度) - 关键修复点
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hnuter")
        mass = self.model.body_mass[body_id]
        gravity_compensation = 2.36 * self.gravity
        
        # 总推力 = 控制输出 + 重力补偿
        total_thrust = thrust_control + gravity_compensation
        
        # 限制推力范围 (0-30N)
        total_thrust = np.clip(total_thrust, 0, 30)
        
        # 存储总推力值
        self.total_thrust = total_thrust
        
        return total_thrust
    
    def apply_control_thrust(self, total_thrust: float):
        """应用控制推力到所有执行器"""
        # 设置倾转角度为0（垂直）
        tilt_right_id = self.actuator_ids['tilt_right']
        tilt_left_id = self.actuator_ids['tilt_left']
        self.data.ctrl[tilt_right_id] = 0.0
        self.data.ctrl[tilt_left_id] = 0.0
        
        # 设置推力 (平均分配到四个主旋翼)
        thrust_per_motor = total_thrust / 4
        tail_thrust = 0.0   # 尾部推进器推力
        
        # 前右螺旋桨 (右下和右上)
        thrust_rt_id = self.actuator_ids['thrust_rt']
        thrust_rb_id = self.actuator_ids['thrust_rb']
        self.data.ctrl[thrust_rt_id] = thrust_per_motor
        self.data.ctrl[thrust_rb_id] = thrust_per_motor
        
        # 前左螺旋桨 (左下和左上)
        thrust_lt_id = self.actuator_ids['thrust_lt']
        thrust_lb_id = self.actuator_ids['thrust_lb']
        self.data.ctrl[thrust_lt_id] = thrust_per_motor
        self.data.ctrl[thrust_lb_id] = thrust_per_motor
        
        # 尾部推进器
        thrust_tail_id = self.actuator_ids['thrust_tail']
        self.data.ctrl[thrust_tail_id] = tail_thrust
    
    def update_dynamics(self):
        """更新动力学状态（加速度和总外力）"""
        current_velocity = self.get_body_velocity()
        
        # 计算加速度：a = (v_current - v_previous) / dt
        self.acceleration = (current_velocity - self.prev_velocity) / self.dt
        self.prev_velocity = current_velocity
        
        # 计算总外力：F = m*a
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "hnuter")
        mass = self.model.body_mass[body_id]
        # print(f"时间: {mass:.2f}kg")
        self.total_force = 2.36 * self.acceleration
    
    def print_status(self):
        """打印当前状态信息"""
        pos = self.get_body_position()
        quat = self.get_body_quaternion()
        euler = np.degrees(self._quat_to_euler(quat))  # 转换为角度
        vel = self.get_body_velocity()
        accel = self.get_body_acceleration()
        total_force = self.get_total_force()
        total_thrust = self.get_total_thrust()
        
        print(f"时间: {time.time():.2f}s")
        print(f"位置: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m | 目标高度: {self.target_z:.2f}m")
        print(f"姿态: Roll={euler[0]:.1f}°, Pitch={euler[1]:.1f}°, Yaw={euler[2]:.1f}°")
        print(f"速度: X={vel[0]:.2f}m/s, Y={vel[1]:.2f}m/s, Z={vel[2]:.2f}m/s")
        print(f"加速度: X={accel[0]:.2f}m/s², Y={accel[1]:.2f}m/s², Z={accel[2]:.2f}m/s²")
        print(f"总外力: X={total_force[0]:.2f}N, Y={total_force[1]:.2f}N, Z={total_force[2]:.2f}N")
        print(f"总推力: {total_thrust:.2f}N")
        print("--------------------------------------------------")

def main():
    """主函数 - 启动仿真测试"""
    print("=== 倾转旋翼无人机基础测试 ===")
    
    # 初始化测试
    test = DroneSimulationTest("scene.xml")
    
    # 设置目标高度
    test.set_target_height(1.5)  # 目标高度1.5米
    
    count = 0

    # 启动 Viewer
    with viewer.launch_passive(test.model, test.data) as v:
        print("\n仿真启动：")
        print("按 Ctrl+C 终止仿真")
        
        start_time = time.time()
        last_print_time = 0
        print_interval = 0.1  # 打印间隔 (秒)
        
        # 初始化速度记录
        test.prev_velocity = test.get_body_velocity()
        
        try:
            while v.is_running:
                current_time = time.time() - start_time
                
                # 计算控制推力
                total_thrust = test.z_position_control()
                
                # 应用控制推力
                test.apply_control_thrust(total_thrust)
                
                count = count + 1
                if count % 1 == 0:
                    # 仿真步进
                    mj.mj_step(test.model, test.data)    
                
                # 更新动力学状态
                test.update_dynamics()
                
                # 同步可视化
                v.sync()
                
                # 定期打印状态
                if current_time - last_print_time > print_interval:
                    test.print_status()
                    last_print_time = current_time
                
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n仿真中断")
        
        print("仿真结束")

if __name__ == "__main__":
    main()
