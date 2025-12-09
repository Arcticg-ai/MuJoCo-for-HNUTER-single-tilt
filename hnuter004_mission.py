import mujoco as mj
import mujoco.viewer as viewer
import time
import numpy as np
from typing import Dict

class MinimalTiltrotorInterface:
    def __init__(self, model_path: str = "mission_scene.xml"):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data  = mj.MjData(self.model)

        # 检查 XML 的 timestep 是否足够小
        if self.model.opt.timestep > 0.004:
            print(f"警告: XML timestep ({self.model.opt.timestep}) 过大，建议在 XML 中改为 0.002 以防止仿真发散！")

        # ID Mapping
        self.act_id: Dict[str, int] = {}
        try:
            actuators = ["tilt_right", "tilt_left", "motor_r_upper", "motor_r_lower", 
                         "motor_l_upper", "motor_l_lower", "motor_rear_upper"]
            for name in actuators:
                self.act_id[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            self.site_imu = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "imu")
        except Exception as e:
            print(f"初始化错误: 找不到执行器或传感器 ID. 错误信息: {e}")
            raise

        # 状态记忆 (用于微分)
        self.prev_alpha_des = np.zeros(2) # [right, left]
        self.dt = self.model.opt.timestep # 自动同步 XML 的时间步长

    def get_state(self):
        pos = self.data.site_xpos[self.site_imu]
        mat = self.data.site_xmat[self.site_imu].reshape(3, 3)
        
        # 获取速度 (转换到世界坐标系更安全)
        # 注意：body_vel 传感器通常输出的是局部坐标系速度，这里简化直接读取
        vel = self.data.sensor("body_vel").data.copy()
        omega = self.data.sensor("body_gyro").data.copy()
        
        # 如果发现 NaN，立即停止，方便调试
        if np.isnan(pos).any() or np.isnan(vel).any():
            print("错误: 检测到状态为 NaN")
            return np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)
            
        return pos, vel, mat, omega

    def set_actuators(self, T_right_total, T_left_total, T_rear, alpha_right, alpha_left):
        # === 数值钳位 (Safety Clamps) ===
        # 防止物理引擎接收到 +/- 无穷大的力
        alpha_right = np.clip(alpha_right, -1.5, 1.5) # 限制倾转角约 +/- 85度
        alpha_left  = np.clip(alpha_left, -1.5, 1.5)
        
        # 限制推力在物理可行范围内 (假设单边最大推力 30N)
        T_right_total = np.clip(T_right_total, 0.0, 40.0)
        T_left_total  = np.clip(T_left_total, 0.0, 40.0)
        T_rear        = np.clip(T_rear, -10.0, 10.0) # 尾部推力允许反转

        # 写入控制量
        self.data.ctrl[self.act_id["tilt_right"]] = alpha_right
        self.data.ctrl[self.act_id["tilt_left"]]  = alpha_left
        
        self.data.ctrl[self.act_id["motor_l_upper"]] = T_left_total / 2.0
        self.data.ctrl[self.act_id["motor_l_lower"]] = T_left_total / 2.0
        self.data.ctrl[self.act_id["motor_r_upper"]] = T_right_total / 2.0
        self.data.ctrl[self.act_id["motor_r_lower"]] = T_right_total / 2.0
        self.data.ctrl[self.act_id["motor_rear_upper"]] = T_rear

class GeometricController:
    def __init__(self, mass=2.3):
        self.m = mass
        self.g = 9.81
        self.e3 = np.array([0, 0, 1])

        # 调低一点增益，增加稳定性
        self.kp_pos = np.diag([6.0, 6.0, 8.0]) 
        self.kd_pos = np.diag([4.0, 4.0, 5.0])
        self.k_R = np.diag([1.0, 1.0, 0.5])
        self.k_omega = np.diag([0.2, 0.2, 0.1])

        self.k_lag_comp = 0.05 

    def update(self, pos, vel, R_curr, omega, pos_des, vel_des, yaw_des):
        # 1. 位置控制
        e_p = pos - pos_des
        e_v = vel - vel_des
        
        # 计算期望力向量
        F_vector = -self.kp_pos @ e_p - self.kd_pos @ e_v + self.m * self.g * self.e3
        
        # === 保护措施 1: 限制最大倾斜力 ===
        # 如果位置误差太大，F_vector会非常大，导致瞬间翻转。
        # 我们限制水平分力不超过重力的 0.5 倍 (即最大倾角约 26度)
        max_horiz_force = 0.5 * self.m * self.g
        horiz_force_norm = np.linalg.norm(F_vector[:2])
        if horiz_force_norm > max_horiz_force:
            scale = max_horiz_force / horiz_force_norm
            F_vector[:2] *= scale

        # 2. 姿态计算
        # 计算期望 Z 轴
        f_norm = np.linalg.norm(F_vector)
        if f_norm < 1e-6: 
            zb_des = self.e3 # 防止除零
        else:
            zb_des = F_vector / f_norm
        
        # 计算期望 Y 轴 (处理奇异点)
        yc_world = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0])
        
        # 如果 zb_des 和 yc_world 平行 (极度俯仰时)，叉乘会得到 0
        # 此时改用 x轴辅助计算
        xb_des = np.cross(yc_world, zb_des)
        if np.linalg.norm(xb_des) < 1e-3:
            xb_des = np.cross(np.array([1, 0, 0]), zb_des)
            
        xb_des = xb_des / np.linalg.norm(xb_des)
        yb_des = np.cross(zb_des, xb_des)
        
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        
        # 3. 姿态误差
        R_err_mat = 0.5 * (R_des.T @ R_curr - R_curr.T @ R_des)
        e_R = np.array([R_err_mat[2, 1], R_err_mat[0, 2], R_err_mat[1, 0]])
        e_omega = omega
        
        tau_des = -self.k_R @ e_R - self.k_omega @ e_omega
        
        # 将期望力转换到机体坐标系
        F_body_des = R_curr.T @ F_vector
        
        return F_body_des, tau_des

    def allocation(self, F_body, tau_body, interface_obj):
        l1 = 0.3 
        l2 = 0.4

        # === 保护措施 2: 限制输出力矩 ===
        # 即使在极端情况下，也不要尝试产生超过物理极限的力矩
        tau_body = np.clip(tau_body, -2.0, 2.0) 

        # 基本推力分配
        f_z_cmd = max(0, F_body[2]) # Z轴推力必须为正
        
        # 滚转力矩 -> 左右推力差
        delta_T_roll = tau_body[0] / l1
        
        T_right = 0.5 * f_z_cmd - 0.5 * delta_T_roll
        T_left  = 0.5 * f_z_cmd + 0.5 * delta_T_roll
        
        # 防止推力为负
        T_right = max(0.0, T_right)
        T_left  = max(0.0, T_left)
        T_tail = 0.0

        # 前向力分配 -> 倾转角
        f_x_cmd = F_body[0]
        
        # 计算所需的倾转角
        # 使用安全的反正弦函数
        safe_Tr = max(1.0, T_right) # 防止除零
        safe_Tl = max(1.0, T_left)
        
        arg_r = np.clip(0.5 * f_x_cmd / safe_Tr, -0.8, 0.8)
        arg_l = np.clip(0.5 * f_x_cmd / safe_Tl, -0.8, 0.8)
        
        alpha_r_des = np.arcsin(arg_r)
        alpha_l_des = np.arcsin(arg_l)
        
        # 偏航力矩辅助 (差动倾转)
        yaw_tilt = np.clip(tau_body[2] * 0.3, -0.2, 0.2)
        alpha_r_des -= yaw_tilt
        alpha_l_des += yaw_tilt
        
        # === 创新点: 舵机迟滞补偿 ===
        d_alpha_r = (alpha_r_des - interface_obj.prev_alpha_des[0]) / interface_obj.dt
        d_alpha_l = (alpha_l_des - interface_obj.prev_alpha_des[1]) / interface_obj.dt
        
        alpha_r_cmd = alpha_r_des + self.k_lag_comp * d_alpha_r
        alpha_l_cmd = alpha_l_des + self.k_lag_comp * d_alpha_l
        
        interface_obj.prev_alpha_des = np.array([alpha_r_des, alpha_l_des])
        
        # 俯仰力矩 -> 尾部推力
        T_pitch_comp = tau_body[1] / l2
        T_tail = -T_pitch_comp 

        return T_right, T_left, T_tail, alpha_r_cmd, alpha_l_cmd

def main():
    try:
        interface = MinimalTiltrotorInterface("mission_scene.xml")
    except Exception:
        return

    controller = GeometricController(mass=2.3)

    # 目标：悬停在 1米高度
    pos_des = np.array([0.0, 0.0, 1.0])
    vel_des = np.array([0.0, 0.0, 0.0])
    yaw_des = 0.0

    print("=== 开始仿真 ===")
    print("提示: 如果无人机翻滚，请检查电机转向定义或 XML 中的质量分布")
    
    with viewer.launch_passive(interface.model, interface.data) as v:
        while v.is_running():
            step_start = time.time()
            
            # 1. 状态估计
            pos, vel, R, omega = interface.get_state()
            if np.isnan(pos[0]): break # 安全退出
            
            # 2. 几何控制
            F_body, tau_body = controller.update(pos, vel, R, omega, pos_des, vel_des, yaw_des)
            
            # 3. 分配 + 补偿
            T_r, T_l, T_tail, a_r, a_l = controller.allocation(F_body, tau_body, interface)
            
            # 4. 执行
            interface.set_actuators(T_r, T_l, T_tail, a_r, a_l)
            
            # 5. 步进
            mj.mj_step(interface.model, interface.data)
            v.sync()
            
            # 保持实时性
            time_until_next = interface.dt - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()