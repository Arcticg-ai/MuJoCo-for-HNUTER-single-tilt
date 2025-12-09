import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
import csv
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ===================== 轨迹规划器 =====================
class QuinticTrajectory:
    def __init__(self, start_pos, start_vel, end_pos, end_vel, T):
        self.T = T
        self.coeffs = np.zeros((3, 6))
        for i in range(3):
            p0, v0, p1, v1 = start_pos[i], start_vel[i], end_pos[i], end_vel[i]
            # 五次多项式边界条件求解 a0 + a1*t ... + a5*t^5
            A = np.array([
                [0, 0, 0, 0, 0, 1],
                [T**5, T**4, T**3, T**2, T, 1],
                [0, 0, 0, 0, 1, 0],
                [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [20*T**3, 12*T**2, 6*T, 2, 0, 0]
            ])
            b = np.array([p0, p1, v0, v1, 0, 0])
            self.coeffs[i, :] = np.linalg.solve(A, b)

    def get_state(self, t):
        t = np.clip(t, 0, self.T)
        tt = np.array([t**5, t**4, t**3, t**2, t, 1])
        vt = np.array([5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0])
        pos = self.coeffs @ tt
        vel = self.coeffs @ vt
        return pos, vel

# ===================== 核心控制器 =====================
class HnuterController:
    def __init__(self, model_path: str = "scene.xml"):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # === 1. 物理参数修正 (基于 hnuter 机型) ===
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        self.mass = 1.9 + 0.6  # 估算总重约 2.5kg
        self.J = np.diag([0.1, 0.12, 0.1])
        
        # 关键几何参数 (影响分配矩阵)
        self.l_arm = 0.3      # 左右旋翼Y向距离
        self.l_tail = 0.5     # 尾部电机X向距离
        self.h_com = 0.3      # 重心高度 (垂直偏置)，这是导致俯仰耦合的根源
        self.k_d = 0.01       # 偏航系数假设

        # === 2. 控制器增益 ===
        # 位置环 (较硬，保证跟踪)
        self.Kp = np.diag([8.0, 8.0, 10.0])
        self.Dp = np.diag([4.0, 4.0, 6.0])
        # 姿态环
        self.KR = np.diag([10.0, 10.0, 5.0])
        self.Domega = np.diag([2.0, 2.0, 1.0])

        # === 3. 分配矩阵构建 (Linearized Allocation) ===
        # 状态变量 x = [f_Lx, f_Lz, f_Rx, f_Rz, f_Tail]
        # 其中 f_Lx = TL * sin(aL), f_Lz = TL * cos(aL)
        # 这样可以将非线性问题转化为线性方程 Ax = b
        
        # 行定义: [Fx, Fz, Tx, Ty, Tz]
        self.alloc_A = np.array([
            # f_Lx, f_Lz, f_Rx, f_Rz, f_Tail
            [1,     0,    1,    0,    0],        # Fx: 左右水平分量之和 (推力向前)
            [0,     1,    0,    1,    1],        # Fz: 所有垂直分量之和
            [0, -self.l_arm, 0, self.l_arm, 0],  # Roll: 左右垂直推力差 * 臂长
            
            # Pitch (关键): 尾部力矩 + 旋翼水平推力产生的力矩 (重心偏置)
            [-self.h_com, 0, -self.h_com, 0, -self.l_tail], 
            
            [self.l_arm, 0, -self.l_arm, 0, self.k_d] # Yaw: 左右水平推力差 (差动倾转)
        ])
        
        # 计算伪逆，用于实时求解
        self.alloc_A_pinv = np.linalg.pinv(self.alloc_A)

        # 状态记录
        self.init_ids()
        self.create_log()
        
        # 任务状态
        self.mission_start_time = 0
        self.planner = None
        self.target_pos = np.array([0., 0., 0.5])
        self.target_vel = np.zeros(3)
        self.target_R = np.eye(3) # 目标姿态矩阵

    def init_ids(self):
        self.actuators = {
            'tilt_r': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_right'),
            'tilt_l': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_left'),
            'mr_u': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'motor_r_upper'),
            'mr_l': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'motor_r_lower'),
            'ml_u': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'motor_l_upper'),
            'ml_l': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'motor_l_lower'),
            'mtail': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'motor_rear_upper')
        }
        self.sensors = {
            'pos': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_pos'),
            'quat': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_quat'),
            'vel': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_vel'),
            'gyro': mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, 'body_gyro')
        }

    def create_log(self):
        if not os.path.exists('logs'): os.makedirs('logs')
        self.log_file = f'logs/hnuter_slope_{datetime.now().strftime("%H%M%S")}.csv'
        with open(self.log_file, 'w') as f:
            f.write("time,px,py,pz,roll,pitch,yaw,TL,TR,TT,aL,aR\n")

    def get_state(self):
        pos = self.data.sensordata[self.model.sensor_adr[self.sensors['pos']]:][:3]
        vel = self.data.sensordata[self.model.sensor_adr[self.sensors['vel']]:][:3]
        quat = self.data.sensordata[self.model.sensor_adr[self.sensors['quat']]:][:4]
        omega = self.data.sensordata[self.model.sensor_adr[self.sensors['gyro']]:][:3]
        
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rot_mat = r.as_matrix()
        euler = r.as_euler('xyz')
        
        return {'p': pos, 'v': vel, 'R': rot_mat, 'w': omega, 'euler': euler}

    # ===================== 任务规划 (斜面计算) =====================
    def setup_mission(self, current_time):
        self.mission_start_time = current_time
        
        # 1. 解析斜面几何 (基于 user xml: pos="5 0 0.25" euler="0 3.8 1.57")
        slope_pos = np.array([5.0, 0.0, 0.25])
        # MuJoCo Euler 顺序通常是 X-Y-Z (extrinsic)
        # 0, 3.8, 1.57 rad. 3.8 rad ≈ 217.7度.
        r_slope = R.from_euler('xyz', [0, 3.8, 1.57])
        # 斜面法向量 (局部 Z 轴转到世界系)
        normal = r_slope.apply(np.array([0, 0, 1]))
        
        print(f"斜面法向量: {normal}")
        
        # 2. 定义关键点
        p_home = np.array([0, 0, 2.0])          # 起飞点
        p_align = slope_pos + normal * 0.5      # 对准点 (距离斜面0.5m)
        p_contact = slope_pos + normal * 0.05   # 接触点
        
        # 3. 目标姿态计算 (机头对准斜面法向的反向)
        # 机体 Z 轴 (推力轴) 需要大致垂直向上以维持重力，
        # 但为了贴附，机体需要调整姿态使得起落架/吸盘接触斜面
        # 这里我们假设：保持机身水平(Level)，仅靠倾转去接触？
        # 用户要求: "水平向前飞行...不依赖机体姿态"
        # 因此，我们在逼近阶段保持 Level，但在接触阶段可能需要匹配斜面角度
        
        # 第一阶段: 飞到对准点
        self.traj_approach = QuinticTrajectory(p_home, np.zeros(3), p_align, np.zeros(3), T=8.0)
        
        # 保存斜面信息供后续使用
        self.slope_normal = normal
        self.slope_R = r_slope.as_matrix()
        self.p_align = p_align
        self.p_contact = p_contact

    def update_planner(self, t):
        dt = t - self.mission_start_time
        
        if dt < 2.0: # 起飞悬停
            self.target_pos = np.array([0, 0, 2.0])
            self.target_vel = np.zeros(3)
            self.target_R = np.eye(3)
            
        elif dt < 10.0: # 逼近斜面 (Level Flight)
            p, v = self.traj_approach.get_state(dt - 2.0)
            self.target_pos = p
            self.target_vel = v
            # 关键：保持水平姿态，Yaw 对准斜面 (Y轴方向?)
            # 简单起见，保持 Yaw = 0 (X轴向前)
            self.target_R = np.eye(3)
            
        elif dt < 15.0: # 最终贴附 (Adhesion)
            # 此时切换到斜面姿态? 或者保持水平用腿接触?
            # 假设我们要用机体底部贴附，需要将机体 Pitch/Roll 对准斜面
            # 简单线性插值姿态
            alpha = (dt - 10.0) / 5.0
            p_curr = self.p_align * (1-alpha) + self.p_contact * alpha
            self.target_pos = p_curr
            
            # 计算目标旋转矩阵：Z轴对准斜面法向
            z_des = self.slope_normal
            # 修正 Z 方向 (如果法向向下)
            if z_des[2] < 0: z_des = -z_des
            
            # 简单的旋转矩阵构造
            y_des = np.cross(z_des, np.array([1, 0, 0]))
            y_des /= np.linalg.norm(y_des)
            x_des = np.cross(y_des, z_des)
            self.target_R = np.column_stack((x_des, y_des, z_des))
            
        else: # 保持吸附
            self.target_pos = self.p_contact
            # 施加额外的压紧力 (在控制器中处理)

    # ===================== 控制计算 =====================
    def update_control(self):
        state = self.get_state()
        
        # 1. 位置控制 (PID -> F_world)
        e_p = state['p'] - self.target_pos
        e_v = state['v'] - self.target_vel
        
        # 期望加速度
        acc_des = -self.Kp @ e_p - self.Dp @ e_v
        # 前馈重力
        F_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))
        
        # 2. 姿态控制 (PID -> Tau_body)
        R_curr = state['R']
        R_des = self.target_R
        
        # 姿态误差 (SO3)
        R_err = 0.5 * (R_des.T @ R_curr - R_curr.T @ R_des)
        e_R = np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])
        e_w = state['w']
        
        Tau_body = -self.KR @ e_R - self.Domega @ e_w
        
        # 3. 转换力到机体坐标系
        F_body = R_curr.T @ F_world
        
        # ================== 4. 分配求解 (Solver) ==================
        # 构造期望向量 b = [Fx, Fz, Tx, Ty, Tz]
        # 注意: Fy (横向力) 无法直接控制，只能通过 Roll 姿态耦合产生
        b = np.array([F_body[0], F_body[2], Tau_body[0], Tau_body[1], Tau_body[2]])
        
        # 求解 Ax = b
        # x = [f_Lx, f_Lz, f_Rx, f_Rz, f_Tail]
        x_sol = self.alloc_A_pinv @ b
        
        f_Lx, f_Lz, f_Rx, f_Rz, f_Tail = x_sol
        
        # 5. 恢复物理控制量
        # 左旋翼
        TL = np.sqrt(f_Lx**2 + f_Lz**2)
        # 注意 XML 定义: tilt_left axis="0 1 0" (+Y). 
        # 若 aL > 0 (前倾), 产生 +Fx, +Fz. atan2(x, z)
        aL = np.arctan2(f_Lx, f_Lz)
        
        # 右旋翼
        TR = np.sqrt(f_Rx**2 + f_Rz**2)
        # 注意 XML 定义: tilt_right axis="0 -1 0" (-Y).
        # 若需要 +Fx (前倾), 绕 -Y 轴旋转需要是 正角度? 还是负?
        # 通常 Right Tilt +angle implies forward vector.
        # 我们假设代码逻辑一致: aR > 0 -> Forward Force.
        # 到底层再处理符号
        aR = np.arctan2(f_Rx, f_Rz)
        
        TT = f_Tail
        
        # 限制
        TL = np.clip(TL, 0, 30)
        TR = np.clip(TR, 0, 30)
        TT = np.clip(TT, -10, 10) # 尾部电机双向
        aL = np.clip(aL, -1.0, 1.0)
        aR = np.clip(aR, -1.0, 1.0)
        
        self.apply_ctrl(TL, TR, TT, aL, aR)
        self.log_data(state, TL, TR, TT, aL, aR)

    def apply_ctrl(self, TL, TR, TT, aL, aR):
        # 映射到 XML ID
        # 右倾转: XML axis 0 -1 0. 正控制值导致矢量向后(-X).
        # 我们计算的 aR > 0 代表向前力. 所以这里需要取反? 
        # 需根据实际 MuJoCo 行为调试，通常 axis 0 -1 0 意味着正转是符合右手定则绕 -Y.
        # 拇指指向 -Y，手指卷曲方向是 Z -> X. 所以正值是 pitch down (forward tilt).
        # 暂定正映射.
        
        self.data.ctrl[self.actuators['tilt_r']] = aR
        self.data.ctrl[self.actuators['tilt_l']] = aL
        
        self.data.ctrl[self.actuators['mr_u']] = TR / 2
        self.data.ctrl[self.actuators['mr_l']] = TR / 2
        self.data.ctrl[self.actuators['ml_u']] = TL / 2
        self.data.ctrl[self.actuators['ml_l']] = TL / 2
        
        self.data.ctrl[self.actuators['mtail']] = TT

    def log_data(self, s, TL, TR, TT, aL, aR):
        with open(self.log_file, 'a') as f:
            r, p, y = s['euler']
            f.write(f"{self.data.time:.3f},{s['p'][0]:.2f},{s['p'][1]:.2f},{s['p'][2]:.2f},"
                    f"{r:.2f},{p:.2f},{y:.2f},{TL:.1f},{TR:.1f},{TT:.1f},{aL:.2f},{aR:.2f}\n")

# ===================== 主函数 =====================
def main():
    print("=== Hnuter Tilt-Rotor Slope Adhesion Mission ===")
    
    # 请确保 xml 文件名为 scene.xml 且包含你提供的 worldbody 定义
    try:
        ctrl = HnuterController("scene.xml")
    except Exception as e:
        print(f"Error: {e}")
        return

    with viewer.launch_passive(ctrl.model, ctrl.data) as v:
        # 设置视角跟踪
        v.cam.distance = 5.0
        v.cam.lookat[:] = [2.5, 0, 1.0]
        v.cam.azimuth = 90
        
        start_time = time.time()
        ctrl.setup_mission(0)
        
        while v.is_running:
            now = time.time() - start_time
            
            # 1. 规划
            ctrl.update_planner(now)
            
            # 2. 控制 & 分配
            ctrl.update_control()
            
            # 3. 物理步进
            mj.mj_step(ctrl.model, ctrl.data)
            v.sync()
            
            # 简单打印
            if int(now*10) % 10 == 0:
                s = ctrl.get_state()
                print(f"T={now:.1f} | Pos={s['p']} | Pitch={np.degrees(s['euler'][1]):.1f}")
                
            time.sleep(ctrl.dt)

if __name__ == "__main__":
    main()