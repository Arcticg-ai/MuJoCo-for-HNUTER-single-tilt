#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import numpy as np
import math
import csv
import os
import time
from datetime import datetime
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class DroneVisualizationNode(Node):
    def __init__(self):
        super().__init__('drone_visualization_node')
        
        # 参数配置
        self.declare_parameter('update_rate', 20.0)  # Hz
        self.declare_parameter('trajectory_length', 200)
        self.declare_parameter('drone_scale', 0.5)
        self.declare_parameter('log_file_path', '')
        
        # 获取参数值
        update_rate = self.get_parameter('update_rate').value
        self.trajectory_length = self.get_parameter('trajectory_length').value
        self.drone_scale = self.get_parameter('drone_scale').value
        log_file_path = self.get_parameter('log_file_path').value
        
        # 初始化存储
        self.trajectory_points = []
        self.current_state = None
        self.last_update_time = time.time()
        
        # 创建发布器
        self.odom_pub = self.create_publisher(Odometry, 'drone/odom', 10)
        self.path_pub = self.create_publisher(Path, 'drone/path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'drone/markers', 10)
        self.status_pub = self.create_publisher(Marker, 'drone/status', 10)
        
        # TF广播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 如果提供了日志文件路径，则读取历史轨迹
        if log_file_path and os.path.exists(log_file_path):
            self.load_trajectory_from_log(log_file_path)
            self.get_logger().info(f"已从日志文件加载历史轨迹: {log_file_path}")
        
        # 创建定时器
        self.timer = self.create_timer(1.0/update_rate, self.update_visualization)
        
        self.get_logger().info("无人机可视化节点已启动")
    
    def load_trajectory_from_log(self, file_path):
        """从CSV日志文件加载轨迹数据"""
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # 提取位置数据
                    x = float(row['pos_x'])
                    y = float(row['pos_y'])
                    z = float(row['pos_z'])
                    
                    # 创建轨迹点
                    self.trajectory_points.append((x, y, z))
                    
                    # 限制轨迹长度
                    if len(self.trajectory_points) > self.trajectory_length:
                        self.trajectory_points.pop(0)
            
            self.get_logger().info(f"成功加载 {len(self.trajectory_points)} 个轨迹点")
        except Exception as e:
            self.get_logger().error(f"加载轨迹日志失败: {str(e)}")
    
    def update_state(self, position, orientation, velocity, angular_velocity, 
                    thrust, tilt_angles, control_force, control_torque):
        """更新无人机当前状态"""
        self.current_state = {
            'position': position,
            'orientation': orientation,
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            'thrust': thrust,
            'tilt_angles': tilt_angles,
            'control_force': control_force,
            'control_torque': control_torque,
            'timestamp': time.time()
        }
        
        # 添加当前位置到轨迹
        self.trajectory_points.append((position[0], position[1], position[2]))
        
        # 限制轨迹长度
        if len(self.trajectory_points) > self.trajectory_length:
            self.trajectory_points.pop(0)
    
    def publish_odometry(self):
        """发布里程计信息"""
        if not self.current_state:
            return
            
        odom = Odometry()
        odom.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        odom.child_frame_id = 'drone_base'
        
        # 设置位置
        odom.pose.pose.position.x = self.current_state['position'][0]
        odom.pose.pose.position.y = self.current_state['position'][1]
        odom.pose.pose.position.z = self.current_state['position'][2]
        
        # 设置方向 (使用四元数)
        roll, pitch, yaw = self.current_state['orientation']
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        
        # 设置速度
        odom.twist.twist.linear.x = self.current_state['velocity'][0]
        odom.twist.twist.linear.y = self.current_state['velocity'][1]
        odom.twist.twist.linear.z = self.current_state['velocity'][2]
        odom.twist.twist.angular.x = self.current_state['angular_velocity'][0]
        odom.twist.twist.angular.y = self.current_state['angular_velocity'][1]
        odom.twist.twist.angular.z = self.current_state['angular_velocity'][2]
        
        self.odom_pub.publish(odom)
    
    def publish_path(self):
        """发布轨迹路径"""
        path = Path()
        path.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        
        for point in self.trajectory_points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            path.poses.append(pose)
        
        self.path_pub.publish(path)
    
    def publish_drone_marker(self):
        """发布无人机可视化标记"""
        if not self.current_state:
            return
            
        marker_array = MarkerArray()
        
        # 无人机主体 (立方体)
        drone_marker = Marker()
        drone_marker.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        drone_marker.ns = 'drone'
        drone_marker.id = 0
        drone_marker.type = Marker.CUBE
        drone_marker.action = Marker.ADD
        drone_marker.scale.x = 0.4 * self.drone_scale
        drone_marker.scale.y = 0.6 * self.drone_scale
        drone_marker.scale.z = 0.2 * self.drone_scale
        drone_marker.color = ColorRGBA(r=0.2, g=0.7, b=1.0, a=0.8)
        
        # 设置位置和方向
        drone_marker.pose.position.x = self.current_state['position'][0]
        drone_marker.pose.position.y = self.current_state['position'][1]
        drone_marker.pose.position.z = self.current_state['position'][2]
        roll, pitch, yaw = self.current_state['orientation']
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        drone_marker.pose.orientation.x = qx
        drone_marker.pose.orientation.y = qy
        drone_marker.pose.orientation.z = qz
        drone_marker.pose.orientation.w = qw
        
        marker_array.markers.append(drone_marker)
        
        # 旋翼标记
        colors = [ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # 红色
                  ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # 绿色
                  ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # 蓝色
                  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)]  # 黄色
        
        # 旋翼位置偏移 (相对无人机主体)
        rotor_offsets = [
            (0.3, 0.3, 0.1),  # 前右
            (0.3, -0.3, 0.1),  # 前左
            (-0.3, 0.3, 0.1),  # 后右
            (-0.3, -0.3, 0.1)  # 后左
        ]
        
        for i, offset in enumerate(rotor_offsets):
            rotor_marker = Marker()
            rotor_marker.header = drone_marker.header
            rotor_marker.ns = 'rotors'
            rotor_marker.id = i + 1
            rotor_marker.type = Marker.CYLINDER
            rotor_marker.action = Marker.ADD
            rotor_marker.scale.x = 0.2 * self.drone_scale
            rotor_marker.scale.y = 0.2 * self.drone_scale
            rotor_marker.scale.z = 0.05 * self.drone_scale
            rotor_marker.color = colors[i]
            
            # 应用旋翼位置和方向
            rotor_marker.pose.position.x = drone_marker.pose.position.x + offset[0]
            rotor_marker.pose.position.y = drone_marker.pose.position.y + offset[1]
            rotor_marker.pose.position.z = drone_marker.pose.position.z + offset[2]
            
            # 应用旋翼倾斜角度
            if i < 2:  # 前旋翼组
                tilt_angle = self.current_state['tilt_angles'][0] if i == 0 else self.current_state['tilt_angles'][1]
                rotor_marker.pose.orientation = self.euler_to_quaternion_msg(0, tilt_angle, yaw)
            else:  # 后旋翼组不倾斜
                rotor_marker.pose.orientation = drone_marker.pose.orientation
            
            marker_array.markers.append(rotor_marker)
        
        # 尾部推进器
        tail_marker = Marker()
        tail_marker.header = drone_marker.header
        tail_marker.ns = 'tail'
        tail_marker.id = 5
        tail_marker.type = Marker.ARROW
        tail_marker.action = Marker.ADD
        tail_marker.scale.x = 0.5 * self.drone_scale
        tail_marker.scale.y = 0.1 * self.drone_scale
        tail_marker.scale.z = 0.1 * self.drone_scale
        tail_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)  # 橙色
        
        # 设置位置和方向 (指向后方)
        tail_marker.pose.position.x = drone_marker.pose.position.x - 0.4
        tail_marker.pose.position.y = drone_marker.pose.position.y
        tail_marker.pose.position.z = drone_marker.pose.position.z
        tail_marker.pose.orientation = self.euler_to_quaternion_msg(0, 0, yaw + math.pi)
        
        marker_array.markers.append(tail_marker)
        
        self.marker_pub.publish(marker_array)
    
    def publish_status_markers(self):
        """发布控制状态标记"""
        if not self.current_state:
            return
            
        status_marker = Marker()
        status_marker.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        status_marker.ns = 'status'
        status_marker.id = 0
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        status_marker.scale.z = 0.15  # 文本大小
        status_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        
        # 设置位置 (在无人机上方)
        status_marker.pose.position.x = self.current_state['position'][0]
        status_marker.pose.position.y = self.current_state['position'][1]
        status_marker.pose.position.z = self.current_state['position'][2] + 0.8
        
        # 构建状态文本
        thrust = self.current_state['thrust']
        tilt_angles = self.current_state['tilt_angles']
        force = self.current_state['control_force']
        torque = self.current_state['control_torque']
        
        status_text = (
            f"Thrust: T12={thrust[0]:.2f}N, T34={thrust[1]:.2f}N, T5={thrust[2]:.2f}N\n"
            f"Tilt: α0={math.degrees(tilt_angles[0]):.1f}°, α1={math.degrees(tilt_angles[1]):.1f}°\n"
            f"Force: X={force[0]:.2f}N, Y={force[1]:.2f}N, Z={force[2]:.2f}N\n"
            f"Torque: X={torque[0]:.2f}Nm, Y={torque[1]:.2f}Nm, Z={torque[2]:.2f}Nm"
        )
        
        status_marker.text = status_text
        self.status_pub.publish(status_marker)
    
    def publish_tf(self):
        """发布TF变换"""
        if not self.current_state:
            return
            
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'drone_base'
        
        # 设置位置
        t.transform.translation.x = self.current_state['position'][0]
        t.transform.translation.y = self.current_state['position'][1]
        t.transform.translation.z = self.current_state['position'][2]
        
        # 设置方向
        roll, pitch, yaw = self.current_state['orientation']
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
    
    def update_visualization(self):
        """更新所有可视化组件"""
        if not self.current_state:
            return
            
        # 发布所有可视化数据
        self.publish_odometry()
        self.publish_path()
        self.publish_drone_marker()
        self.publish_status_markers()
        self.publish_tf()
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """欧拉角转四元数"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def euler_to_quaternion_msg(self, roll, pitch, yaw):
        """欧拉角转四元数消息"""
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        return self.create_quaternion(qx, qy, qz, qw)
    
    def create_quaternion(self, x, y, z, w):
        """创建四元数消息"""
        q = Quaternion()
        q.x = x
        q.y = y
        q.z = z
        q.w = w
        return q

def main(args=None):
    rclpy.init(args=args)
    node = DroneVisualizationNode()
    
    # 示例数据更新（实际应用中应从MuJoCo仿真接收）
    def simulate_drone_movement():
        t = time.time()
        # 模拟莫比乌斯环轨迹
        radius = 1.0
        height = 2.5
        period = 20.0
        u = 2 * np.pi * t / period
        x = radius * (1 + 0.5 * np.cos(u/2)) * np.cos(u)
        y = radius * (1 + 0.5 * np.cos(u/2)) * np.sin(u)
        z = 0.5 * radius * np.sin(u/2) + height
        
        # 模拟方向（偏航角跟随速度方向）
        yaw = u
        
        # 模拟推力
        thrust = (15.0, 15.0, 5.0)
        
        # 模拟倾角
        tilt_angles = (math.sin(u), math.cos(u))
        
        # 设置模拟状态
        node.update_state(
            position=[x, y, z],
            orientation=[0.0, 0.0, yaw],  # roll, pitch, yaw
            velocity=[0.5, 0.5, 0.0],
            angular_velocity=[0.0, 0.0, 0.1],
            thrust=thrust,
            tilt_angles=tilt_angles,
            control_force=[5.0, 0.0, 20.0],
            control_torque=[0.1, 0.05, 0.2]
        )
    
    # 创建定时器模拟无人机运动
    node.create_timer(0.05, simulate_drone_movement)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()