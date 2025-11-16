import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_drone_logs(log_file_path):
    """
    读取无人机日志文件并生成专业可视化图表
    
    参数:
        log_file_path (str): 日志文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"错误：文件 {log_file_path} 不存在")
        return
    
    # 读取CSV文件
    df = pd.read_csv(log_file_path)
    
    # 转换时间戳为可读格式
    start_time = df['timestamp'].iloc[0]
    df['time_elapsed'] = df['timestamp'] - start_time
    
    # 创建图表
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    plt.subplots_adjust(hspace=1.5)
    
    a=0
    b=1
    c=2
    d=3

    # 1. 位置变化图
    axs[a].plot(df['time_elapsed'], df['pos_x'], 'r-', label='X')
    axs[a].plot(df['time_elapsed'], df['pos_y'], 'g-', label='Y')
    axs[a].plot(df['time_elapsed'], df['pos_z'], 'b-', label='Z')
    axs[a].plot(df['time_elapsed'], df['target_z'], 'k--', label='Target high')
    axs[a].set_title('Position')
    axs[a].set_xlabel('time(s)')
    axs[a].set_ylabel('position(m)')
    axs[a].legend(loc='upper right')
    axs[a].grid(True)
    
    # 2. 姿态变化图
    axs[b].plot(df['time_elapsed'], df['roll'], 'r-', label='Roll')
    axs[b].plot(df['time_elapsed'], df['pitch'], 'g-', label='Pitch')
    axs[b].plot(df['time_elapsed'], df['yaw'], 'b-', label='Yaw')
    axs[b].set_title('Attitude')
    axs[b].set_xlabel('time(s)')
    axs[b].set_ylabel('Degrees')
    axs[b].legend(loc='upper right')
    axs[b].grid(True)
    
    # 3. 推力变化图
    axs[c].plot(df['time_elapsed'], df['T12'], 'r-', label='T12(left)')
    axs[c].plot(df['time_elapsed'], df['T34'], 'g-', label='T34(right)')
    axs[c].plot(df['time_elapsed'], df['T5'], 'b-', label='T5')
    axs[c].set_title('Thrust')
    axs[c].set_xlabel('time(s)')
    axs[c].set_ylabel('Thrust(N)')
    axs[c].legend(loc='upper right')
    axs[c].grid(True)
    
    # 4. 倾转角度变化图
    axs[d].plot(df['time_elapsed'], df['alpha0'], 'r-', label='Right Tilt (α0)')
    axs[d].plot(df['time_elapsed'], df['alpha1'], 'g-', label='Left Tilt (α1)')
    axs[d].axhline(y=85, color='r', linestyle='--', alpha=0.5, label='SAFE LIMIT')
    axs[d].axhline(y=-85, color='r', linestyle='--', alpha=0.5, label='SAFE LIMIT')
    axs[d].set_title('Tilt Angles')
    axs[d].set_xlabel('time(s)')
    axs[d].set_ylabel('Degrees')
    axs[d].legend(loc='upper right')
    axs[d].grid(True)
    
    # 添加整体标题
    # timestamp = datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    # fig.suptitle(f'Tiltrotors aerocraft fly data', fontsize=16)
    
    # 保存图表
    output_path = log_file_path.replace('.csv', '_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"图表已保存至: {output_path}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 自动查找最新的日志文件
    log_dir = 'logs'
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        # 获取所有日志文件
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv') and f.startswith('drone_log_')]
        
        if log_files:
            # 按时间排序获取最新文件
            log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            latest_log = os.path.join(log_dir, log_files[0])
            print(f"找到最新日志文件: {latest_log}")
            plot_drone_logs(latest_log)
        else:
            print("日志目录中没有找到无人机日志文件")
    else:
        print(f"错误：日志目录 '{log_dir}' 不存在")