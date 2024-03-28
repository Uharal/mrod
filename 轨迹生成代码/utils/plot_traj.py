
import numpy as np
import pandas as pd
def plot_traj(traj,starttime = None,days = 1,fix = False):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    traj['time'] = pd.to_datetime(traj['time'])

    if fix:
        g = traj
        g = g.sort_values(['uid','time'])
        g2 = g.copy()
        g2['time'] = pd.to_datetime(g2['time'])
        g2['time'] = g2['time'].shift(-1)-1*pd.Timedelta('1 second')
        g2 = g2[g2['uid'] == g2['uid'].shift(-1)]
        traj = pd.concat([g,g2]).sort_values(['uid','time'])
        
    if starttime is None:
        starttime = pd.to_datetime(traj['time'].min().date())
    else:
        starttime = pd.to_datetime(starttime)

    endtime = starttime + pd.Timedelta(days,unit='d')

    traj = traj[(traj['time']>=starttime)&(traj['time']<endtime)]

    traj_start = traj.iloc[:1].copy()
    traj_start['time'] = pd.to_datetime(starttime)
    traj_end = traj.iloc[-1:].copy()
    traj_end['time'] = pd.to_datetime(endtime)
    traj = pd.concat([traj_start,traj,traj_end],axis=0)
    traj['timestamp'] = (traj['time']-pd.to_datetime(starttime)).dt.total_seconds()

    # 1. 数据预处理
    # 请确保DataFrame df 中的日期时间列已经正确解析为datetime类型

    # 2. 使用Matplotlib绘制轨迹图
    fig = plt.figure(figsize=(6, 6),dpi = 300)  # 调整图形大小
    ax = fig.add_subplot(111, projection='3d')


    # 绘制轨迹线
    ax.plot(traj['lon'], traj['lat'], zs=traj['timestamp']/3600)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 设置 z 轴标题为 "Hour"，并旋转 90°
    ax.set_zlabel('Hour', rotation=90)

    # 设置旋转角度
    ax.view_init(elev=15, azim=25)

    # 设置 z 轴范围

    #ax.set_zlim(0,24)
    ax.set_xlim(traj['lon'].min(),traj['lon'].max())
    ax.set_ylim(traj['lat'].min(),traj['lat'].max())

    # 设置 x 轴和 y 轴的刻度位置
    x_ticks = np.linspace(traj['lon'].min(), traj['lon'].max(), 5)
    y_ticks = np.linspace(traj['lat'].min(), traj['lat'].max()-0.01, 5)
    #z_ticks = np.linspace(0, 24, 7)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    #ax.set_zticks(z_ticks)


    # 设置纬度刻度不使用科学计数法
    def format_func(value, tick_number):
        return "{:.2f}".format(value)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))


    # 设置底部的背景色为白色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('w')  # 设置底部边缘颜色为白色

    # 设置格线颜色

    # 设置网格线颜色
    ax.xaxis._axinfo['grid'].update(color = '#dfe5ef', linestyle = '-')
    ax.yaxis._axinfo['grid'].update(color = '#dfe5ef', linestyle = '-')
    ax.zaxis._axinfo['grid'].update(color = '#dfe5ef', linestyle = '-')

    # 调整图形边距
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
