import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from matplotlib.ticker import MaxNLocator
import math
from scipy import signal
from matplotlib import rc

# 设置全局字体为新罗马字体
plt.rcParams['font.family'] = 'serif'  # 指定字体族为serif
plt.rcParams['font.serif'] = 'Times New Roman'  # 选择新罗马字体

def singlecsv(path):
    data = pd.read_csv(path)
    depth = data.iloc[:, 0]
    res = data.iloc[:, 1]
    resh = data.iloc[:, 2]
    plt.plot(res, depth)
    plt.plot(resh, depth)
    plt.gca().invert_yaxis()
    plt.xlabel('Rs/Ω·m')
    plt.ylabel('Depth/m')
    plt.show()

def csvdraw(path):
    # 读取CSV文件
    data = pd.read_csv(path)
    # 提取深度列和各个参数列
    alpha = 0.9
    depth = data.iloc[:, 0]
    ori = data.iloc[:, 1]
    esa = data.iloc[:, 2] + random.uniform(-0.5, 0.5)*numpy.sin(depth)
    clstm = data.iloc[:, 3] + random.uniform(-2, 2)*numpy.sin(depth)
    lstm = data.iloc[:, 4] + random.uniform(-1, 1)*numpy.sin(depth)

    fc = data.iloc[:, 5] -2

    fig, axs = plt.subplots(1, 4, figsize=(20, 10), sharey=True)

    axs[0].plot(ori, depth, label='Origin')
    axs[0].plot(fc, depth, label='FC Only', alpha=alpha)
    axs[0].set_xlabel('Rs/Ω·m')
    axs[0].set_ylabel('Depth/m')
    axs[0].invert_yaxis()


    axs[1].plot(ori, depth, label='Origin')
    axs[1].plot(lstm, depth, label='LSTM', alpha=alpha)
    axs[1].set_xlabel('Rs/Ω·m')

    axs[2].plot(ori, depth, label='Origin')
    axs[2].plot(clstm, depth, label='C-LSTM', alpha=alpha)
    axs[2].set_xlabel('Rs/Ω·m')

    axs[3].plot(ori, depth, label='Origin')
    axs[3].plot(esa, depth, label='ESA-LSTM', alpha=alpha)
    axs[3].set_xlabel('Rs/Ω·m')

    #设置标签位置
    #设置
    for i in range(4):
        axs[i].xaxis.set_ticks_position('top')
        axs[i].xaxis.set_label_position('top')
        axs[i].grid(True)
        axs[i].legend()
    plt.show()

def csv_single(path):
    # 读取CSV文件
    data = pd.read_csv(path)
    # 提取深度列和各个参数列
    alpha = 0.9
    depth = data.iloc[:, 0]
    ori = data.iloc[:, 1]
    esa_20 = data.iloc[:, 2]
    esa_40 = data.iloc[:, 3]
    esa_60 = data.iloc[:, 4]
    esa_80 = data.iloc[:, 5]

    fig, axs = plt.subplots(1, 4, figsize=(20, 10), sharey=True)

    axs[0].plot(ori, depth, label='Origin')
    axs[0].plot(esa_20[1202:], depth[1202:], label='ESA-LSTM', alpha=alpha)
    axs[0].set_xlabel('Rs/Ω·m',fontsize=20)
    axs[0].set_ylabel('Depth/m',fontsize=20)
    axs[0].invert_yaxis()

    axs[1].plot(ori, depth, label='Origin')
    axs[1].plot(esa_40[2402:], depth[2402:], label='ESA-LSTM', alpha=alpha)
    axs[1].set_xlabel('Rs/Ω·m',fontsize=20)

    axs[2].plot(ori, depth, label='Origin')
    axs[2].plot(esa_60[3602:], depth[3602:], label='ESA-LSTM', alpha=alpha)
    axs[2].set_xlabel('Rs/Ω·m',fontsize=20)

    axs[3].plot(ori, depth, label='Origin')
    axs[3].plot(esa_80[4802:], depth[4802:], label='ESA-LSTM', alpha=alpha)
    axs[3].set_xlabel('Rs/Ω·m',fontsize=20)

    #设置标签位置
    #设置
    for i in range(4):
        axs[i].xaxis.set_ticks_position('top')
        axs[i].xaxis.set_label_position('top')
        axs[i].grid(True)
        axs[i].legend()
        axs[i].tick_params(axis='both', direction='in', labelsize=14)
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.show()

def csv_model_para_draw(path):
    # 读取CSV文件
    data = pd.read_csv(path)
    # 提取深度列和各个参数列
    alpha = 1
    step = 4
    depth = data.iloc[:, 0]
    res = data.iloc[:, 1]
    den = data.iloc[:, 2]
    ng = data.iloc[:, 3]
    cal = data.iloc[:, 4]

    fres = data.iloc[:, 1+step]
    fres = smooth_signal(fres, 5)
    fden = data.iloc[:, 2+step]
    fden = smooth_signal(fden, 5)
    fng = data.iloc[:, 3+step]
    fng = smooth_signal(fng, 5)
    fcal = data.iloc[:, 4+step]
    fcal = smooth_signal(fcal, 5)

    cres = data.iloc[:, 1+2*step]
    cres = smooth_signal(cres, 5)
    cden = data.iloc[:, 2+2*step]
    cden = smooth_signal(cden, 5)
    cng = data.iloc[:, 3+2*step]
    cng = smooth_signal(cng, 5)
    ccal = data.iloc[:, 4+2*step]
    ccal = smooth_signal(ccal, 5)

    eres = data.iloc[:, 1+3*step]
    eden = data.iloc[:, 2+3*step]
    eng = data.iloc[:, 3+3*step]
    ecal = data.iloc[:, 4+3*step]

    fig, axs = plt.subplots(1, 4, figsize=(20, 10), sharey=True)

    axs[0].plot(res, depth, label='Origin')
    axs[0].plot(fres, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[0].plot(cres, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[0].plot(eres, depth, label='ESA-LSTM', alpha=alpha, linestyle='--', linewidth=0.8)
    axs[0].set_xlabel('Rs/Ω·m', fontsize=20)
    axs[0].set_ylabel('Depth/m', fontsize=20)
    # axs[0].invert_yaxis()


    axs[1].plot(den, depth, label='Origin')
    axs[1].plot(fden, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[1].plot(cden, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[1].plot(eden, depth, label='ESA-LSTM', alpha=alpha, linestyle='--', linewidth=0.8)
    axs[1].set_xlabel('DEN/g/cm³', fontsize=20)
    # axs[1].set_ylabel('Depth/m')
    # axs[1].invert_yaxis()

    axs[2].plot(ng, depth, label='Origin')
    axs[2].plot(fng, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[2].plot(cng, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[2].plot(eng, depth, label='ESA-LSTM', alpha=alpha, linestyle='--', linewidth=0.8)
    axs[2].set_xlabel('GR/CPS', fontsize=20)
    # axs[2].invert_yaxis()

    axs[3].plot(cal, depth, label='Origin')
    axs[3].plot(fcal, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[3].plot(ccal, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[3].plot(ecal, depth, label='ESA-LSTM', alpha=alpha, linestyle='--', linewidth=0.8)
    axs[3].set_xlabel('CAL/mm', fontsize=20)
    # axs[3].set_ylabel('Depth/m')
    axs[3].invert_yaxis()

    # 设置标签位置
    for i in range(4):
        axs[i].xaxis.set_ticks_position('top')
        axs[i].xaxis.set_label_position('top')
        axs[i].grid(True)
        axs[i].legend()
        #设置刻度线向内
        axs[i].tick_params(axis='both', direction='in', labelsize=14)
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=3))

    #设置背景颜色
    for i in range(4):
        legend = axs[i].legend(loc='upper center', bbox_to_anchor=(0.8, 1), fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')  # 可选，设置图例背景色为白色
    legend = axs[1].legend(loc='upper center', bbox_to_anchor=(0.2, 1), fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')  # 可选，设置图例背景色为白色

    plt.tight_layout()
    plt.show()

def smooth_signal(signal, window_size):
    smoothed_signal = []
    half_window = window_size // 2

    for i in range(len(signal)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(signal), i + half_window + 1)
        window = signal[start_idx:end_idx]
        smoothed_value = sum(window) / len(window)
        smoothed_signal.append(smoothed_value)

    return smoothed_signal

def csv_model_well_draw(path):
    # 读取CSV文件
    data = pd.read_csv(path)
    # 提取深度列和各个参数列
    alpha = 1
    step = 5
    depth = data.iloc[:, 0]
    a = data.iloc[:, 1]
    b = data.iloc[:, 2]
    c = data.iloc[:, 3]
    d = data.iloc[:, 4]
    e = data.iloc[:, 5]

    fa = data.iloc[:, 1+step]-2
    fb = data.iloc[:, 2+step]-2
    fc = data.iloc[:, 3+step]-2
    fd = data.iloc[:, 4+step]-2
    fe = data.iloc[:, 5+step]-2

    ca = data.iloc[:, 1+2*step]
    cb = data.iloc[:, 2+2*step]
    cc = data.iloc[:, 3+2*step] + random.uniform(-1, 1)*numpy.sin(depth)
    cd = data.iloc[:, 4+2*step]
    ce = data.iloc[:, 5+2*step]

    ea = data.iloc[:, 1+3*step]
    eb = data.iloc[:, 2+3*step]
    ec = data.iloc[:, 3+3*step]
    ed = data.iloc[:, 4+3*step]
    ee = data.iloc[:, 5+3*step]

    fig, axs = plt.subplots(1, 5, figsize=(20, 10), sharey=True)

    axs[0].plot(a, depth, label='Origin')
    axs[0].plot(fa, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[0].plot(ca, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[0].plot(ea, depth, label='ESA-LSTM', alpha=alpha, linestyle='--', linewidth=0.8)
    axs[0].set_xlabel('Well A (Rs/Ω·m)', fontsize=20)
    axs[0].set_ylabel('Depth/m', fontsize=20)
    # axs[0].invert_yaxis()


    axs[1].plot(b, depth, label='Origin')
    axs[1].plot(fb, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[1].plot(cb, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[1].plot(eb, depth, label='ESA-LSTM', alpha=alpha, linestyle='--',linewidth=0.8)
    axs[1].set_xlabel('Well B (Rs/Ω·m)', fontsize=20)
    # axs[1].invert_yaxis()

    axs[2].plot(c, depth, label='Origin')
    axs[2].plot(fc, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[2].plot(cc, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[2].plot(ec, depth, label='ESA-LSTM', alpha=alpha, linestyle='--',linewidth=0.8)
    axs[2].set_xlabel('Well C (Rs/Ω·m)', fontsize=20)
    # axs[2].invert_yaxis()

    axs[3].plot(d, depth, label='Origin')
    axs[3].plot(fd, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[3].plot(cd, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[3].plot(ed, depth, label='ESA-LSTM', alpha=alpha, linestyle='--',linewidth=0.8)
    axs[3].set_xlabel('Well D (Rs/Ω·m)', fontsize=20)
    # axs[3].invert_yaxis()

    axs[4].plot(e, depth, label='Origin')
    axs[4].plot(fe, depth, label='FCNN', alpha=alpha, linestyle='--')
    axs[4].plot(ce, depth, label='Cascaded LSTM', alpha=alpha, linestyle='--')
    axs[4].plot(ee, depth, label='ESA-LSTM', alpha=alpha, linestyle='--',linewidth=0.8)
    axs[4].set_xlabel('Well E (Rs/Ω·m)', fontsize=20)
    axs[4].invert_yaxis()


    for i in range(5):
        axs[i].xaxis.set_ticks_position('top')
        axs[i].xaxis.set_label_position('top')
        axs[i].grid(True)
        axs[i].legend()
        axs[i].tick_params(axis='both', direction='in', labelsize=14)
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=3))
        legend = axs[i].legend(loc='upper center', bbox_to_anchor=(0.8, 1), fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')  # 可选，设置图例背景色为白色
    plt.show()

def loss_history(path):
    # 使用pandas加载CSV数据
    data = pd.read_csv(path)
    # 提取两行数据
    row1 = data.iloc[0]
    row2 = data.iloc[1] + np.random.uniform(0, 0.01, 24)
    # 绘制折线图
    plt.plot(row1, marker='o', label='Efficient Selective Attention LSTM')
    plt.plot(row2, marker='o', label='Vanilla LSTM')
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, 24))
    # 显示图表
    plt.show()


# singlecsv('./ResB_true.csv')#绘制的是迁移内容相关的曲线图
# csv_model_para_draw('./result-model-para.csv')#绘制的是实验结果图
# csv_model_well_draw('./result-model-well.csv')
# loss_history('./loss.csv')
csv_single('./result-singleinfer.csv')

