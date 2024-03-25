import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
csv_file = "../dataset/litho_map.csv"
def core_draw(csv_file):
    data_list = []
    with open(csv_file, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data_list.append(row)
    data = np.array(data_list, dtype=float)
    # Y = data[1035264:, -1]
    # X = data[1035264:, 0:-1]
    depth = data[1035264:, 0]
    core = data[1035264:, -1]
    # 岩性序列
    rock_sequence = core
    # 颜色映射
    color_map = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'yellow',
        4: 'purple',
        5: 'orange',
        6: 'pink',
        7: 'brown',
        8: 'gray',
        9: 'cyan',
        10: 'magenta',
        11: 'lime',
        12: 'black'
    }
    fig, ax = plt.subplots()

    for i in range(len(rock_sequence)):
        ax.barh(depth[i], 1, color=color_map[rock_sequence[i]])
    ax.invert_yaxis()
    # 设置标题和标签
    ax.set_title('岩心柱状图')
    ax.set_xlabel('岩性')
    ax.set_ylabel('深度')
    # 显示图形
    plt.show()
core_draw(csv_file)
