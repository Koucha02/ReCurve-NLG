#该程序用于把所有空位填0
import os
import csv

# 目标文件夹的路径
folder_path = '../dataset/paras/'

# 获取文件夹中的所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 遍历每个CSV文件
for csv_file in csv_files:
    # 构建CSV文件的完整路径
    csv_file_path = os.path.join(folder_path, csv_file)

    # 读取CSV文件
    with open(csv_file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)

    # 找到每列的最大长度
    max_column_length = max(len(row) for row in rows)

    # 填充短列并写入新的CSV文件
    with open(csv_file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for row in rows:
            # 如果列长度不足，用0填充
            while len(row) < max_column_length:
                row.append('0')
            csv_writer.writerow(row)

print("处理完成")
