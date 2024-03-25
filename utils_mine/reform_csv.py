#该程序用来删除.csv中的指定行和指定列,以及更改每个csv中的参数title
import os
import csv
import pandas as pd
import shutil
#
# # 指定包含.csv文件的文件夹路径
# folder_path = "../dataset/paras/"
#
# # 获取文件夹中所有.csv文件的文件名
# csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
#
# # 遍历每个.csv文件并删除第一行
# for csv_file in csv_files:
#     file_path = os.path.join(folder_path, csv_file)
#
#     # 使用二进制模式打开文件，确保不会因为编码问题导致乱码
#     with open(file_path, "rb") as input_file:
#         lines = input_file.readlines()
#
#     # 删除第一行
#     lines.pop(0)
#
#     # 使用二进制模式重新写入文件
#     with open(file_path, "wb") as output_file:
#         output_file.writelines(lines)
#
# print("完成！所有.csv文件的第一行已删除。")

#设置目标文件夹路径和要删除的列号（从0开始）
# input_folder = '../dataset/paras/'
# output_folder = '../dataset/temp/'
# columns_to_delete = [4,5,7,9,10,11,13,15,16,18,19,20,
#                      22,23,24,25,26,27,28,29,30,31]
# # columns_to_delete = [10,11,12,13,14,15,16,17,18,19]
# # 要删除的列索引列表，这里假设要删除第1列和第3列
# # columns_to_delete = [0, 2]  # 请根据实际需求修改
#
# # 遍历输入文件夹中的所有.csv文件
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv'):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
#
#         # 打开输入文件和输出文件
#         with open(input_path, 'r', newline='') as input_file, open(output_path, 'w', newline='') as output_file:
#             csv_reader = csv.reader(input_file)
#             csv_writer = csv.writer(output_file)
#
#             for row in csv_reader:
#                 # 删除指定列
#                 new_row = [row[i] for i in range(len(row)) if i not in columns_to_delete]
#                 csv_writer.writerow(new_row)
#
#         print(f"处理文件 {filename} 完成")
#
# print("所有文件处理完成")

#
# input_folder = '../dataset/paras/'
#
# # 新的列名列表
# new_column_names = ['depth', 'den', 'res', 'ng', 'cal', 'st', 'ct', 'SP', 'FD3019g', 'dg','litho',]
#
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv'):
#         input_path = os.path.join(input_folder, filename)
#
#         # 打开输入文件并读取数据
#         with open(input_path, 'r', newline='') as input_file:
#             csv_reader = csv.reader(input_file)
#             rows = list(csv_reader)
#
#         # 更新第一行的列名
#         if len(rows) > 0:
#             rows[0] = new_column_names
#
#         # 再次打开文件并写入更新后的数据
#         with open(input_path, 'w', newline='') as output_file:
#             csv_writer = csv.writer(output_file)
#             csv_writer.writerows(rows)
#
#         print(f"处理文件 {filename} 完成")
#
# print("所有文件处理完成")






