import os
import csv

# 指定包含CSV文件的文件夹路径
input_folder = './newlabel'
# 指定保存填充后CSV文件的文件夹路径
output_folder = './newlabels'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中所有CSV文件的文件名
csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

# 定义填充的行数
desired_row_count = 20

# 遍历每个CSV文件
for file_name in csv_files:
    input_file_path = os.path.join(input_folder, file_name)
    output_file_path = os.path.join(output_folder, file_name)

    with open(input_file_path, 'r', newline='') as input_file, open(output_file_path, 'w', newline='') as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)


        # 遍历每一行
        rows = [row for row in csv_reader]
        for _ in range(desired_row_count - len(rows)):
            rows.append(['0'] * len(rows[0]))

        # 写入填充后的数据
        for row in rows:
            csv_writer.writerow(row)

print("CSV文件已填充并保存到指定文件夹。")
