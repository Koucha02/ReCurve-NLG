import os
import csv
# 用来删除最后一列（标号）
# 指定包含CSV文件的文件夹路径
folder_path = '../newlabel'

# 获取文件夹中所有CSV文件的文件名
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 遍历每个CSV文件
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)

    # 创建一个临时文件来保存修改后的数据
    temp_file_path = os.path.join(folder_path, 'temp.csv')

    with open(file_path, 'r', newline='') as input_file, open(temp_file_path, 'w', newline='') as output_file:
        csv_reader = csv.reader(input_file)
        csv_writer = csv.writer(output_file)

        # 遍历每一行，删除最后一列
        for row in csv_reader:
            if row:
                del row[-1]
                csv_writer.writerow(row)

    # 替换原文件
    os.remove(file_path)
    os.rename(temp_file_path, file_path)
