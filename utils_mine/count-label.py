import os
import csv

# 指定文件夹路径
folder_path = '../newlabel'  # 替换为你的文件夹路径

# 创建一个空字典，用于存储文件名（不包括扩展名）和对应的标号
file_label_dict = {}

# 遍历文件夹下的CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_name = os.path.splitext(filename)[0]  # 获取文件名（不包括扩展名）
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if row:  # 确保行不为空
                    label = row[-1]  # 获取最后一列的标号
                    file_label_dict[file_name] = label
                    break  # 只读取每个文件的第一行以获取标号

# 创建一个新的CSV文件，保存文件名和标号的对应关系
output_csv = '../file_label_mapping.csv'
with open(output_csv, 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(['Filename', 'Label'])  # 写入标题行
    for file_name, label in file_label_dict.items():
        writer.writerow([file_name, label])

print(f"文件名和标号的对应关系已保存在 {output_csv} 文件中。")
