import os
import csv
#该程序用来清空岩性代码为0的行
# 指定输入文件夹的路径
input_folder = '../dataset/paras/'

csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
for file in csv_files:
    file_path = os.path.join(input_folder, file)
    temp_file_path = os.path.join(input_folder, 'temp_' + file)

    with open(file_path, 'r', newline='') as csvfile, open(temp_file_path, 'w', newline='') as temp_csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(temp_csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if row['litho'] != '0':
                writer.writerow(row)
    os.remove(file_path)
    os.rename(temp_file_path, file_path)
