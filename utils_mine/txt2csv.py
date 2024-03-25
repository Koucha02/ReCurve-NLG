import os
import csv
# 该程序用于将某个文件夹下所有(txt)文档写入同名的csv
source_folder = '../dataset/rocs/'
target_folder = '../dataset/paras/'

for root, dirs, files in os.walk(source_folder):
    for file in files:
        file_path = os.path.join(root, file)
        csv_file_path = os.path.join(target_folder, file.replace(file, file+'.csv'))
        print(file)
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            with open(file_path, 'r') as txt_file:
                for line in txt_file:
                    columns = line.strip().split()
                    csv_writer.writerow(columns)

            print(f'已创建CSV文件: {csv_file_path}')
