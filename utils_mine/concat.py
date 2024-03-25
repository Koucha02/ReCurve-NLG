#该程序用来把岩性代码附加到参数列表的最后一列
import os
import csv

# 指定A和B文件夹的路径
folder_a = '../dataset/paras/'
folder_b = '../dataset/litho/'

csv_files_b = [f for f in os.listdir(folder_b) if f.endswith('.csv')]
for csv_file_b in csv_files_b:
    path_b = os.path.join(folder_b, csv_file_b)
    csv_file_a = os.path.join(folder_a, csv_file_b)
    with open(path_b, 'r', newline='') as file_b:
        csv_reader = csv.reader(file_b)
        rows_b = list(csv_reader)
    with open(csv_file_a, 'r+', newline='') as file_a:
        csv_reader_a = csv.reader(file_a)
        rows_a = list(csv_reader_a)

        for i, row_b in enumerate(rows_b):
            if len(row_b) > 1:
                last_column_b = row_b[1]

                if i < len(rows_a):
                    rows_a[i].append(last_column_b)
                else:
                    new_row_a = [''] * (len(rows_b[0]) - 1) + [last_column_b]
                    rows_a.append(new_row_a)

        file_a.seek(0)
        csv_writer = csv.writer(file_a)
        csv_writer.writerows(rows_a)
        file_a.truncate()
