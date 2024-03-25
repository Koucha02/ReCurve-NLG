import os
import csv

# 创建一个用于存放切分后CSV文件的文件夹
output_folder = '../newlabel'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开原始CSV文件
with open('../dataset/litho_map.csv', 'r') as input_file:
    reader = csv.reader(input_file)
    next(reader)  # 跳过标题行

    current_label = None
    current_rows = []
    current_row_count = 0
    file_count = 1

    for row in reader:
        label = row[-1]  # 获取当前行的标号
        current_rows.append(row)  # 添加当前行到当前批次

        if current_label is None:
            current_label = label
        elif current_label != label or current_row_count >= 20:
            # 当标号变化或长度达到20时，切分当前批次，但保留最后一行到下一批次
            output_filename = os.path.join(output_folder, f'{file_count}.csv')
            with open(output_filename, 'w', newline='') as output_file:
                writer = csv.writer(output_file)
                writer.writerows(current_rows[:-1])  # 写入除了最后一行的数据
                current_rows = current_rows[-1:]  # 保留最后一行
                current_row_count = 1  # 重置行计数
                current_label = label
                file_count += 1
        else:
            current_row_count += 1

    # 处理剩余的行
    if current_rows:
        output_filename = os.path.join(output_folder, f'{file_count}.csv')
        with open(output_filename, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(current_rows)

print("CSV文件切分完成！")
