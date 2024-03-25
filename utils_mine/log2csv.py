#从log文件中读取绘图信息
import re
import csv

# 输入和输出文件的路径
input_file_path = '../log/model-lstm-4-mid.txt'
output_file_path = '../log/model-lstm-mid.csv'

# 打开输入文件和输出CSV文件
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
    # 创建CSV写入器
    csv_writer = csv.writer(output_file)

    # 写入CSV文件的标题行
    csv_writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])

    # 逐行读取输入文件
    for line in input_file:
        # 使用正则表达式来提取所需信息
        match = re.search(r'Epoch \[(\d+)/\d+\] - Train Loss: ([\d.]+), Test Loss: ([\d.]+), Test Accuracy: ([\d.]+)%',
                          line)
        if match:
            epoch = match.group(1)
            train_loss = match.group(2)
            test_loss = match.group(3)
            test_accuracy = match.group(4)

            # 将提取的信息写入CSV文件
            csv_writer.writerow([epoch, train_loss, test_loss, test_accuracy])

print(f"提取并保存成功至 {output_file_path}")
