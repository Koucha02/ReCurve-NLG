import csv

# 输入文件名和输出文件名
input_file = "../dataset/merged.csv"
output_file = "merge_out.csv"

# 打开输入文件以及输出文件
with open(input_file, "r", errors="replace") as infile, open(output_file, "w", newline="") as outfile:
    # 创建CSV读取器和写入器
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 遍历输入文件的每一行
    for row in reader:
        try:
            # 尝试将最后一列转换为数值
            last_column = float(row[-1])
            # 如果成功，将整行写入输出文件
            writer.writerow(row)
        except (ValueError, IndexError):
            # 如果转换失败（不是数值或索引错误），跳过该行
            pass

# 提示处理完成
print("处理完成。已将包含数值的行保存到 '{}' 文件中。".format(output_file))
