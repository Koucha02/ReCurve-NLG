import csv
import chardet

# 输入文件名和输出文件名
input_file = "../dataset/merged.csv"
output_file = "merge_out.csv"

# 检测文件编码
with open(input_file, "rb") as rawdata:
    result = chardet.detect(rawdata.read())
encoding = result['encoding']

# 打开输入文件以及输出文件，并指定编码
with open(input_file, "r", encoding=encoding, errors="replace") as infile, open(output_file, "w",
                                                                                newline="") as outfile:
    # 创建CSV读取器和写入器
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 遍历输入文件的每一行
    for row in reader:
        # 使用列表推导式检查每个单元格是否为数值
        is_numeric = all(cell.replace(".", "", 1).isdigit() for cell in row)

        # 如果整行都是数值，将其写入输出文件
        if is_numeric:
            writer.writerow(row)

# 提示处理完成
print("处理完成。已删除不包含数值的行，并将结果保存到 '{}' 文件中。".format(output_file))
