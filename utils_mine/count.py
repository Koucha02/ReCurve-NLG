import pandas as pd
# 用来统计标号的数量和每个标号对应的样本数量
# 读取CSV文件
file_path = '../dataset/litho_map.csv'  # 将 'your_file.csv' 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 获取最后一列数据
last_column = df.iloc[:, -1]

# 使用 value_counts() 统计不同数字的出现次数
counts = last_column.value_counts()

# 打印结果
print("不同数字及其出现次数：")
for value, count in counts.items():
    print(f"{value}: {count}")
