import pandas as pd
# 用来修改数据中的标号
file_path = '../dataset/litho.csv'  # 将 'your_file.csv' 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 定义要进行修改的映射关系
replacement_mapping = {
    10: 0,
    16: 1,
    17: 9,
    23: 10,
    36: 11
}

# 修改最后一列数据
df['last_column'] = df['last_column'].replace(replacement_mapping)

# 保存修改后的CSV文件
output_file_path = '../litho_mapping.csv'  # 保存修改后的文件路径
df.to_csv(output_file_path, index=False)

print("数据修改完成，并已保存到文件:", output_file_path)
