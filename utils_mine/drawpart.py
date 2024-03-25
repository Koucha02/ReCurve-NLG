import pandas as pd
import matplotlib.pyplot as plt
# 用来画出曲线的一部分
# 从CSV文件加载数据
file_path = "../dataset/litho_map.csv"  # 将文件路径替换为你的CSV文件路径
data = pd.read_csv(file_path)
# 获取第一列的列名作为自变量
x_column = data.columns[0]
# 获取所有列名作为因变量的选项
y_columns = data.columns
# 移除自变量列名，因为它不应该在因变量选项中
y_columns = y_columns.drop(x_column)
# 打印因变量选项，以供用户选择
print("请选择一个因变量:")
for i, y_col in enumerate(y_columns):
    print(f"{i + 1}: {y_col}")
# 用户选择因变量
selected_index = int(input("请输入选定的因变量的序号: ")) - 1
selected_y_column = y_columns[selected_index]
# 用户选择要绘制的区间
start_index = int(input("请输入要绘制的起始索引: "))
end_index = int(input("请输入要绘制的结束索引: "))
# 从数据中提取自变量和因变量的数据
x_data = data[x_column][start_index:end_index]
y_data = data[selected_y_column][start_index:end_index]
# 创建绘图
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, marker='o', linestyle='-')
plt.xlabel(x_column)
plt.ylabel(selected_y_column)
plt.title(f"{selected_y_column} vs. {x_column}")
# plt.grid(True)
plt.show()
