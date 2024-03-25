import torch
import numpy as np
import matplotlib.pyplot as plt

# 加载你的.pth模型
model_path = r'E:\PycharmPrograms\Recurve\models\at30-model.pth'
model_dict = torch.load(model_path, map_location=torch.device('cpu'))

# 保存权重和偏差到txt文件
txt_file_path = './parameters.txt'

with open(txt_file_path, 'w') as f:
    f.write(r"Layer\tParameter\tShape\tValues\n")

    for name, param in model_dict.items():
        if 'weight' in name or 'bias' in name:
            values = param.cpu().numpy().flatten()
            values_str = '\t'.join(map(str, values))
            f.write(f"{name}\t{len(values)}\t{values_str}\n")

print(f"Parameters saved to {txt_file_path}")

# 可视化权重和偏差的分布
for name, param in model_dict.items():
    if 'weight' in name or 'bias' in name:
        values = param.cpu().numpy().flatten()

        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=50, color='blue', alpha=0.7)
        plt.title(f'{name} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
