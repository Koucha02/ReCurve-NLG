# import pandas as pd
# # 读取 CSV 文件，指定原始编码（假设原始编码是 'gbk'）
# df = pd.read_csv('../dataset/merged.csv', encoding='gbk')
# # 将数据重新保存为 UTF-8 编码的 CSV 文件
# df.to_csv('../dataset/merged1.csv', encoding='utf-8', index=False)
import chardet

with open('../dataset/merged.csv', 'rb') as file:
    result = chardet.detect(file.read())

detected_encoding = result['encoding']
confidence = result['confidence']

print(f"Detected encoding: {detected_encoding} with confidence: {confidence}")

