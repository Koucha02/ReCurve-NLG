import os
from matplotlib import pyplot as plt
import lasio
import csv
import torch

"""该程序用于从.las文件中提取数据并写入csv"""
"""
    密度管参数：
    1Depth 深度 2DevV偏差电压 3Cal井径 4DevI偏差电流 5Ng天然伽马 6Lgg长距离Gamma射线 7Sgg短距离Gamma射线
    8Ps电阻率 9Cond电导率

    声波管参数：
    1Depth深度 2TD2第二次声波时效 3T1第一次声波时效

    组合管参数：
    1Depth深度 2SP自然电位 3ResP 16'正常电阻率 4SPR单点电阻率 5NGamma自然伽马总计数 6Tip（Tilt）倾斜 7Azimuth方位角

    温度管参数：1Depth深度 2Temp温度 3F.Ps液体电阻率
"""


i = 1
filePath = r'./Data/MD/'
files = os.listdir(filePath)
# 用来把150个MD数据的其中一列归到一个csv里
with open('data-md-5.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    for fileName in files:
        las = lasio.read(filePath + fileName)
        if 'md' in fileName:
            fileName = fileName.replace('md', 'MD')
        print(fileName)
        z = las.data[2000:5000, 5].astype('float32')
        writer.writerow(z)




# print(type(las.data))





#以下是lasio的常用方法
# 以HeadItem的方式显示曲线文件头
# print(las.version)
# # 以CurveItem的方式显示曲线道头
# print(las.curves)
# # 显示曲线道的名称
# print(las.keys())
# # 显示las.data的数据类型
# print(type(las.data))
# # 显示测井数据体的形状
# print(las.data.shape)
# # 显示测井数据道的数据类型
# print(type(las[1]))

# 以下都是画图部分
#fig = plt.figure()
# trackNum = las.data.shape[1]

# for i in np.arange(1, trackNum):
#     ax = fig.add_subplot(1, trackNum - 1, 5)
#     ax.plot(las.data[:, 1],las.index)
#     ax.set_xlabel(las.keys()[i])
#     ax.xaxis.tick_top()
#     ax.invert_yaxis()

"""
draw_index = las.index[4000:6000]

for i in range(1, trackNum):
    ax = fig.add_subplot(1, trackNum - 1, i)
    ax.plot(las.data[4000:6000, i], draw_index)

    ax.set_xlabel(las.keys()[i])
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])

plt.show()
"""
