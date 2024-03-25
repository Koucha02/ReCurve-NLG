import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('../dataset/litho_map.csv')
# 提取深度列和各个参数列
alpha = 0.9
depth = data.iloc[1035264:, 0]
# den = data.iloc[:, 1]
res = data.iloc[1035264:, 2]
ng = data.iloc[1035264:, 3]
cal = data.iloc[1035264:, 4]
# st = data.iloc[:, 5]
# ct = data.iloc[:, 6]
# sp = data.iloc[:, 7]
# dg = data.iloc[:, 8]


fig, axs = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

axs[0].plot(res, depth, label='Res')
axs[0].set_xlabel('Res/Ω·m')
axs[0].set_ylabel('Depth/m')
axs[0].invert_yaxis()

# axs[1].plot(ng, depth, label='GR', alpha=alpha)
# axs[1].set_xlabel('Res/Ω·m')

axs[1].plot(ng, depth, label='Origin')
axs[1].set_xlabel('Ng/CPS')

axs[2].plot(cal, depth, label='Origin')
axs[2].set_xlabel('Cal/mm')

for i in range(3):
    axs[i].xaxis.set_ticks_position('top')
    axs[i].xaxis.set_label_position('top')
    axs[i].grid(True)
    # axs[i].legend()
plt.show()
