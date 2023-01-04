import os
import shutil
import xlwt
import numpy as np
"用于抽取文件、抽取数据和更改后缀"
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
def ExtractData(file_in, file_out):
    files = os.listdir(file_in)
    for zkns in files:
        laspath = file_in + zkns + "/原始资料/"
        if os.path.isdir(laspath) == 0:
            laspath = file_in + zkns + "/原始数据/"
        lasdic = os.listdir(laspath)

        for lasfile in lasdic:
            shutil.move(laspath + lasfile, file_out)
            print(lasfile, "转移成功")

def update(fp, oldend, newend):
    # listdir：返回指定的文件夹包含的文件或文件夹的名字的列表
    files = os.listdir(fp)
    for file in files:
        fileName = fp + os.sep + file
        path1 = fp
        # 运用递归;isdir：判断某一路径是否为目录
        if os.path.isdir(fileName):
            update(fileName)
            continue
        else:
            if file.endswith(oldend):
                test = file.replace(oldend, newend)
                print("修改前:" + path1 + os.sep + file)
                print("修改后:" + path1 + os.sep + test)
                os.renames(path1 + os.sep + file, path1 + os.sep + test)

def txt2xlsx():
    fd = os.listdir(r'./OriginData')
    for filename in fd:
        bias = 140
        filedata = open("./OriginData/" + filename)
        num_line = 0
        new_workbook = xlwt.Workbook()
        sheet = new_workbook.add_sheet('写数据')
        for line in filedata:
            num_line += 1
            if num_line > bias:  # 140行对应到10m开始的数据；40行对应的是0m
                line_list = line.split()  # 把单行内容分割成多个字符串
                for col in range(0, len(line_list)):
                    sheet.write(num_line-bias, col, line_list[col])
        new_workbook.save("./Data/" + filename + ".xlsx")

if __name__ == '__main__':
    filepath_in = r"D:/Pychram_Programs/Graduate/纳岭沟物探工作/归档材料-地研院/纳岭沟物探资料/"
    filepath_out = r"D:/Pychram_Programs/Graduate/OriginData/"
    ExtractData(filepath_in, filepath_out)
    update(fp='./OriginData', oldend='las', newend='.las')
    # txt2xlsx()
    # update(fp='./Data', oldend='.txt.xlsx', newend='.xlsx')


