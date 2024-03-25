#用来从NLG原始资料中提取.roc和PREC文件并整理成多个.txt

import os
import shutil

source_folder = '../dataset/data/'
target_folder = '../dataset/rocs/'

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file == 'PREC':
        # if file == 'Normal.roc':
            source_file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            target_file_name = folder_name
            target_file_path = os.path.join(target_folder, target_file_name)
            shutil.copy(source_file_path, target_file_path)
