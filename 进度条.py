# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:48:42 2023

@author: chengyi.luo
"""


#%%
# 比较数据差别

import pandas as pd

# 读取两个 CSV 文件为 DataFrame
file_path_python = './Results/temp_python.csv'
file_path_ddb = './Results/temp_ddb.csv'

df_python = pd.read_csv(file_path_python)
df_ddb = pd.read_csv(file_path_ddb)

# 删除两列
df_ddb = df_ddb.drop(['__index_level_0__', 'data___index_level_0__'], axis=1)


# 比较两个 DataFrame 是否完全相等
are_equal = df_python.equals(df_ddb)

if are_equal:
    print("两个表的数值完全相等")
else:
    print("两个表的数值不完全相等")


# 比较两个 DataFrame，找出不相等的位置
diff_locations = df_ddb.compare(df_python, align_axis=1)

# 输出不相等的位置
print("不相等的位置：")
print(diff_locations)


# 单独查列名的位置
comparison_result = df_python['S_INFO_COMPIND_NAME1'] == df_ddb['S_INFO_COMPIND_NAME1']  # 两个都是nan时，结果会是false

print(comparison_result)
print(sum(comparison_result))

false_indices = comparison_result[comparison_result == False].index


df_ddb1 = df_ddb.loc[false_indices]
df_python1 = df_python.loc[false_indices]



#%%

# -*- coding=utf-8 -*-
 
import time
from progressbar import *
 
total = 1000
 
def dosomework():
    time.sleep(0.01)
 
progress = ProgressBar()
for i in progress(range(1000)):
    dosomework()


# %%

from alive_progress.styles import showtime

showtime()
# %%
