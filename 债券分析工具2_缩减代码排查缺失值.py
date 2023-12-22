# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:31:03 2023

@author: chengyi.luo
"""


import os 
os.getcwd()
os.chdir(r'D:\Jupyter\spread')  


import sys
# sys.path.insert(0, '/home/gfzg/workspace/数据下载/Wind/Packages')
# sys.path.insert(0, '/home/gfzg/workspace/债券分析/Packages')
import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import warnings
from importlib import reload
import re
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm
import math
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import janitor   
# 需要找到C:\Users\lenovo\anaconda3\Lib\site-packages路径下的janitor.py 
# 把ConfigParser改成configparse
# from werkzeug.contrib.fixers import ProxyFix 改成 from werkzeug.middleware.proxy_fix import ProxyFix
import time
from progressbar import progressbar as pbar # 该用pg进度条了
from tabulate import tabulate
# 核心包
# import Wind
import Bonder
import Selector
import dolphindb as ddb

import matplotlib as mpl
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


# 执行筛选
def select(data,Filters):
    if Filters == None: return data
    def compare(Series,value,symbol='=='):
        if symbol == '==':
            return (Series==value).values
        elif symbol == '!=':
            return (Series!=value).values
        elif symbol == '>=':
            return (Series>=value).values
        elif symbol == '<=':
            return (Series<=value).values
        elif symbol == '>':
            return (Series>value).values
        elif symbol == '<':
            return (Series<value).values
        elif symbol == 'isin':
            return (Series.isin(value)).values
        elif symbol == 'isna':
            return (Series.isna()).values
        elif symbol == 'isnotna':
            return (~Series.isna()).values
        else:
            print('请检查逻辑符号，有效逻辑符号为["==","!=",">=","<=",">","<","isin","isna","isnotna"]')
            return np.array([False]*len(Series))        

    
    print('数据筛选中...请等待')
    final_selection = np.array([True] * data.shape[0])
    for filter in Filters:
        if type(filter) is tuple:
            newselection = compare(data[filter[0]],filter[2],symbol=filter[1])
        elif type(filter) is list:
            newselection = np.array([False] * data.shape[0])
            for fil in filter:
                selection = compare(data[fil[0]],fil[2],symbol=fil[1])
                newselection = newselection | selection
        final_selection = final_selection & newselection  # 关键在这里用了一个 与 如果都为1被筛选
    ret = data.loc[final_selection]
    print('筛选结果数据量：', ret.shape[0])
    return ret

# 显示筛选条件   （这块比较独立）
def slectionString(Filters):
    
    def parse_tuple_condition(x):
        x = [str(i) for i in list(x)]
        return ' '.join(x)
    
    condition = ''
    for filter in Filters:
        if condition != '': condition+=' and '
        if type(filter) is tuple:  #and
            string = parse_tuple_condition(filter)
            condition +=  string
        elif type(filter) is list:  #or
            strings = ''
            for fil in filter:
                string = parse_tuple_condition(fil)
                if strings == '': strings = string
                else: strings +=  (' or '+ string)
            condition += '(' + strings + ')'
    return condition



def dataCenter(config={}):

    # 静态筛选
    STATIC = pd.read_parquet(config['StaticPath'])
    STATIC.drop('B_INFO_ISSUERCODE',inplace=True,axis=1)
    STATIC = Selector.select(STATIC,config['BondStaticFilters'])
    print('STATIC:', slectionString(config['BondStaticFilters']))

    # 动态筛选
    BONDS = pd.read_parquet(config['BondsPath'])
    BONDS = Selector.select(BONDS,config['BondDynamicFilters'])
    print('BONDS:', slectionString(config['BondDynamicFilters']))

    # 动态静态结合
    df = pd.merge(STATIC, BONDS, on='S_INFO_WINDCODE',validate='one_to_many')  

    # 信用计算利差
    print('\n信用计算利差中')
    FILLED_GUOKAI = pd.read_parquet(config['FilledGuokaiPath'])
    df['B_ANAL_MATU_CNBD'] = df['B_ANAL_MATU_CNBD'].round(2)
    df = pd.merge(df,FILLED_GUOKAI,on=['TRADE_DT','B_ANAL_MATU_CNBD'],how='left')
    df['CREDIT_SPREAD'] = df['B_ANAL_YIELD_CNBD'] - df['B_ANAL_YIELD_GUOKAI']

    print('数据处理完成')
    return df


#%%


# 一、三表合并

# 债券的静态信息
STATIC = pd.read_parquet('./Results/STATIC.parquet')     # 初始行数21930
STATIC.drop('B_INFO_ISSUERCODE', inplace=True, axis=1)   # 删掉B_INFO_ISSUERCODE列
BondStaticFilters = [
    ('suffix','isin',['IB','SH','SZ']),     # 默认选择标准代码  21722    
    ('CATEGORY1','==','信用债'),            # 默认选择信用债，不含利率债、同业存单、ABS、可转债  14673
    ('isPerpetualBonds','==',0),            # 剔除：永续债    14568
    ('IS_FAILURE','==',0)                   # 剔除：发行失败  14039
]              
    # ('B_INFO_GUARTYPE','isna','')       # 剔除：附带担保债    全部都是non-null，这个条件直接变0
STATIC = select(data = STATIC, Filters = BondStaticFilters)
print('STATIC:', slectionString(BondStaticFilters))
STATIC

# 债券信息
BONDS = pd.read_parquet('./Results/BONDS.parquet')
BondDynamicFilters = [('B_ANAL_MATU_CNBD','<=', 10)]
BONDS = select(BONDS,BondDynamicFilters)
print('BONDS:', slectionString(BondDynamicFilters))   # 14837981 => 11014822
BONDS

# 国开数据
FILLED_GUOKAI = pd.read_parquet('./Results/FILLED_GUOKAI.parquet')
# FILLED_GUOKAI.to_csv('./Results/FILLED_GUOKAI.csv',encoding='gbk')
FILLED_GUOKAI


# 动态静态结合
df = pd.merge(STATIC, BONDS, on='S_INFO_WINDCODE', validate='one_to_many')  


# 计算信用利差
df['B_ANAL_MATU_CNBD'] = df['B_ANAL_MATU_CNBD'].round(2)  # 保留两位小数
df = pd.merge(df,FILLED_GUOKAI,on=['TRADE_DT','B_ANAL_MATU_CNBD'],how='left')  # 行数没变
df['CREDIT_SPREAD'] = df['B_ANAL_YIELD_CNBD'] - df['B_ANAL_YIELD_GUOKAI']

#%%

df.to_csv('./Results/temp_python.csv', index=False, encoding='utf-8-sig')
print(df.S_INFO_COMPIND_NAME1.isnull().sum())

df_python = pd.read_csv('./Results/temp_python.csv')
print(df_python.S_INFO_COMPIND_NAME1.isnull().sum())

#%%

# 二、绘制走势图和条形图


division_label = 'S_INFO_COMPIND_NAME1'    # 根据这个字段的类别画线  债券主体公司所属Wind一级行业名称
Filters = [('BOND_TYPE','==','产业债'),]    # 对债券数据进行筛选
CrossSectionConfig = {
    'date':'20230921', # 截面日期
    'plotCrossSection':True, # 是否绘制date时间的柱状截面图   横截面（cross-section）
}

# 参数
divisionLabel = division_label
filters = Filters
plotCrossSection = CrossSectionConfig['plotCrossSection']
methods = [0, 1, 2]
config = CrossSectionConfig


data = copy.deepcopy(df)   # 原本1050227
# data.dropna(subset=['S_INFO_COMPIND_NAME1'],inplace=True)   # 变成1047733
# data.S_INFO_COMPIND_NAME1.info()


if not divisionLabel:   # 如果为空，新增label列都为ALL
    data['LABEL'] = 'ALL'
else:    # 如果divisionLabel非空，新增一列label为division_label列
    data['LABEL'] = data[divisionLabel]
data.dropna(subset=['LABEL'],inplace=True)   # 变成1047733

methodDict = {
    0: '简单平均',
    1: '中位数',
    2: '余额加权平均',
}

data = select(data,filters)  # 只剩100630
data = data.sort_values(['TRADE_DT','LABEL'])
index = data.set_index(['TRADE_DT','LABEL']).index.unique()
dailyspread = pd.DataFrame(index = index)   # 只有索引的数据框


labels = data['LABEL'].unique()  # array(['公用事业', '可选消费', '工业', '房地产', '日常消费', '材料', '能源', '金融']
lengths = (len(methods),len(labels))   # (1, 8)
mus = np.linspace(100,2000,lengths[0]*lengths[1])  # 100-2000的等间隔数据  1*8总共8个数

# 设置颜色
cmap = plt.cm.gist_ncar # 一个颜色条colormap
palette_raw = dict()
palette = dict()
palette_reverse = dict()
for i in range(lengths[0]):
    method = methods[i]
    for j in range(lengths[1]):
        label = labels[j]
        mu = mus[i*lengths[1]+j]
        color = cmap(1-mu/np.max(mus))  
        hue = methodDict[method]+'-'+str(label)
        palette[hue] = color
        palette_raw[label] = color
        palette_reverse[color] = hue

# 计算信用利差
if 0 in methods:
    dailyspread[f'CREDIT_SPREAD_0'] = data.groupby(['TRADE_DT','LABEL'])['CREDIT_SPREAD'].mean() *100 #BP basic point 基点
if 1 in methods:
    dailyspread[f'CREDIT_SPREAD_1'] = data.groupby(['TRADE_DT','LABEL'])['CREDIT_SPREAD'].median() *100 #BP
if 2 in methods:
    total = data.groupby(['TRADE_DT','LABEL'])['B_ANAL_RESIDUALPRI'].sum().to_frame('B_ANAL_RESIDUALPRI_TOTAL').reset_index()
    data = pd.merge(data, total, how='left', on=['TRADE_DT','LABEL'])
    data['CREDIT_SPREAD'] = data['B_ANAL_RESIDUALPRI']/data['B_ANAL_RESIDUALPRI_TOTAL']*data['CREDIT_SPREAD']
    dailyspread[f'CREDIT_SPREAD_2'] = data.groupby(['TRADE_DT','LABEL'])['CREDIT_SPREAD'].sum() *100 #BP

    
dailyspread = dailyspread.reset_index()
dailyspread['TRADE_DT_DATETIME'] = pd.to_datetime(dailyspread['TRADE_DT'])


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,5), dpi=100)
for i in methods:
    sns.lineplot(data=dailyspread, x="TRADE_DT_DATETIME", y=f"CREDIT_SPREAD_{i}",
                 hue=methodDict[i]+'-'+dailyspread["LABEL"].astype(str),linewidth = 2, palette=palette)
conditionString = slectionString(filters)  # 'BOND_TYPE == 产业债'
ax.set_xlabel('日期')
ax.set_ylabel('信用利差(Bp)')
ax.set_title(f"信用利差（中债个债估值收益率 - 基准收益率）\n 筛选条件：{conditionString if conditionString else '不筛选'}")
ax.tick_params(axis='x', rotation=90)


# “Bp”通常是“基点（Basis Point）”的缩写。在金融领域，基点表示利率或者收益率的一种微小变动单位，相当于0.01%。
# 因此，当提到“信用利差（Bp）”时，它指的是信用利差以基点为单位的数值。
# 例如，如果某个债券的利率是5%，而无风险利率是3%，那么其信用利差就是200个基点（5% - 3% = 200 Bp）

for i in range(len(ax.get_xticklabels())):  # 获取一个图表（假设是用 Matplotlib 库创建的）的 x 轴刻度标签
    tick = ax.get_xticklabels()[i]   # 但是这个循环不是只能拿到最后一个吗？
    
if plotCrossSection:  # if true 就画图
    bardata = dailyspread.loc[dailyspread['TRADE_DT'] == '20230821']   # 原本是config['date'] 202309 18192021这最后四天都是0
    fig, barax = plt.subplots(ncols=1, nrows=1, figsize=(10,5), dpi=100)
    order = bardata.sort_values([f"CREDIT_SPREAD_{methods[0]}"],ascending=False)['LABEL'].values  # 顺序 降序排序
    sns.barplot(data=bardata, x='LABEL', y=f"CREDIT_SPREAD_{methods[0]}",ci=False, ax=barax, order=order,palette=palette_raw)
    barax.set_xlabel(divisionLabel)
    barax.set_ylabel(f'信用利差 {methodDict[methods[0]]}')   
    barax.set_title(f"{config['date']}日截面 信用利差（中债个债估值收益率 - 基准收益率）\n 筛选条件：{conditionString if conditionString else '不筛选'}")
    barax.tick_params(axis='x', rotation=90)
    for i in range(len(barax.get_xticklabels())):
        tick = barax.get_xticklabels()[i]

# 第一个图的参数
ax.grid()
ax.legend(loc=0, ncol=1, bbox_to_anchor=[1.02,1,0,0])
plt.show() 

