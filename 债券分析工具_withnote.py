# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:36:52 2023

@author: lenovo
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


#%% 先变成函数

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


# method: 0, 1, 2
# 0: simple average
# 1: median
# 2: 余额加权平均


def plotCreditSpread(data,filters,divisionLabel='',methods=[0],config={'plotCrossSection':False}):

    plotCrossSection = config['plotCrossSection']
    data = copy.deepcopy(data)
    
    if not divisionLabel:
        data['LABEL'] = 'ALL'
    else:
        data['LABEL'] = data[divisionLabel]
    data.dropna(subset=['LABEL'],inplace=True)
    
    methodDict = {
        0: '简单平均',
        1: '中位数',
        2: '余额加权平均',
    }
    data = select(data,filters)
    data = data.sort_values(['TRADE_DT','LABEL'])
    index = data.set_index(['TRADE_DT','LABEL']).index.unique()
    dailyspread = pd.DataFrame(index = index)

    labels = data['LABEL'].unique()
    lengths = (len(methods),len(labels))
    mus = np.linspace(100,2000,lengths[0]*lengths[1])
    cmap = plt.cm.gist_ncar
    palette_raw = dict()
    palette =dict()
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
    
    if 0 in methods:
        dailyspread[f'CREDIT_SPREAD_0'] = data.groupby(['TRADE_DT','LABEL'])['CREDIT_SPREAD'].mean() *100 #BP
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
        sns.lineplot(data=dailyspread, x="TRADE_DT_DATETIME", y=f"CREDIT_SPREAD_{i}",hue=methodDict[i]+'-'+dailyspread["LABEL"].astype(str),linewidth = 2, palette=palette)

    # i = methods[0]
    # if config['upperquantile'] and len(labels)==1:
    #     quantile = dailyspread[f"CREDIT_SPREAD_{i}"].rolling(config['upperquantile_period'],1).apply(lambda x: np.nanquantile(x.values,config['upperquantile_level'],interpolation='linear'))
    #     sns.lineplot(x=dailyspread['TRADE_DT_DATETIME'], y=quantile, ax = ax, label=f'过去{config["upperquantile_period"]}天 {int(config["upperquantile_level"]*100)} quantile',linewidth=2,color='r', linestyle='--')
    # if config['lowerquantile'] and len(labels)==1:
    #     quantile = dailyspread[f"CREDIT_SPREAD_{i}"].rolling(config['lowerquantile_period'],1).apply(lambda x: np.nanquantile(x.values,config['lowerquantile_level'],interpolation='linear'))
    #     sns.lineplot(x=dailyspread['TRADE_DT_DATETIME'], y=quantile, ax = ax, label=f'过去{config["lowerquantile_period"]}天 {int(config["lowerquantile_level"]*100)} quantile',linewidth=2,color='blue', linestyle='--')

    conditionString = slectionString(filters)
    ax.set_xlabel('日期')
    ax.set_ylabel('信用利差(Bp)')   
    ax.set_title(f"信用利差（中债个债估值收益率 - 基准收益率）\n 筛选条件：{conditionString if conditionString else '不筛选'}")
    ax.tick_params(axis='x', rotation=90)
    for i in range(len(ax.get_xticklabels())):
        tick = ax.get_xticklabels()[i]
    if plotCrossSection:
        bardata = dailyspread.loc[dailyspread['TRADE_DT'] == config['date']]
        fig, barax = plt.subplots(ncols=1, nrows=1, figsize=(10,5), dpi=100)
        order = bardata.sort_values([f"CREDIT_SPREAD_{methods[0]}"],ascending=False)['LABEL'].values
        sns.barplot(data=bardata, x='LABEL', y=f"CREDIT_SPREAD_{methods[0]}",ci=False, ax=barax, order=order,palette=palette_raw)
        barax.set_xlabel(divisionLabel)
        barax.set_ylabel(f'信用利差 {methodDict[methods[0]]}')   
        barax.set_title(f"{config['date']}日截面 信用利差（中债个债估值收益率 - 基准收益率）\n 筛选条件：{conditionString if conditionString else '不筛选'}")
        barax.tick_params(axis='x', rotation=90)
        for i in range(len(barax.get_xticklabels())):
            tick = barax.get_xticklabels()[i]

    ax.grid()
    ax.legend(loc=0, ncol=1, bbox_to_anchor=[1.02,1,0,0])
    plt.show() 


#%% 把类先拆成函数

## 一、三表合并
BondStaticFilters = [
    ('suffix','isin',['IB','SH','SZ']),     # 默认选择标准代码  21722    原本21930行
    ('CATEGORY1','==','信用债'),            # 默认选择信用债，不含利率债、同业存单、ABS、可转债  14673
    ('isPerpetualBonds','==',0),            # 剔除：永续债    14568
    ('IS_FAILURE','==',0)                   # 剔除：发行失败  14039
]    
BondDynamicFilters = [('B_ANAL_MATU_CNBD','<=', 10)]
dataConfig = {
    'StaticPath': './Results/STATIC.parquet',
    'BondsPath': './Results/BONDS.parquet', 
    'FilledGuokaiPath': './Results/FILLED_GUOKAI.parquet',
    'BondStaticFilters': BondStaticFilters,
    'BondDynamicFilters': BondDynamicFilters,
}
df = dataCenter(dataConfig)

## 二、画图

Filters = [('BOND_TYPE','==','产业债'),]
division_label = 'S_INFO_COMPIND_NAME1' # 根据这个字段的类别画线
CrossSectionConfig = {
    'date':'20230917', # 截面日期   原本是21
    'plotCrossSection':True, # 是否绘制date时间的柱状截面图
}
# methods 0: 简单平均 1: 中位数 2: 剩余本金加权
plotCreditSpread(
    data = df,
    filters = Filters,
    divisionLabel=division_label,
    methods=[2],
    config=CrossSectionConfig
)


#%%  函数拆分


# 一、三表合并

# 债券的静态信息
STATIC = pd.read_parquet('./Results/STATIC.parquet')   # 初始行数21930
# STATIC.to_csv('./Results/STATIC.csv', encoding='gbk')
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

condition = STATIC['suffix'].isin(['IB', 'SH', 'SZ'])
sum(condition)

# STATIC.B_INFO_GUARTYPE的分布
# STATIC.info()
# plt.figure(figsize=(8, 6))  # 设置图像大小
# values = STATIC.IS_SUB
# sns.boxplot(y=values)  # 绘制箱线图
# plt.show()


# 假设 filter = ('suffix','isin',['IB','SH','SZ'])
filter = ('suffix','isin',['IB','SH','SZ'])
filter_column, filter_method, filter_values = filter
filtered_data = STATIC[STATIC[filter_column].isin(filter_values)]
print(filtered_data)


# 筛选 IS_SUB 列为空值的行
# filtered_data = STATIC[STATIC['IS_SUB'].isnull()]
# print(filtered_data.shape[0])  # 输出筛选后的行数  对原来的表进行筛选，应该有12726行

# 债券信息
BONDS = pd.read_parquet('./Results/BONDS.parquet')
# BONDS.to_csv('./Results/BONDS.csv',encoding='gbk')
BondDynamicFilters = [('B_ANAL_MATU_CNBD','<=', 10)]
BONDS = select(BONDS,BondDynamicFilters)
print('BONDS:', slectionString(BondDynamicFilters))   # 14837981 => 11014822
BONDS


FILLED_GUOKAI = pd.read_parquet('./Results/FILLED_GUOKAI.parquet')
# FILLED_GUOKAI.to_csv('./Results/FILLED_GUOKAI.csv',encoding='gbk')
FILLED_GUOKAI



# 动态静态结合
df = pd.merge(STATIC, BONDS, on='S_INFO_WINDCODE', validate='one_to_many')  
# df2 = pd.merge(STATIC, BONDS, on='S_INFO_WINDCODE')    # 行数相同，validate='one_to_many'其实没有用上


# 计算信用利差
df['B_ANAL_MATU_CNBD'] = df['B_ANAL_MATU_CNBD'].round(2)  # 保留两位小数
df = pd.merge(df,FILLED_GUOKAI,on=['TRADE_DT','B_ANAL_MATU_CNBD'],how='left')  # 行数没变
df['CREDIT_SPREAD'] = df['B_ANAL_YIELD_CNBD'] - df['B_ANAL_YIELD_GUOKAI']




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
methods = [2]
config = CrossSectionConfig


data = copy.deepcopy(df)   #原本1050227

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
dailyspread = pd.DataFrame(index = index)   # 把索引变成数据框


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




#%%
## 四、个债预警

# 展示指定债券的图像
def showSingleBond(data, bond_codes=[], YIELD_CHANGE_PERIOD=7,QUANTILE=0.95,ROLLING_WINDOW=250,ALARM_PERIOD=20,TODAY='20230921',config={'plot':True,'plot_abs':True}):
    plot= config['plot']
    pos_res = []
    neg_res = []
    for bond_code in pbar(bond_codes):
        single_bond = data.loc[bond_code]
        single_bond = single_bond.loc[single_bond['TRADE_DT']<=TODAY]
        if single_bond.shape[0] < ROLLING_WINDOW:
            continue
        single_bond = single_bond.reset_index()
        single_bond['ACC_YIELD_CHANGE'] = (single_bond['B_ANAL_YIELD_CNBD'] - single_bond['B_ANAL_YIELD_CNBD'].shift(YIELD_CHANGE_PERIOD)) * 100 #bp
        single_bond['ACC_YIELD_CHANGE_ABS'] = single_bond['ACC_YIELD_CHANGE'].abs()
        quantile = single_bond['ACC_YIELD_CHANGE_ABS'].rolling(ROLLING_WINDOW,1).apply(lambda x: np.nanquantile(x.values,QUANTILE,interpolation='linear'))

        if plot:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,5), dpi=100)
            single_bond['TRADE_DT_DATETIME'] = pd.to_datetime(single_bond['TRADE_DT'])
            plot_abs = config['plot_abs']
            if plot_abs:
                sns.lineplot(data=single_bond, x=f"TRADE_DT_DATETIME", y=f"ACC_YIELD_CHANGE_ABS",linewidth = 2)
            else:
                sns.lineplot(data=single_bond, x=f"TRADE_DT_DATETIME", y=f"ACC_YIELD_CHANGE",linewidth = 2)
            sns.lineplot(x=single_bond['TRADE_DT_DATETIME'], y=quantile, ax = ax, label=f'过去250天 {str(int(QUANTILE*100))} quantile',linewidth=2,color='blue', linestyle='--')
            if not plot_abs:
                sns.lineplot(x=single_bond['TRADE_DT_DATETIME'], y=-quantile, ax = ax,linewidth=2,color='blue', linestyle='--')
            ax.set_title(f'当天存续债{bond_code}历史信用利差走势图')
            ax.set_ylabel(f'{self.TODAY}过去{YIELD_CHANGE_PERIOD}天累计变动 BP')
            ax.set_xlabel(f'日期')
            ax.grid()
            plt.show()

        
        single_bond_alarm_period_data = single_bond[-ALARM_PERIOD:]
        quantile_alarm_period_data = quantile[-ALARM_PERIOD:]
        SEL = single_bond_alarm_period_data.loc[
            single_bond_alarm_period_data['ACC_YIELD_CHANGE_ABS'] >= quantile_alarm_period_data
        ]
        if SEL.shape[0]>=1:
            if SEL['ACC_YIELD_CHANGE'].values[-1] >= 0 :
                pos_res.append([bond_code,SEL['TRADE_DT'].values[-1],SEL['ACC_YIELD_CHANGE'].values[-1],quantile.values[-1]])
            else:
                neg_res.append([bond_code,SEL['TRADE_DT'].values[-1],SEL['ACC_YIELD_CHANGE'].values[-1],quantile.values[-1]])
            if plot: print('ALARM!!!')
    POS_RES_DF = pd.DataFrame(pos_res,columns=['S_INFO_WINDCODE','最近突破日期','突破时变化程度bp','绝对预警线bp']).sort_values('最近突破日期',ascending=False).reset_index(drop=True)
    NEG_RES_DF = pd.DataFrame(neg_res,columns=['S_INFO_WINDCODE','最近突破日期','突破时变化程度bp','绝对预警线bp']).sort_values('最近突破日期',ascending=False).reset_index(drop=True)
    
            
    return POS_RES_DF, NEG_RES_DF   





