# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:12:04 2023

@author: chengyi.luo
"""

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

# import progressbar
# bar = progressbar.ProgressBar()
from progressbar import *
pbar = ProgressBar()

from tabulate import tabulate
# 核心包
# import Wind
import Bonder 
# from Bonder import * 
import Selector
import dolphindb as ddb

import matplotlib as mpl
plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体
plt.rcParams["axes.unicode_minus"]=False   # 该语句解决图像中的“-”负号的乱码问题


#%%

# reload(Bonder)    # 重新载入之前载入的模块
# 初始化债券分析框架 Bonder
BondAnalyzer = Bonder.BondAnalyzer()

# 导入3张表
# STATIC
# BONDS
# FILLED_GUOKAI 见最上方国开债部分

# 静态筛选 
BondStaticFilters = [
    ('suffix','isin',['IB','SH','SZ']), # 默认选择标准代码
    ('CATEGORY1','==','信用债'),  # 默认选择信用债，不含利率债、同业存单、ABS、可转债
    ('isPerpetualBonds','==',0),  # 剔除：永续债
    ('IS_FAILURE','==',0)        # 剔除：发行失败
#     ('B_INFO_GUARTYPE','isna','') # 剔除：附带担保债
]
# 动态筛选
# 只考虑对应计算节点剩余期 限在 10 年以内个券
# 0. 仅保留 IB SH SZ债券
BondDynamicFilters = [
    ('B_ANAL_MATU_CNBD','<=', 10)
]
# 生成债券池
dataConfig = {
    'StaticPath': './Results/STATIC.parquet',
    'BondsPath': './Results/BONDS.parquet', 
    'FilledGuokaiPath': './Results/FILLED_GUOKAI.parquet',
    'BondStaticFilters': BondStaticFilters,
    'BondDynamicFilters': BondDynamicFilters,
}
df = BondAnalyzer.dataCenter(dataConfig)

#%%

# 时序图像绘制
# 参数
division_label = 'S_INFO_COMPIND_NAME1' # 根据这个字段的类别画线
Filters = [
    ('BOND_TYPE','==','产业债'),
]
CrossSectionConfig = {
    'date':'20230821', # 截面日期   原本是20230921，但是数据都是0，所以改掉
    'plotCrossSection':True, # 是否绘制date时间的柱状截面图
}
# methods 0: 简单平均 1: 中位数 2: 剩余本金加权
BondAnalyzer.plotCreditSpread(
    data=df,
    filters=Filters,
    divisionLabel=division_label,
    methods=[0,1,2],
    config=CrossSectionConfig
)

#%%  个债预警

# Mac 3500个债 约90s Windows较慢 4mins30s
# 债券累计存续时常小于250天不预警
YIELD_CHANGE_PERIOD=7
QUANTILE=0.98
ROLLING_WINDOW=250
ALARM_PERIOD=20
alarmConfig = {
    'TODAY':'20230821' #扫描哪天的存续债
}
ALARMRES = BondAnalyzer.initBondAlarms(
    df,
    YIELD_CHANGE_PERIOD=YIELD_CHANGE_PERIOD, # 累计收益周期
    QUANTILE=QUANTILE, # 预警分位数
    ROLLING_WINDOW=ROLLING_WINDOW, # 历史分位窗口
    ALARM_PERIOD=ALARM_PERIOD, # 预警窗口
    config=alarmConfig
)
df_alarmed_pos_bond = ALARMRES['df_alarmed_pos_bond']
df_alarmed_neg_bond = ALARMRES['df_alarmed_neg_bond']
alarmed_pos_bond_codes = ALARMRES['alarmed_pos_bond_codes']
alarmed_neg_bond_codes = ALARMRES['alarmed_neg_bond_codes']
ALARMED_POS_BONDS = ALARMRES['ALARMED_POS_BONDS']
ALARMED_NEG_BONDS = ALARMRES['ALARMED_NEG_BONDS']

#%%

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

# 监测特定日期的价格异动债券   输入合并后的df数据  然后返回股票代码 和 一些信息(?
def initBondAlarms(data,YIELD_CHANGE_PERIOD=7,QUANTILE=0.95,ROLLING_WINDOW=250,ALARM_PERIOD=20,config={}):
    # config = {
    #     'TODAY':''
    # }
    TODAY = config['TODAY']
    
    data = copy.deepcopy(data)
    data = data.sort_values(['S_INFO_WINDCODE','TRADE_DT']).set_index('S_INFO_WINDCODE')   #wind数据库标识码
    print(f'开始检测最近{ALARM_PERIOD}日出现超越{YIELD_CHANGE_PERIOD}日累计估值收益变动的绝对值在历史{ROLLING_WINDOW}日内的{QUANTILE}分位数的{self.TODAY}债券')
    
    # 今天有哪些债券需要检验
    bond_codes = data.loc[
        (data['TRADE_DT']==TODAY) &
        (data['B_ANAL_MATU_CNBD']>0)
    ].index.unique()
    showBondConfig = {
        'plot':False,
    }
    df_alarmed_pos_bond,df_alarmed_neg_bond = showSingleBond(
        data=data,
        bond_codes=bond_codes,
        YIELD_CHANGE_PERIOD=YIELD_CHANGE_PERIOD,
        QUANTILE=QUANTILE,
        ROLLING_WINDOW=ROLLING_WINDOW,
        ALARM_PERIOD=ALARM_PERIOD,
        TODAY = TODAY,
        config=showBondConfig
    )
    
    alarmed_pos_bond_codes,alarmed_neg_bond_codes = df_alarmed_pos_bond['S_INFO_WINDCODE'].values, df_alarmed_neg_bond['S_INFO_WINDCODE'].values

    ALARMED_POS_BONDS = data.loc[
        (data.index.isin(alarmed_pos_bond_codes))
    ]
    ALARMED_NEG_BONDS = data.loc[
        (data.index.isin(alarmed_neg_bond_codes))
    ]

    RES = {
        'df_alarmed_pos_bond':df_alarmed_pos_bond,
        'df_alarmed_neg_bond':df_alarmed_neg_bond,
        'alarmed_pos_bond_codes':alarmed_pos_bond_codes,
        'alarmed_neg_bond_codes':alarmed_neg_bond_codes,
        'ALARMED_POS_BONDS':ALARMED_POS_BONDS,
        'ALARMED_NEG_BONDS':ALARMED_NEG_BONDS
    }
    return RES


#%% 类拆成函数往下运行

YIELD_CHANGE_PERIOD=7
QUANTILE=0.95
ROLLING_WINDOW=250
ALARM_PERIOD=20
TODAY='20230921'
config={'plot':True,'plot_abs':True}


# Mac 3500个债 约90s Windows较慢 4mins30s
# 债券累计存续时常小于250天不预警
YIELD_CHANGE_PERIOD=7
QUANTILE=0.98
ROLLING_WINDOW=250
ALARM_PERIOD=20
alarmConfig = {
    'TODAY':'20230821' #扫描哪天的存续债
}
ALARMRES = BondAnalyzer.initBondAlarms(
    df,
    YIELD_CHANGE_PERIOD=YIELD_CHANGE_PERIOD, # 累计收益周期
    QUANTILE=QUANTILE, # 预警分位数
    ROLLING_WINDOW=ROLLING_WINDOW, # 历史分位窗口
    ALARM_PERIOD=ALARM_PERIOD, # 预警窗口
    config=alarmConfig
)