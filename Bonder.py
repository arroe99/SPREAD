import pandas as pd
import numpy as np
import re
import math
import copy
# from progressbar import progressbar as pbar
from tqdm import tqdm
from progressbar import *
pbar = ProgressBar()
# pbar.max_error = False
# pbar = ProgressBar(max_error = False)

import matplotlib.pyplot as plt
import seaborn as sns
import Selector
from tabulate import tabulate
from  matplotlib.colors import LinearSegmentedColormap


class BondAnalyzer():

    def __init__(self,config={}):
        pass
        
    # 三表合并
    def dataCenter(self,config={}):
        # config = {
        #     'StaticPath': ,
        #     'BondsPath':,
        #     'FilledGuokaiPath':,
        #     'BondStaticFilters':,
        #     'BondDynamicFilters':,
        # }

        # 静态筛选
        STATIC = pd.read_parquet(config['StaticPath'])
        STATIC.drop('B_INFO_ISSUERCODE',inplace=True,axis=1)
        STATIC = Selector.select(STATIC,config['BondStaticFilters'])
        print('STATIC:', Selector.slectionString(config['BondStaticFilters']))

        # 动态筛选
        BONDS = pd.read_parquet(config['BondsPath'])
        BONDS = Selector.select(BONDS,config['BondDynamicFilters'])
        print('BONDS:', Selector.slectionString(config['BondDynamicFilters']))

        # 动态静态结合
        df = pd.merge(STATIC, BONDS, on='S_INFO_WINDCODE',validate='one_to_many')

        # 信用计算利差
        print('信用计算利差中')
        FILLED_GUOKAI = pd.read_parquet(config['FilledGuokaiPath'])
        df['B_ANAL_MATU_CNBD'] = df['B_ANAL_MATU_CNBD'].round(2)
        df = pd.merge(df,FILLED_GUOKAI,on=['TRADE_DT','B_ANAL_MATU_CNBD'],how='left')
        df['CREDIT_SPREAD'] = df['B_ANAL_YIELD_CNBD'] - df['B_ANAL_YIELD_GUOKAI']

        print('数据处理完成')
        return df

    # 展示指定债券的图像
    def showSingleBond(self, data, bond_codes=[], YIELD_CHANGE_PERIOD=7,QUANTILE=0.95,ROLLING_WINDOW=250,ALARM_PERIOD=20,TODAY='20230921',config={'plot':True,'plot_abs':True}):
        plot= config['plot']
        pos_res = []
        neg_res = []
        for bond_code in tqdm(bond_codes):
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

    # 监测特定日期的价格异动债券
    def initBondAlarms(self,data,YIELD_CHANGE_PERIOD=7,QUANTILE=0.95,ROLLING_WINDOW=250,ALARM_PERIOD=20,config={}):
        # config = {
        #     'TODAY':''
        # }
        self.TODAY = config['TODAY']
        
        data = copy.deepcopy(data)
        data = data.sort_values(['S_INFO_WINDCODE','TRADE_DT']).set_index('S_INFO_WINDCODE')
        print(f'开始检测最近{ALARM_PERIOD}日出现超越{YIELD_CHANGE_PERIOD}日累计估值收益变动的绝对值在历史{ROLLING_WINDOW}日内的{QUANTILE}分位数的{self.TODAY}债券')
        
        # 今天有哪些债券需要检验
        bond_codes = data.loc[
            (data['TRADE_DT']==self.TODAY) &
            (data['B_ANAL_MATU_CNBD']>0)
        ].index.unique()
        showBondConfig = {
            'plot':False,
        }
        df_alarmed_pos_bond,df_alarmed_neg_bond = self.showSingleBond(
            data=data,
            bond_codes=bond_codes,
            YIELD_CHANGE_PERIOD=YIELD_CHANGE_PERIOD,
            QUANTILE=QUANTILE,
            ROLLING_WINDOW=ROLLING_WINDOW,
            ALARM_PERIOD=ALARM_PERIOD,
            TODAY = self.TODAY,
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

    # 绘制热力图
    def getHeatMap(self,data,column,config):
        alarmres = config['alarmres']
        df_alarmed_pos_bond = alarmres['df_alarmed_pos_bond']
        df_alarmed_neg_bond = alarmres['df_alarmed_neg_bond']
        alarmed_pos_bond_codes = alarmres['alarmed_pos_bond_codes']
        alarmed_neg_bond_codes = alarmres['alarmed_neg_bond_codes']
        ALARMED_POS_BONDS = alarmres['ALARMED_POS_BONDS']
        ALARMED_NEG_BONDS = alarmres['ALARMED_NEG_BONDS']

        filters = []
        if 'filters' in config.keys():
            filters = config['filters']

        if isinstance(column,list):
            temp = None
            for c in column:
                if temp is None:
                    temp = data[c].fillna('NaN').astype(str)
                else:
                    temp = temp + '_' + data[c].fillna('NaN').astype(str)
            newc = 'CUSTOMIZED_' + ('_'.join(list(map(str, column))))
            data[newc] = temp
            column = newc
            
        def getCountTable(data, column):
            table = data[column].value_counts()
            total = table.sum()
            printdf = pd.DataFrame(index=pd.Series(list(table.keys()),name=column), columns=['count','percentageInAlarm'])
            for each in table.keys():
                printdf.loc[each] = [table[each], table[each]/total]
            return printdf


        data = data.loc[
            (data['TRADE_DT'] == self.TODAY)
        ]
        ALARMED_BONDS_TOTAL = data.groupby([column])['S_INFO_WINDCODE'].count().to_frame('count')
        
        ALARMED_POS_BONDS_NOW = data.loc[
            (data['S_INFO_WINDCODE'].isin(alarmed_pos_bond_codes))  
        ]
        ALARMED_POS_BONDS_NOW = Selector.select(ALARMED_POS_BONDS_NOW,filters)
        print('ALARMED_POS_BONDS_NOW:', Selector.slectionString(filters))

        
        print(f'存在{ALARMED_POS_BONDS_NOW[column].isna().sum()}条正向突破预警线债券因缺失分类而不考虑进统计数据中')
        ALARMED_POS_BONDS_STATISTICS = getCountTable(data = ALARMED_POS_BONDS_NOW, column=column)
        ALARMED_POS_BONDS_STATISTICS = ALARMED_POS_BONDS_STATISTICS.apply(pd.to_numeric) 
        ALARMED_POS_BONDS_STATISTICS['recentChangeDirection'] = '+'
        POSTOTAL = ALARMED_POS_BONDS_STATISTICS['count'].sum()
        # print(tabulate(ALARMED_POS_BONDS_STATISTICS, headers='keys', tablefmt='psql'))
        

        ALARMED_NEG_BONDS_NOW = data.loc[
            (data['S_INFO_WINDCODE'].isin(alarmed_neg_bond_codes))
        ]
        ALARMED_NEG_BONDS_NOW = Selector.select(ALARMED_NEG_BONDS_NOW,filters)
        print('ALARMED_NEG_BONDS_NOW:', Selector.slectionString(filters))


        
        print(f'存在{ALARMED_NEG_BONDS_NOW[column].isna().sum()}条反向突破预警线债券因缺失分类而不考虑进统计数据中')
        ALARMED_NEG_BONDS_STATISTICS = getCountTable(data = ALARMED_NEG_BONDS_NOW, column=column)
        ALARMED_NEG_BONDS_STATISTICS = ALARMED_NEG_BONDS_STATISTICS.apply(pd.to_numeric)
        NEGTOTAL = ALARMED_NEG_BONDS_STATISTICS['count'].sum()

        ALARMED_NEG_BONDS_STATISTICS = -ALARMED_NEG_BONDS_STATISTICS
        ALARMED_NEG_BONDS_STATISTICS['recentChangeDirection'] = '-'
        # print(tabulate(ALARMED_NEG_BONDS_STATISTICS, headers='keys', tablefmt='psql'))

        PRESSION = pd.DataFrame(index=pd.Series(['+','-'],name='涨跌比'),columns=['count','percentageInAlarm'])


        c = ["darkgreen","green","palegreen","white", "lightcoral","red","darkred"]
        v = [0,.15,.4,.5,0.6,.9,1.]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

        
        PRESSION.loc['+',:] = [POSTOTAL,POSTOTAL/(POSTOTAL+NEGTOTAL)]
        PRESSION.loc['-',:] = [NEGTOTAL,NEGTOTAL/(POSTOTAL+NEGTOTAL)]


        ALARMED_POS_BONDS_STATISTICS['industryTotalCount'] = ALARMED_BONDS_TOTAL['count']
        ALARMED_POS_BONDS_STATISTICS['percentageInIndustry'] = ALARMED_POS_BONDS_STATISTICS['count'] / ALARMED_BONDS_TOTAL['count']
        ALARMED_NEG_BONDS_STATISTICS['industryTotalCount'] = ALARMED_BONDS_TOTAL['count']
        ALARMED_NEG_BONDS_STATISTICS['percentageInIndustry'] = ALARMED_NEG_BONDS_STATISTICS['count'] / ALARMED_BONDS_TOTAL['count']

        
        PRESSION = PRESSION.reset_index()
        
        ALARM_SUMMARY = pd.concat([ALARMED_POS_BONDS_STATISTICS,ALARMED_NEG_BONDS_STATISTICS]).sort_values('percentageInAlarm',ascending=False).reset_index()
        return {
            'ALARM_PRESSION': PRESSION.style.background_gradient(vmin=0, vmax=1, cmap=cmap, subset=['percentageInAlarm']),
            'ALARM_SUMMARY': ALARM_SUMMARY[[
                column,'recentChangeDirection','count','percentageInAlarm','percentageInIndustry','industryTotalCount'
            ]].style.background_gradient(vmin=-1, vmax=1, cmap=cmap, subset=['count','percentageInAlarm','percentageInIndustry']),
        }
        

    # 绘制走势图和条形图
    # method: 0, 1, 2
    # 0: simple average
    # 1: median
    # 2: 余额加权平均
    def plotCreditSpread(self,data,filters,divisionLabel='',methods=[0],config={'plotCrossSection':False}):
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
        data = Selector.select(data,filters)
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
    
        conditionString = Selector.slectionString(filters)
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
    
  


    
    
