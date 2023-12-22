# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 17:35:33 2023

@author: chengyi.luo
"""

import time

# time.clock()默认单位为s
# 获取开始时间
start = time.perf_counter()




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




# 获取结束时间
end = time.perf_counter()
# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")
