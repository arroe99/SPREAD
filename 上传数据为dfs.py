# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:29:57 2023

@author: chengyi.luo
"""

import os 
import dolphindb as ddb 
import pandas as pd
import dolphindb.settings as keys
    
os.getcwd()
os.chdir(r'D:\Jupyter\spread')   # 设置本地数据工作路径

# 连接dolphinDB服务器节点
s = ddb.session()
isConnect = s.connect("115.239.209.123", 8892, "admin", "123456")  # 连接数据库
print(isConnect)

# 读取本地数据
STATIC = pd.read_parquet('./Results/STATIC.parquet')     # 初始行数21930
BONDS = pd.read_parquet('./Results/BONDS.parquet')
# B_INFO_CARRYDATE 的范围：从 1996-05-15 00:00:00 到 2023-09-28 00:00:00
FILLED_GUOKAI = pd.read_parquet('./Results/FILLED_GUOKAI.parquet')



# # 创建分区数据库

# s.database(dbName='temp', partitionType=keys.VALUE, partitions=["AMZN","NFLX", "NVDA"], dbPath="dfs://temp")
# 等效于 s.run("db=database('dfs://valuedb', VALUE, ['AMZN','NFLX','NVDA'])") 

# 删除数据库
# if s.existsDatabase("dfs://temp"):
#     s.dropDatabase("dfs://temp")

#%% Sol1 先上传到服务器  然后建库加表

# 上传数据到服务器
s.upload({'STATIC' : STATIC,
'BONDS': BONDS,
'FILLED_GUOKAI': FILLED_GUOKAI
}
)   # 运行时间很久很久....

# 试验是否上传数据成功
print(s.loadTable("STATIC").toDF())


# 失败
# s.database(dbName='STATIC', partitionType=keys.VALUE, partitions=["公司债", "资产支持证券", "短期融资券", "定向工具", "中期票据", "可转债", "企业债", "可分离转债存债", "可交换债", "金融债", "项目收益票据", "政府支持机构债"], dbPath='dfs://STATIC')
# 等效于 s.run("db=database('dfs://valuedb', VALUE, ['AMZN','NFLX','NVDA'])")
# s.run("db=database('dfs://temp', VALUE, ["公司债", "资产支持证券", "短期融资券", "定向工具", "中期票据", "可转债", "企业债", "可分离转债存债", "可交换债", "金融债", "项目收益票据", "政府支持机构债"])")

# 设置共享表
s.run("share STATIC as STATIC_shared")
s.run("share BONDS as BONDS_shared")
s.run("share FILLED_GUOKAI as FILLED_GUOKAI_shared")




#%% 建立分区数据库

script = """
db = database('dfs://temp', VALUE, 1996.01M..2023.12M)
pt = db.createPartitionedTable(STATIC, `pt, `B_INFO_CARRYDATE)
pt.append!(STATIC)
"""

s.run(script)


#%% 获取表的信息

# 缺失情况
STATIC.info()
print(STATIC.B_INFO_CARRYDATE.isnull().sum())


# 获取各列的不重复数
column_unique_counts = STATIC.nunique()
print(column_unique_counts)

# 获取分布情况
payment_type_range = STATIC['S_INFO_EXCHMARKET'].value_counts()
print(payment_type_range)

# 查时间范围
min_date = STATIC['B_INFO_CARRYDATE'].min()
max_date = STATIC['B_INFO_CARRYDATE'].max()
print("B_INFO_CARRYDATE 的范围：从", min_date, "到", max_date)



# 国开数据情况

# 缺失情况
FILLED_GUOKAI.info()
FILLED_GUOKAI.isnull().sum()
FILLED_GUOKAI.B_INFO_CARRYDATE.isnull().sum()


# 获取各列的不重复数
FILLED_GUOKAI.nunique()

# 获取分布情况
FILLED_GUOKAI['S_INFO_EXCHMARKET'].value_counts()

# 查时间范围
min_date = FILLED_GUOKAI['TRADE_DT'].min()
max_date = FILLED_GUOKAI['TRADE_DT'].max()
print("B_INFO_CARRYDATE 的范围：从", min_date, "到", max_date)


# BONDS数据
BONDS.info()
BONDS.isnull().sum()
BONDS.B_INFO_CARRYDATE.isnull().sum()

# 获取各列的不重复数
BONDS.nunique()

# 获取分布情况
BONDS['S_INFO_EXCHMARKET'].value_counts()

# 查时间范围
min_date = BONDS['TRADE_DT'].min()
max_date = BONDS['TRADE_DT'].max()
print("B_INFO_CARRYDATE 的范围：从", min_date, "到", max_date)


#%% 直接用csv数据导入




