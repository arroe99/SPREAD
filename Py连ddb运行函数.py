# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:27:35 2023

@author: chengyi.luo
"""

import dolphindb as ddb


# 杨晗服务器节点
s = ddb.session()
isConnect = s.connect("192.168.100.43", 7988, "admin", "123456")  # 连接数据库
print(isConnect)


