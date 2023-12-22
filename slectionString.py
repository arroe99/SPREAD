# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:02:01 2023

@author: chengyi.luo
"""

def slectionString(Filters):
    
    def parse_tuple_condition(x):
        x = [str(i) for i in list(x)]  # 转成字符串
        return ' '.join(x)   # 所有字符串通过空格连接成一个字符串
    
    condition = ''   # 始化一个空字符串，用于存储最终的条件字符串。
    for filter in Filters:
        if condition != '': condition+=' and '  # 如果非空，加上个and
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

BondStaticFilters = [
    ('suffix','isin',['IB','SH','SZ']),     # 默认选择标准代码  21722    原本21930行
    ('CATEGORY1','==','信用债'),            # 默认选择信用债，不含利率债、同业存单、ABS、可转债  14673
    ('isPerpetualBonds','==',0),            # 剔除：永续债    14568
    ('IS_FAILURE','==',0),                   # 剔除：发行失败  14039
    ('B_INFO_GUARTYPE','isna','')
]    
slectionString(BondStaticFilters)


BondStaticFilters = [
    ('isPerpetualBonds','==',0),
    [('CNBD_HIDDEN_CREDITRATING','==','AAA'),('CNBD_HIDDEN_CREDITRATING','==','AA+')],
]
slectionString(BondStaticFilters)



