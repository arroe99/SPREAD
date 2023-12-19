import pandas as pd
import numpy as np
import re
import math
import copy

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
        final_selection = final_selection & newselection
    ret = data.loc[final_selection]
    print('筛选结果数据量：', ret.shape[0])
    return ret

# 显示筛选条件
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