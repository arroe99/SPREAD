import dolphindb as ddb

# s = ddb.session()
# s.connect("115.239.209.123", 8868, "admin", "123456")  # 连接数据库
# print('链接成功')
# s.run(loadParquet("/home/appadmin/hy/ficc/GuangFa/Results/STATIC.parquet"))


# 杨晗服务器节点
s = ddb.session()
isConnect = s.connect("192.168.100.43", 7988, "admin", "123456")  # 连接数据库
print(isConnect)

# 两种读取数据的方法
# script = """parquet::loadParquet("/home/hanyang/luodata/STATIC.parquet")"""
# STATIC = s.run(script)

script = """STATIC = parquet::loadParquet("/home/hanyang/luodata/STATIC.parquet")"""
s.run(script)
STATIC = s.run("STATIC")


script = """
// undef all 

// 打印筛选条件
def slectionString(Filters){   // isna 和 isnotna的情况还要改一下  先跳过了
    print('筛选条件：')
    for(filter in Filters){
        if (size(filter[2]) > 1){
            print(filter[0] + ' ' + filter[1] + ' ' + concat(filter[2], ', '))
        }
        else {
            print(filter[0] + ' ' + filter[1] + ' ' + filter[2])
        }
    }
}

// 筛选数据
def selector(df, Filters){    // 改进方向： 里面嵌一层函数
    slectionString(Filters)
    data = df
    // if (isNull(Filters)) return data // 如果过滤条件为空，直接返回数据
    selection = take(true, data.rows())
    for(filter in Filters){
        if (filter[1]=='isin'){
            selection = selection && data[filter[0]] in filter[2]
        }
        else if (filter[1] == 'isna'){
            print(filter[1])
            selection = selection && isNull(data[filter[0]])
        }
        else if (filter[1] == 'isnotna'){
            selection = selection && not isNull(data[filter[0]])
        }
        else if (filter[1] == '=='){
            selection = selection && data[filter[0]] == filter[2] && !isNull(data[filter[0]])
        }
        else if (filter[1] == '!='){
            selection = selection && data[filter[0]] != filter[2] && !isNull(data[filter[0]])
        }
        else if (filter[1] == '>='){
            selection = selection && data[filter[0]] >= filter[2] && !isNull(data[filter[0]])
        }
        else if (filter[1] == '<='){
            selection = selection && data[filter[0]] <= filter[2] && !isNull(data[filter[0]])
        }
        else if (filter[1] == '>'){
            selection = selection && data[filter[0]] > filter[2] && !isNull(data[filter[0]])
        }
        else if (filter[1] == '<'){
            selection = selection && data[filter[0]] < filter[2] && !isNull(data[filter[0]])
        }
        else {
            print('请检查逻辑符号，有效逻辑符号为["==","!=",">=","<=",">","<","isin","isna","isnotna"]')
        }
    }
    data = data[selection]
    // data = data[each(isValid, data.values()).rowAnd()]   // 删除含有缺失值的行
    return data
}

// 导入插件
// loadPlugin("/home/hanyang/luodata/PluginParquet.txt")   // 需要导入一次即可

// 导入数据
// 1、STATIC数据
STATIC = parquet::loadParquet("/home/hanyang/luodata/STATIC.parquet")   // 原本21930行
STATIC.dropColumns!(`B_INFO_ISSUERCODE)   // 好像没有必要删这一列

///2、BONDS数据
BONDS = parquet::loadParquet("/home/hanyang/luodata/BONDS.parquet") 

///3、国开数据
FILLED_GUOKAI = parquet::loadParquet("/home/hanyang/luodata/FILLED_GUOKAI.parquet")

// 筛选条件
BondStaticFilters = [
    ('suffix','isin',['IB','SH','SZ']),     // ANY VECTOR
    ('CATEGORY1','==','信用债'),          
    ('isPerpetualBonds','==',0),     
    ('IS_FAILURE','==',0)
]            
BondDynamicFilters = [('B_ANAL_MATU_CNBD','<=', 10)]

STATIC = selector(df = STATIC, Filters = BondStaticFilters)
BONDS = selector(df = BONDS, Filters = BondDynamicFilters)   

// 合并静态动态数据  并计算信用利差
df = ej(STATIC, BONDS, `S_INFO_WINDCODE) // 这个表里的B_ANAL_YIELD_CNBD	是BONDS数据里的 有四位小数
df['B_ANAL_MATU_CNBD'] = round(df['B_ANAL_MATU_CNBD'], 2)   // 或者 update df set B_ANAL_MATU_CNBD = round(B_ANAL_MATU_CNBD, 2)
df = lj(df, FILLED_GUOKAI, `TRADE_DT`B_ANAL_MATU_CNBD)
df = select *, B_ANAL_YIELD_CNBD - B_ANAL_YIELD_GUOKAI as CREDIT_SPREAD from df 

"""

s.run(script)

df = s.run("df")

df.to_csv('./Results/temp_ddb.csv', index=False, encoding='utf-8-sig')


