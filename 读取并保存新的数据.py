import dolphindb as ddb

s = ddb.session()
s.connect("115.239.209.123", 8868, "admin", "123456")  # 连接数据库
print('链接成功')
