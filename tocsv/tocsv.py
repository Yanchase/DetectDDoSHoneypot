import os
import re
import numpy
import csv

path = r'collect_data.log'  # 配置日志路径
file = open(path)  # 读取日志
iterf = iter(file)
csv_file = open('collect.csv', 'w', encoding='gbk', newline='')  # 生成csv文件
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["时间戳", "流平均包数", "流包平均比特", "端口增速","流增长速率","源ip增速","流量类型"])  # 设计csv文件字段

for line in iterf:
    c = line.find('内存使用')  # 查找内存使用字段在日志行中的索引位置，位置为42
    d = line.find('可用')
    e = line.find('/')
    f = line.find('物理')
    g = line.find('8GB】')
    # print(g)
    if c > -1:
        # print(c,d)
        time = line[7:29]  # 时间字段在打印日志中的7—29索引位置的内容
        use_mem = line[c + 5:d - 1]  # 同上
        free_mem = line[d + 3:e]
        wuli_mem = line[f + 3:g - 4]
        csv_writer.writerow([time, use_mem, free_mem, wuli_mem])  # 将剪切内容导入csv文件

exit()
