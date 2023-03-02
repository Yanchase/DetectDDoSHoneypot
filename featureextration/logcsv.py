#   coding: UTF-8
#   @author: winwin
#   Created by yww on 2022-03-28
#   IDE PyCharm

import csv

a = open('ddos.csv', 'r')
reader = csv.reader(a)

# 以空格分界输入转换为ddos.log
with open('ddos.log', 'w') as f:
    for i in reader:
        for x in i:
            f.write(x)
            f.write(' ')
        f.write('\n')
a.close()

