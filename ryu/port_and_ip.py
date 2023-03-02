#   coding: UTF-8
#   @author: winwin
#   Created by yww on 2022-03-28
#   IDE PyCharm

# h1-h5 h6-h10 h11-h15
s1_all_ip =['121.0.0.1', '121.0.0.2', '121.0.0.3', '121.0.0.4', '121.0.0.5']
s2_all_ip =['122.0.0.1', '122.0.0.2', '122.0.0.3', '122.0.0.4', '122.0.0.5']
s3_all_ip =['123.0.0.1', '123.0.0.2', '123.0.0.3', '123.0.0.4', '123.0.0.5']


s1_ports = [21, 23, 25, 80]
s2_ports = [21, 22, 23, 25]
s3_ports = [21, 22, 23, 25]

# 给出三台交换机端口
ext_ports = [1599, 2233]
k1 = s1_ports
k2 = s2_ports + ext_ports
k3 = s3_ports + ext_ports

# destination ip 目的地址
s1_dst_ip = s2_all_ip + s3_all_ip
s2_dst_ip = s1_all_ip + s3_all_ip
s3_dst_ip = s1_all_ip + s2_all_ip
