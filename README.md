# Research on DDoS attack defense method based on honeypot technology
Network information security has become an issue that cannot be ignored, and Distributed Denial of Service (DDoS) attacks are increasingly becoming a major threat to network security. Since DDoS attacks use standard protocols and services for intrusion, traditional methods are difficult to detect and defend. The active defense technique of honeypot technology can effectively detect and defend against DDoS attacks, but it faces anti-honeypot technology against detection, so this paper proposes a model that uses machine learning methods to detect in real time and forward DDoS attack streams to honeypots to effectively detect DDoS attacks and improve DDoS attack defense capabilities.

This paper firstly describes the classification of DDoS attacks and common defense methods, followed by the classification and defense process of honeypots. After preprocessing the PACP dataset of CIC-IDS2017, we use random forest to extract features, complete the training and testing of four classifiers, compare the support vector machine (SVM) model, K-nearest neighbors (KNN) model, decision tree model and The accuracy, precision, recall, and overall evaluation metrics of the Logistic Regression model were compared, and the accuracy and overall evaluation metrics of the Logistic Regression model were the highest with this small sample data. The logistic regression model is deployed in the network environment simulated by Mininet, and the honeypot system is deployed to record the network flow characteristics using Ryu Controller, and the attack flow detected by the logistic regression model is forwarded to the honeypot, which records the attack information to achieve the purpose of detecting and real-time defense against DDoS attacks using honeypot technology.

In this paper, a simulated environment is used to generate network traffic, and the effectiveness of the method is proved by analyzing the detection results of the model and the recording information of the honeypot. The accuracy of the logistic regression model selected in this environment is 99.33%, and the time to detect a single attack is 0.068 seconds. The experimental results show that DDoS attacks can be effectively detected and defended under the simulated environment using machine learning and honeypot technology.

__Keywords__: DDoS attacks; machine learning; Honeypot; Mininet


网络信息安全已经成为一个不容忽视的问题，分布式拒绝服务（Distributed Denial of Service, DDoS）攻击日益成为威胁网络安全的重大威胁。由于DDoS攻击使用标准的协议和服务进行入侵，传统的方法很难检测和防御。蜜罐技术的主动防御技术可以有效检测和防御DDoS攻击，但是面临着反蜜罐技术对抗检测，因此本文提出了使用机器学习方法实时检测，并将DDoS攻击流转发至蜜罐的模型，以有效检测DDoS攻击，提升DDoS攻击防御能力。

本文首先阐述了DDoS攻击的分类、常见防御方法，蜜罐的分类和防御过程。其次提出了基于机器学习的DDoS攻击实时检测方法，在对CIC-IDS2017的PACP数据集进行预处理后，使用随机森林提取特征，完成四种分类器的训练和测试，比较了支持向量机（Support Vector Machine, SVM）模型、K最近邻（K-nearest neighbors, KNN）模型、决策树模型和Logistic回归模型的准确率、精确率、召回率和综合评价指标，实验结果表明，小样本数据下Logistic回归模型的准确率和综合评价指标均为最高。最后提出了基于蜜罐技术的DDoS攻击主动防御方法，在Mininet模拟的网络环境中，部署Logistic回归模型和蜜罐系统，使用Ryu Controller记录网络流特征，通过Logistic回归模型检测攻击流并转发至蜜罐，进而由蜜罐记录攻击信息，实验结果表明，该方法实现了基于蜜罐技术的实时检测和防御DDoS攻击的目的。

本文采用模拟环境生成网络流量，通过分析模型的检测结果和蜜罐的记录信息，证明了该方法的有效性。在模拟环境下Logistic回归模型的准确度为99.33%，检测单个攻击的时间为0.068秒，实验结果表明，利用机器学习和蜜罐技术在模拟环境下能够有效检测并防御DDoS攻击。

### In Mininet server:
https://console.cloud.tencent.com/lighthouse/image/detail?rid=1&id=lhbp-a7x7ygjl
``` shell
# Implement the Topo
/bin/bash topo.py
```

### In Ryu Controller server:
https://drive.google.com/file/d/1mj9u675tgisSiUFOXSPTI1JqHPvgF3W1/view?usp=drive_link
```shell
# Start the normal stream
/bin/bash ./flow_simulate/normal_flow/bak_flow.sh
```

```shell
# Collect the normal stream
ryu-manager Switch_app.py collect_normal.py
```

```shell
# Start the attack stream
sudo ./trafgen --cpp --dev s3 --conf syn.trafgen --gap 10000
```

```shell
# Collect the attack stream
ryu-manager Switch_app.py collect_attack.py
```


