#export.py
import pandas as pd
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from bento_service import IrisClassifier
from sklearn.decomposition import PCA
iris = datasets.load_iris()
X, Y = iris.data, iris.target
le = preprocessing.LabelEncoder()
#x = X.iloc[:,1:17]
#y = le.fit_transform(df_temp1.iloc[:,17])

print(X)

'''
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, test_size=0.2, random_state=42)
estimators = [
    ('GB', GradientBoostingClassifier(learning_rate=1.0, max_depth=8, max_features=0.35000000000000003, min_samples_leaf=9, min_samples_split=6, n_estimators=100, subsample=0.6500000000000001,random_state=42)),
    ('tree', DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=1, min_samples_split=2,random_state=42))
]
model = StackingClassifier(
    estimators=estimators,stack_method='predict',final_estimator=KNeighborsClassifier(n_neighbors=3, p=2, weights="uniform")
)
model.fit(X_train,y_train) #模型1
X_train, X_test, y_train, y_test = train_test_split(x1, y1,
                                                    train_size=0.75, test_size=0.25, random_state=42)
clf=GaussianNB()
clf.fit(X_train,y_train) #模型2
# 从bento_service.py里面抽取的服务类
iris_classifier_service = IrisClassifier()

# 打包两个模型
iris_classifier_service.pack('StackingClassifier', model)
iris_classifier_service.pack('GaussianNB', clf)
# 保存到本地，路径可自己写
saved_path = iris_classifier_service.save()
'''