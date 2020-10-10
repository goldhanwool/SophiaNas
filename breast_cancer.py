#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

breast_cancer = load_breast_cancer()
print(type(dir(breast_cancer)))

breast_cancer.keys()

breast_cancer_data = breast_cancer.data
print(breast_cancer_data.shape) 

breast_cancer_label = breast_cancer.target
print(breast_cancer_label)

print(breast_cancer.target_names)

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data, breast_cancer_label, test_size=0.2, random_state=12)

decision_tree = DecisionTreeClassifier(random_state=24)
decision_tree.fit(X_train, y_train)

randomforest = RandomForestClassifier(random_state=24)
randomforest.fit(X_train, y_train)

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)
y_pred_forest = randomforest.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_sgd_model = sgd_model.predict(X_test)
y_pred_logistic_model = logistic_model.predict(X_test)


print(classification_report(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_sgd_model))
print(classification_report(y_test, y_pred_logistic_model))

#평가지표는 recall 암의 경우에 0번 '악성'에 대한 실제로 실제데이터와 얼마나 일치하는지가 중요하기 때문에 recall이 평가지표가 되어야하고 그 중에서도 0번에 대한 recall이 중요하기때문에 결정트리모델이 가장 우수하다고 판단된다.


# In[ ]:




