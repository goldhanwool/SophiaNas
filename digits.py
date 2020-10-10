#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

digits = load_digits()
print(type(dir(digits)))

digits.keys()

digits_data = digits.data
print(digits_data.shape) 

digits_label = digits.target
print(digits_label)

print(digits.target_names)

X_train, X_test, y_train, y_test = train_test_split(digits_data, digits_label, test_size=0.2, random_state=12)

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

# 평가지표는 precision으로 얼마나 잘 예측해서 나누었는가가 중요하다. 적합   sgd, LogisticRegression


# In[ ]:




