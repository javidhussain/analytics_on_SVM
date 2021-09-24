import pandas as pd
import numpy as np


data=pd.read_csv('diabetes.csv')
dataset=data.pop('Outcome')


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,dataset,test_size=0.3)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
  

from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

from sklearn.svm import SVC  
model_svc = SVC(kernel='linear') 
model_svc.fit(x_train, y_train)
y_pred_svc = model_svc.predict(x_test)
  

from sklearn import metrics
print("Support vector machine model accuracy(in %):", metrics.accuracy_score(y_test, y_pred_svc)*100)
