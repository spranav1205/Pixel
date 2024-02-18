import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

file_path = '../../PycharmProjects/TeamAnantPayload/IRIS.csv'

iris_data = pd.read_csv(file_path)
summary = iris_data.describe()
print(summary)

y=iris_data["species"]
x=iris_data.drop("species",axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=69)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

''' 
The limitations are as follows 
 - Overfitting
 - Linearity (we assume linearity)
 - Any anomaly in the data set should be taken care off before training the data 
'''
