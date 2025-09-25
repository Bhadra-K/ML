import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data=pd.read_csv("diabetes_prediction_dataset.csv")
print(data.head())
print(data.shape)
print(data.isnull().sum())

data['gender']=data['gender'].map({'Male':0,'Female':1,'Other':2})
print(data.head())
print(data['smoking_history'].value_counts())
data=pd.get_dummies(data,columns=['smoking_history'],drop_first=True)
print(data)
print(data.isnull().sum())

x=data[['gender','age','hypertension','smoking_history_former',
        'smoking_history_never','smoking_history_not current','smoking_history_ever','bmi','HbA1c_level','blood_glucose_level']]
y=data['diabetes']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

acc=accuracy_score(y_test,y_pred)*100
print(f"Accuracy: {acc:.2f}%")
cr=classification_report(y_test,y_pred)
print("Classification Report: ",cr)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ",cm)

plt.scatter(x_test['bmi'], y_test)
# sns.boxplot(x=y_test, y=x_test['bmi'])
plt.title("Bmi vs Diabetes")
plt.xlabel("Bmi")
plt.ylabel("Diabetes")
plt.show()
