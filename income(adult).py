import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
data=pd.read_csv("adult.csv")
print(data.head())
print(data.isnull().sum())
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
x= data.drop('income', axis=1)
y= data['income']
# categorical_cols=x.select_dtypes(include='object').columns
# for col in categorical_cols:
#     le = LabelEncoder()
#     x[col] = le.fit_transform(x[col])
data=pd.DataFrame(data)
x_encoded= pd.get_dummies(x)
print(x_encoded.head())
x_train,x_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
ypred=model.predict(x_test)
acc=accuracy_score(ypred,y_test)*100
print(f"Accuracy: {acc:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, ypred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ypred))
