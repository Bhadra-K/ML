import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# Load data
data=pd.read_csv("mail_data.csv")
print(data.head())
# Check for missing values
print(data.isnull().sum())
# Encode labels
data['Category']=data['Category'].map({'ham':1,'spam':0})
print(data.head())
# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Message'])  # Converts text to numerical matrix
print(X)
# Features and target
x=data[['Message']]
y=data[['Category']]
# Train-test split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Train the model
model=LogisticRegression()
model.fit(x_train,y_train)
# Predict
ypred=model.predict(x_test)
# Evaluate
acc=accuracy_score(ypred,y_test)*100
print(f"Accuracy: {acc:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, ypred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ypred))
