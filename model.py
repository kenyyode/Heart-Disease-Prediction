## imports 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import joblib, os

## load the data into the dataset 
df = pd.read_csv('heart-diseaseuci/heart.csv')

## feature selection 
X = df.drop('target', axis=1)
y = df['target']

## split data into training and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

## initialize model 
model = LogisticRegression(max_iter=1000)

## train_model 
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

## make directory 
os.makedirs('model', exist_ok=True)

## dump model using joblib 
joblib.dump(model, 'model/heart_disease_model.pkl')