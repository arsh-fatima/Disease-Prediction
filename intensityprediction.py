import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('diseasepredict.csv')

feature_col = ['Disease','Symptoms','Weight','Height','Severity','Age','Gender','BMI_Level','Region','Season'] 
df['Disease'],_ = pd.factorize(df['Disease'])
df['Symptoms'],_ = pd.factorize(df['Symptoms'])
df['Severity'],_ = pd.factorize(df['Severity'])
df['Gender'],_ = pd.factorize(df['Gender'])
df['Region'],_ = pd.factorize(df['Region'])
df['Season'],_ = pd.factorize(df['Season'])

X = df[feature_col] 
y = df.Intensity # Target variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
test = model.fit(X_train,y_train)

pickle.dump(test, open('intensitypredict.pkl', 'wb'))
