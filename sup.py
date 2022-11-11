import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import pickle
df = pd.read_csv("hearing_test.csv")
X = df.drop('test_result', axis=1)
y = df['test_result']
print(df.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(scaled_X_train,y_train)
pickle.dump(model,open("model.pkl","wb"))

# print(help(train_test_split()))
