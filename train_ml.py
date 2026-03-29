import numpy as np
import pandas as pd

data = pd.read_csv('student_depression_dataset.csv')
print(data.head(15))
print(data.dtypes)
print(data.isnull().sum())
print(f"จำนวน rows: {len(data)}")
print(f"จำนวน columns: {len(data.columns)}")

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
col_en = ['Gender','City','Profession','Sleep Duration','Dietary Habits','Degree','Have you ever had suicidal thoughts ?','Financial Stress','Family History of Mental Illness'] #column for encode
for col in col_en:
  data[col] = le.fit_transform(data[col])


print(data.head(15))
print(data.dtypes)

X = data.drop(columns = ['id','Depression'])
Y = data['Depression']

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.pipeline import Pipeline

RandomForest_pipe = Pipeline([
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

LogisticRegression_pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('lr',LogisticRegression(C=1.0, max_iter=1000, random_state = 42))
])

KNN_pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('knn',KNeighborsClassifier(n_neighbors=5))
])

ensemble = VotingClassifier(estimators = [
    ('rf',RandomForest_pipe),
    ('lr',LogisticRegression_pipe),
    ('knn',KNN_pipe)
],voting = 'hard')

ensemble.fit(X_train,Y_train)

from sklearn.metrics import classification_report,confusion_matrix

y_prediction = ensemble.predict(X_test)

cm = confusion_matrix(Y_test,y_prediction)
print(cm)
print(classification_report(Y_test,y_prediction))

import joblib
joblib.dump(ensemble, 'model_ml.pkl')
print("save สำเร็จ!")