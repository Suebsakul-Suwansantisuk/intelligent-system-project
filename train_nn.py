import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# โหลดข้อมูล
df = pd.read_csv('diabetes.csv')

X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# สร้าง Neural Network
model_nn = MLPClassifier(
    hidden_layer_sizes=(16, 8),  # 2 hidden layers
    activation='relu',
    max_iter=200,
    random_state=42
)

model_nn.fit(X_train, y_train)

# ประเมินผล
y_pred = model_nn.predict(X_test)
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model_nn, 'nn_model.pkl')
joblib.dump(scaler, 'scaler_nn.pkl')
print("Save สำเร็จ!")