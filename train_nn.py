import pandas as pd
import numpy as np

# ── 1. โหลดข้อมูล ──────────────────────────────────────────
data = pd.read_csv('diabetes.csv')
print(data.head(15))
print(data['Outcome'].value_counts())

print(data.dtypes)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = data.drop(columns=['Outcome'])
y = data['Outcome']  # Binary: 0 = No, 1 = Yes

# ── 3. Scale ───────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ── 4. แบ่ง train/test ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dropout(0.2),         # ป้องกัน overfit
    Dense(8,  activation='relu'),
    Dropout(0.2),
    Dense(1,  activation='sigmoid')  # Binary → sigmoid
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Binary → binary_crossentropy
    metrics=['accuracy']
)

model.summary()

# ── 6. Train ────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ── 7. ประเมินผล ────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# classification report
from sklearn.metrics import classification_report, confusion_matrix
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # threshold 0.5

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Diabetes', 'Diabetes']))

# ── 10. Save model + scaler ─────────────────────────────────
import joblib
model.save('nn_model.h5')
joblib.dump(scaler, 'scaler_nn.pkl')
print("Model และ scaler ถูกบันทึกเรียบร้อยแล้ว!")