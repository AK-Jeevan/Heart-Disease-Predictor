# This model predicts the presence of heart disease based on clinical features like age, cholesterol, blood pressure, and more.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Load and clean data
data = pd.read_csv("C://Users//akjee//Documents//ML//heart_disease_uci.csv")
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
print("Data Size:", data.shape)
print(data.head(10))

# Encode categorical variables (if any)
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.iloc[:, :-1]  # Independent variables
y = data.iloc[:, -1]   # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f"X Train shape is :{X_train.shape}")
print(f"X Test shape is :{X_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

# Feature scaling (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
classifier = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = classifier.predict(X_test_scaled)
print("Predictions:", y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model Accuracy:", classifier.score(X_test_scaled, y_test))