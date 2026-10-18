import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from src.features import normalize_landmarks

# 1. Mock classes to convert CSV rows back into MediaPipe-style objects
class MockPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockLandmarks:
    def __init__(self, points):
        self.landmark = points

print("Loading and normalizing data...")
df = pd.read_csv('data/landmarks.csv')

X = []
y = df['label'].values

# 2. Extract and normalize every row
for index, row in df.iterrows():
    points = []
    for i in range(21):
        points.append(MockPoint(row[f'x{i}'], row[f'y{i}'], row[f'z{i}']))
        
    mock_landmarks = MockLandmarks(points)
    normalized_vector = normalize_landmarks(mock_landmarks)
    X.append(normalized_vector)

X = np.array(X)

# 3. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n--- MACHINE LEARNING RESULTS ---")
print(f"Model Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
labels = sorted(list(set(y)))
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# 6. Save the trained model
if not os.path.exists('models'):
    os.makedirs('models')
    
joblib.dump(model, 'models/gesture_model.pkl')
print("\nModel saved successfully to models/gesture_model.pkl")