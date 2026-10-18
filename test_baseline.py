import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.rules import rule_based_classifier

# 1. Create Mock classes so the CSV data looks like MediaPipe objects
class MockPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MockLandmarks:
    def __init__(self, points):
        self.landmark = points

# 2. Load the dataset
print("Loading data...")
try:
    df = pd.read_csv('data/landmarks.csv')
except FileNotFoundError:
    print("Error: data/landmarks.csv not found. Run collect.py first.")
    sys.exit()

y_true = df['label'].values
y_pred = []

# 3. Iterate through every row and make a prediction
print("Running rule-based classifier...")
for index, row in df.iterrows():
    points = []
    for i in range(21):
        x = row[f'x{i}']
        y = row[f'y{i}']
        points.append(MockPoint(x, y))
        
    mock_landmarks = MockLandmarks(points)
    prediction = rule_based_classifier(mock_landmarks)
    y_pred.append(prediction)

# 4. Print the Report Card
print("\n--- BASELINE RESULTS ---")
acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("Confusion Matrix:")
labels = sorted(list(set(y_true)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)