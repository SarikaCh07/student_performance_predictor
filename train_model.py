# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'past_grade': [45, 50, 55, 60, 62, 66, 70, 75, 80, 90],
    'participation': [1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    'passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Features and target
X = df[['study_hours', 'attendance', 'past_grade', 'participation']]
y = df['passed']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
