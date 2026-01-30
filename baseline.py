# --- import
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Data Load & Check Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

train.info()

# --- Pre-Processing
missing_ratio = train.isnull().mean()
columns_to_drop = missing_ratio[missing_ratio > 0.8].index.tolist()

train = train.drop(columns=columns_to_drop, axis=1)
test = test.drop(columns=columns_to_drop, axis=1)

train.head(2)

missing_cols = train.columns[train.isnull().any()].tolist()

for col in missing_cols:
    mode_value = train[col].mode()[0]
    train[col] = train[col].fillna(mode_value)
    test[col] = test[col].fillna(mode_value)

train['is_major_it'] = train['major_field'].str.contains('IT', regex=True).astype(int)
test['is_major_it'] = test['major_field'].str.contains('IT', regex=True).astype(int)

object_columns = train.select_dtypes(['object', 'bool']).columns

label_encoders = {}
for col in object_columns:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    
    le = LabelEncoder()
    le = le.fit(train[col])
    train[col] = le.transform(train[col])
    
    for label in np.unique(test[col]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# --- EDA
plt.figure(figsize=(20,14))
sns.heatmap(train.corr().round(2), annot = True)

# --- Feature Selection 
X_train = train[['class1', 're_registration', 'inflow_route', 'time_input','is_major_it']]
y_train = train['completed']

X_test = test[['class1', 're_registration', 'inflow_route', 'time_input','is_major_it']]

# --- Modeling 
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Inference 
pred = model.predict(X_test)

# --- Submission
submission = pd.read_csv('sample_submission.csv')
submission['completed'] = pred
submission.to_csv('submit.csv', index = False)