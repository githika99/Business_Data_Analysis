import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
from torch import nn, optim

# Load dataset
df = pd.read_csv('./Customer_Churn.csv')

# Encode target variable
df['LEAVE'] = df['LEAVE'].map({'STAY': 0, 'LEAVE': 1}).astype(int)

# Features and target
X = df[['COLLEGE','INCOME','OVERAGE','LEFTOVER','HOUSE','HANDSET_PRICE',
        'OVER_15MINS_CALLS_PER_MONTH','AVERAGE_CALL_DURATION',
        'REPORTED_SATISFACTION','REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']]
Y = df['LEAVE']

# One-hot encode categorical features
categorical_features = ['COLLEGE', 'REPORTED_SATISFACTION','REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
X = pd.get_dummies(X, columns=categorical_features)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE (for neural net later)
sm = SMOTE(random_state=42)
X_train_sm, Y_train_sm = sm.fit_resample(X_train_scaled, Y_train)

### Logistic Regression ###
param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
lr = LogisticRegression(solver='liblinear', class_weight='balanced')
grid_lr = GridSearchCV(lr, param_grid, scoring='roc_auc', cv=5)
grid_lr.fit(X_train_scaled, Y_train)

y_pred_lr = grid_lr.predict(X_test_scaled)

print("\n--- Logistic Regression Results ---")
print(f"Best Parameters: {grid_lr.best_params_}")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_lr):.3f}")
print(f"AUC-ROC: {roc_auc_score(Y_test, y_pred_lr):.3f}")
print(f"F1-score: {f1_score(Y_test, y_pred_lr):.3f}")

### Decision Tree ###
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train, Y_train)
y_pred_dt = dt.predict(X_test)

print("\n--- Decision Tree Results ---")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_dt):.3f}")
print(f"AUC-ROC: {roc_auc_score(Y_test, y_pred_dt):.3f}")
print(f"F1-score: {f1_score(Y_test, y_pred_dt):.3f}")

### KNN ###
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, Y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\n--- KNN Results ---")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_knn):.3f}")
print(f"AUC-ROC: {roc_auc_score(Y_test, y_pred_knn):.3f}")
print(f"F1-score: {f1_score(Y_test, y_pred_knn):.3f}")

### Neural Network (Different Architecture) ###
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Convert to tensors
X_train_tensor = torch.tensor(X_train_sm, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_sm.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Model, loss, optimizer
model = Net(X_train_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

# Prediction
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    y_pred_nn = (outputs.numpy() > 0.5).astype(int)

print("\n--- Neural Network Results ---")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_nn):.3f}")
print(f"AUC-ROC: {roc_auc_score(Y_test, y_pred_nn):.3f}")
print(f"F1-score: {f1_score(Y_test, y_pred_nn):.3f}")
