import pdb
import csv
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format="retina"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import random
import torch
from torch import nn, optim
import math
from IPython import display

def convert_features_to_one_hot(df, feature_name_list):
    for feature_name in feature_name_list:
        df = pd.get_dummies(df, columns=[feature_name])

        return df
    

def save_prediction_to_csv_file(y_test_pred, file_name):
  # Convert predicted churn values to 'yes' and 'no'
  churn_list = ['no' if pred == 0 else 'yes' for pred in y_test_pred]

  # Create a DataFrame containing the IDs and predicted churn values
  submission_df = pd.DataFrame({'id': test_df['id'], 'churn': churn_list})

  # Save the DataFrame to a CSV file
  submission_df.to_csv(file_name, index=False)

# Show first five elements in the input
def show_first_five_elements(inp):
    print(list(inp)[:5])

if __name__ == "__main__":

    # Linear Regression
    seed = 1
    random.seed(seed)

    lst = df = pd.read_csv('./Customer_Churn.csv')
    print(type(lst))

    train_df, test_df = train_test_split(lst, test_size=None, train_size=None)
    #train_df, test_df = train_test_split(lst, test_size=None, train_size=None)

    print(train_df.head())
    print(train_df.describe())
    print("hi")
    
    torch.manual_seed(seed)

    train_df_index = train_df.index
    print("Initial index of train_df:", show_first_five_elements(train_df_index))

    shuffled_index = np.random.permutation(train_df.index)
    print("Shuffled index:", show_first_five_elements(shuffled_index))

    train_df = train_df.reindex(shuffled_index)
    print("Shuffled index of train_df:", show_first_five_elements(train_df.index))

    print("\nThe examples in train_df are shuffled:")
    train_df.head()

    
    # select features for 
    X = train_df[['COLLEGE','INCOME','OVERAGE','LEFTOVER','HOUSE','HANDSET_PRICE','OVER_15MINS_CALLS_PER_MONTH',
                  'AVERAGE_CALL_DURATION','REPORTED_SATISFACTION','REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']]
    Y = train_df['LEAVE']

    X_test = test_df[['COLLEGE','INCOME','OVERAGE','LEFTOVER','HOUSE','HANDSET_PRICE','OVER_15MINS_CALLS_PER_MONTH',
                  'AVERAGE_CALL_DURATION','REPORTED_SATISFACTION','REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']]

    test_id = test_df['LEAVE']

    print(f"Count of missing values in the train_df: {train_df.isnull().sum().sum()}")
    print(f"Count of missing values in the test_df: {test_df.isnull().sum().sum()}")

    ###### Your codes start here.######
    #Y = Y.replace({'STAY': 0, 'LEAVE': 1})
    Y = Y.map({'STAY': 0, 'LEAVE': 1}).astype(int)

    ###### Your codes end here.######

    # Convert Y DataFrame to integer type
    Y = Y.astype(int)

    ###### Your codes start here.######
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=0)
    # since test_size=None, train_size=None not specified, sets test_size to 0.25 and train size to 0.75
    # these numbers must be between 0 and 1
    # to make different sizes for learning curve I could just manually truncate X train and X val later rather than messing with this
    ###### Your codes end here.######

    print(X_train.head())
    print(X_val.head())

    print(Y_train.head())
    print(Y_val.head())


    # List of categorical features to be one-hot encoded
    ###### Your codes start here.######
    # categorical_features = ['COLLEGE', 'REPORTED_SATISFACTION','REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN','LEAVE']

    # X_train = convert_features_to_one_hot(X_train, categorical_features)
    # X_val = convert_features_to_one_hot(X_val, categorical_features)
    # X_test = convert_features_to_one_hot(X_test, categorical_features)

    # Combine for consistent one-hot encoding
    full_df = pd.concat([X_train, X_val, X_test], axis=0)

    # One-hot encode categorical columns all at once
    categorical_features = ['COLLEGE', 'REPORTED_SATISFACTION','REPORTED_USAGE_LEVEL','CONSIDERING_CHANGE_OF_PLAN']
    full_df_encoded = pd.get_dummies(full_df, columns=categorical_features)

    # Split back into original sets
    X_train_encoded = full_df_encoded.iloc[:len(X_train)]
    X_val_encoded = full_df_encoded.iloc[len(X_train):len(X_train)+len(X_val)]
    X_test_encoded = full_df_encoded.iloc[len(X_train)+len(X_val):]

    ###### Your codes end here.######

    X_train.head()

    ###### Your codes start here.######
    # LR_model = LogisticRegression(penalty="l1", n_jobs=-1).fit(X_train, X_val)
    LR_model = LogisticRegression(penalty="l1", solver='liblinear', n_jobs=-1).fit(X_train_encoded, Y_train)

    # For penalties try: l1, l2, and elasticnet (which is both l1 and l2)
    # Fit is the test variables you want the model to learn
    ###### Your codes end here.######

    ###### Your codes start here.######
    #y_val_pred = LR_model.predict(Y_train)
    y_val_pred = LR_model.predict(X_val_encoded)

    print(f"Logistic regression model validation accuracy: {np.sum(y_val_pred == Y_val)/len(Y_val)}")
    ###### Your codes end here.######

    # Testing model with Y_train and Y_val (part 2 of training set)
    # This is where you could make a learning curve, by trying this process with different sizes of X_train
    # (model complexity graph doesn't apply here cause theres only one model)

    y_test_pred = LR_model.predict(X_test)
    save_prediction_to_csv_file(y_test_pred, "submission_lr.csv")
    

    # Neural Network

    # Convert pandas DataFrame to PyTorch tensor
    X_train_tensor = torch.tensor(X_train_encoded.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val_encoded.values, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_encoded.values, dtype=torch.float32)

    # Define Neural Network architecture
    class ChurnNN(nn.Module):
        def __init__(self, input_dim):
            super(ChurnNN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    input_dim = X_train_tensor.shape[1]
    model = ChurnNN(input_dim)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Neural Network
    num_epochs = 100
    batch_size = 32

    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])

        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_Y = X_train_tensor[indices], Y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_predictions = (val_outputs >= 0.5).float()
                val_accuracy = (val_predictions.eq(Y_val_tensor).sum().item()) / len(Y_val_tensor)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Validation predictions
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_predictions = (val_outputs >= 0.5).float().squeeze().numpy()

    print(f"Neural network model validation accuracy: {np.sum(val_predictions == Y_val.values) / len(Y_val):.4f}")

    # Test predictions
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs >= 0.5).float().squeeze().numpy()

    save_prediction_to_csv_file(test_predictions, "submission_nn.csv")

    ###### Your codes end here.######


