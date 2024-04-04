# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn.preprocessing as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# Define the GRU class
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward_pass(self, tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)  # Add sequence length dimension
        
        # Pass through GRU layer
        out, _ = self.gru(tensor)
        
        # Flatten the output tensor to 2D
        out = out[:, -1, :]  # Consider only the last time step
        out = out.contiguous().view(out.size(0), -1)  # Flatten
        
        out = self.fc(out) 
        return out

# Load and preprocess data
csv_file_path = r"C:\Users\dcmar\stockr.ai\sp500_stocks.csv"
df = pd.read_csv(csv_file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'].dt.year >= 2024]
df = df[['Date', 'Symbol', 'Close']]
price = df[['Close']]

# Use min-max scaling
scaler = sk.MinMaxScaler(feature_range=(-1, 1))
price.loc[:, 'Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))

# Partition data
def partition_data(df, train_ratio=0.7, test_ratio=0.15):
    df = df.sample(frac=1).reset_index(drop=True)
    num_samples = len(df)
    num_train = int(train_ratio * num_samples)
    num_test = int(test_ratio * num_samples)
    train_set = df.iloc[:num_train]
    test_set = df.iloc[num_train:num_train + num_test]
    val_set = df.iloc[num_train + num_test:]
    return train_set, test_set, val_set

train_set, test_set, val_set = partition_data(df, train_ratio=0.7, test_ratio=0.15)

# Convert data to tensors
def df_to_tensor(df):
    x = torch.tensor(df.drop(columns=['Date', 'Symbol']).values, dtype=torch.float32)
    y = torch.tensor(df['Close'].values, dtype=torch.float32)
    return x, y

x_train, y_train = df_to_tensor(train_set)
x_test, y_test = df_to_tensor(test_set)
x_val, y_val = df_to_tensor(val_set)

# Define a function to train and evaluate the model
def train_and_evaluate_model(model, criterion, optimizer, x_train, y_train, x_val, y_val, num_epochs):
    for epoch in range(10):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred = model.forward_pass(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model.forward_pass(x_val)
            val_loss = criterion(y_val_pred, y_val)
        
        # Print training and validation loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Define a grid of hyperparameters to search over
param_grid = {
    'hidden_dim': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.01, 0.1],
}

best_model = None
best_score = float('inf')

# Iterate over hyperparameter combinations
for params in ParameterGrid(param_grid):
    # Initialize model with current hyperparameters
    model = GRU(input_dim=1, hidden_dim=params['hidden_dim'], num_layers=params['num_layers'], output_dim=1)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Train and evaluate the model
    train_and_evaluate_model(model, criterion, optimizer, x_train, y_train, x_val, y_val, num_epochs=100)
    
    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        y_val_pred = model.forward_pass(x_val)
        val_loss = criterion(y_val_pred, y_val)
    
    # Update best model if current model performs better
    if val_loss < best_score:
        best_model = model
        best_score = val_loss

# Use the best model for prediction
best_model.eval()
with torch.no_grad():
    y_pred = best_model.forward_pass(x_test)

# Convert predictions to numpy array and inverse transform
y_pred_np = y_pred.numpy()
y_pred_inv = scaler.inverse_transform(y_pred_np.reshape(-1, 1))

# Convert test set to numpy array
y_test_np = y_test.numpy()

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_pred_inv, label='Predicted')
plt.plot(y_test_np, label='Actual')
plt.title('Predictions vs Actual Values')
plt.xlabel('Sample Index')
plt.ylabel('Close Price')
plt.legend()
plt.show()
