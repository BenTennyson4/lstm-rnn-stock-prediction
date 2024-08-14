import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import random

# Set a random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load the dataset
url = 'https://raw.githubusercontent.com/BenTennyson4/stock-market-datasets/main/HistoricalData_1722206120177.csv'
df = pd.read_csv(url)

# Convert 'Date' column to datetime and sort the data by date
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Remove '$' from price columns and convert them to float
df = df.replace({'\$': '', ',': ''}, regex=True).astype(float)

# Normalize the features
feature_scaler = MinMaxScaler()
scaled_features = feature_scaler.fit_transform(df)

# Use separate scaler for the 'Close/Last' column
close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(df[['Close/Last']])

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Assuming we want to predict the 'Close/Last' price
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_features, seq_length)

# Fixed window size
window_size = 200  # Example fixed window size
validation_window_size = 10  # Example validation window size
n_splits = len(X) - window_size - validation_window_size

train_losses = []
val_losses = []
train_rmses = []
val_rmses = []

# Hyperparameters
hyperparams = {
    'LSTM Units': [50, 50],
    'Batch Size': 10,  # Increased batch size for better memory management
    'Epochs': 10,  # Adjusted number of epochs
    'Optimizer': 'adam',
    'Loss': 'mean_squared_error'
}

# Build the model once
model = Sequential()
model.add(LSTM(hyperparams['LSTM Units'][0], return_sequences=True, input_shape=(seq_length, X.shape[2])))
model.add(Dropout(0.3))  # Adding dropout for regularization
model.add(LSTM(hyperparams['LSTM Units'][1], return_sequences=False))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer=hyperparams['Optimizer'], loss=hyperparams['Loss'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

start_time = time.time()

for i in range(n_splits):
    X_train = X[i:i + window_size]
    y_train = y[i:i + window_size]
    X_val = X[i + window_size:i + window_size + validation_window_size]
    y_val = y[i + window_size:i + window_size + validation_window_size]

    history = model.fit(X_train, y_train, batch_size=hyperparams['Batch Size'], epochs=hyperparams['Epochs'], validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Append the losses
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

    # Calculate RMSE for the last epoch
    train_rmse = np.sqrt(history.history['loss'][-1])
    val_rmse = np.sqrt(history.history['val_loss'][-1])
    train_rmses.append(train_rmse)
    val_rmses.append(val_rmse)

    # Print progress
    if i % 10 == 0:
        elapsed_time = time.time() - start_time
        print(f"Iteration {i}/{n_splits}, elapsed time: {elapsed_time:.2f} seconds")

total_time = time.time() - start_time
print(f"Total time: {total_time:.2f} seconds")

# Flatten the loss lists
train_losses = [item for sublist in train_losses for item in sublist]
val_losses = [item for sublist in val_losses for item in sublist]

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Dev loss')
plt.title('Train and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation RMSE
plt.figure(figsize=(12, 6))
plt.plot(train_rmses, label='Train RMSE')
plt.plot(val_rmses, label='Dev RMSE')
plt.title('Train and validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# Average the predictions
fold_predictions = []

for i in range(n_splits):
    # Predict tomorrow's price
    last_sequence = scaled_features[-seq_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    tomorrow_pred = model.predict(last_sequence)
    tomorrow_pred_reshaped = tomorrow_pred.reshape(-1, 1)
    tomorrow_pred_price = close_scaler.inverse_transform(tomorrow_pred_reshaped)[0][0]
    fold_predictions.append(tomorrow_pred_price)

final_prediction = np.mean(fold_predictions)
print(f"Predicted price for tomorrow: ${final_prediction:.2f}")

# Evaluate the model using the last window
train_pred = model.predict(X_train)
test_pred = model.predict(X_val)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_val, test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_val, test_pred)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_val, test_pred)
train_evs = explained_variance_score(y_train, train_pred)
test_evs = explained_variance_score(y_val, test_pred)

print(f"\nPredicted price for tomorrow: ${final_prediction:.2f}")
print(f"Training MSE: {train_mse}")
print(f"Testing MSE: {test_mse}")
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")
print(f"Training MAE: {train_mae}")
print(f"Testing MAE: {test_mae}")
print(f"Training R²: {train_r2}")
print(f"Testing R²: {test_r2}")
print(f"Training Explained Variance Score: {train_evs}")
print(f"Testing Explained Variance Score: {test_evs}")

metrics = {
    'Training MSE': train_mse,
    'Testing MSE': test_mse,
    'Training RMSE': train_rmse,
    'Testing RMSE': test_rmse,
    'Training MAE': train_mae,
    'Testing MAE': test_mae,
    'Training R²': train_r2,
    'Testing R²': test_r2,
    'Training Explained Variance Score': train_evs,
    'Testing Explained Variance Score': test_evs
}

hyperparams_df = pd.DataFrame(list(hyperparams.items()), columns=['Hyperparameter', 'Value'])
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])

print("\nHyperparameters:")
print(hyperparams_df)
print("\nMetrics:")
print(metrics_df)

# Tabulate and print all metrics
train_metrics = []
val_metrics = []

for i in range(n_splits):
    X_train = X[i:i + window_size]
    y_train = y[i:i + window_size]
    X_val = X[i + window_size:i + window_size + validation_window_size]
    y_val = y[i + window_size:i + window_size + validation_window_size]

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_metrics.append([
        mean_squared_error(y_train, train_pred),
        np.sqrt(mean_squared_error(y_train, train_pred)),
        mean_absolute_error(y_train, train_pred),
        r2_score(y_train, train_pred),
        explained_variance_score(y_train, train_pred)
    ])

    val_metrics.append([
        mean_squared_error(y_val, val_pred),
        np.sqrt(mean_squared_error(y_val, val_pred)),
        mean_absolute_error(y_val, val_pred),
        r2_score(y_val, val_pred),
        explained_variance_score(y_val, val_pred)
    ])

train_metrics_df = pd.DataFrame(train_metrics, columns=['MSE', 'RMSE', 'MAE', 'R²', 'EVS'])
val_metrics_df = pd.DataFrame(val_metrics, columns=['MSE', 'RMSE', 'MAE', 'R²', 'EVS'])

print("\nTrain Metrics per Split:")
print(train_metrics_df.describe())
print("\nValidation Metrics per Split:")
print(val_metrics_df.describe())
