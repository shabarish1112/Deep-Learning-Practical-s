import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Step 1: Load dataset
tf.keras.utils.set_random_seed(42)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

df = pd.read_csv(url, usecols=[1])
data = df.values.astype("float32")

print("Dataset shape:", data.shape)

# Step 2: Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

# Step 3: Split data
train_size = int(len(dataset) * 0.70)

train_data = dataset[:train_size]
test_data = dataset[train_size:]

# Step 4: Create sequences
def create_dataset(data, look_back):
    X, y = [], []

    for i in range(len(data) - look_back - 1):
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])

    return np.array(X), np.array(y)

look_back = 3

X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

X_train = X_train.reshape(-1, look_back, 1)
X_test = X_test.reshape(-1, look_back, 1)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Step 5: Build LSTM model
model = Sequential([
    Input(shape=(look_back, 1)),
    LSTM(64, activation="tanh"),
    Dense(1)
])

# Step 6: Compile model
model.compile(optimizer="adam", loss="mse")
model.summary()

# Step 7: Train model
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    verbose=1
)

# Step 8: Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Step 9: Convert predictions to original values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 10: Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

print("\nTrain RMSE:", round(train_rmse, 2))
print("Test RMSE:", round(test_rmse, 2))

# Plot predictions
train_plot = np.full_like(data, np.nan)
test_plot = np.full_like(data, np.nan)

train_plot[look_back:look_back + len(train_predict)] = train_predict

test_start = train_size + look_back
test_plot[test_start:test_start + len(test_predict)] = test_predict

plt.figure(figsize=(10, 5))

plt.plot(data, label="Actual")
plt.plot(train_plot, label="Training Prediction")
plt.plot(test_plot, label="Testing Prediction")

plt.title("Airline Passenger Forecasting using LSTM")
plt.xlabel("Months")
plt.ylabel("Passengers")
plt.legend()
plt.show()
