import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Step 1: Set random seed
tf.keras.utils.set_random_seed(42)

# Step 2: Load Sunspots dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"

df = pd.read_csv(url, usecols=[1])
data = df.values.astype("float32")

plt.plot(data[:200])
plt.title("Monthly Sunspots")
plt.show()

# Step 3: Create sequences
def create_sequences(data, length):
    X, y = [], []

    for i in range(len(data) - length):
        X.append(data[i:i + length])
        y.append(data[i + length])

    return np.array(X), np.array(y)

sequence_length = 20

X, y = create_sequences(data[:500], sequence_length)

# Step 4: Reshape input
X = X.reshape(X.shape[0], X.shape[1], 1)

print("Input shape:", X.shape)
print("Output shape:", y.shape)

# Step 5: Split data
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 6: Build LSTM model
model = Sequential([
    Input(shape=(20, 1)),
    LSTM(32),
    Dense(1)
])

# Step 7: Compile model
model.compile(optimizer="adam", loss="mse")
model.summary()

# Step 8: Train model
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1
)

# Evaluate model
predictions = model.predict(X_test)

loss = model.evaluate(X_test, y_test, verbose=0)

print("Test MSE:", round(loss, 4))

# Step 9: Plot actual vs predicted values
plt.figure(figsize=(10, 5))

plt.plot(y_test[:100], label="Actual")
plt.plot(predictions[:100], label="Predicted", linestyle="dashed")

plt.title("LSTM Sunspots Prediction")
plt.xlabel("Time Step")
plt.ylabel("Sunspot Count")
plt.legend()
plt.show()

# Step 10: Plot training loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
