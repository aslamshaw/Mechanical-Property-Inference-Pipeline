# Copyright 2025 Anvar Mohamed Aslam Sha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Required Libraries
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm

sns.set_theme()

# Utility Function to Calculate Metrics
def calculate_metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred, multioutput='raw_values'),
        'MAE': mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values'),
        'R2': r2_score(y_true, y_pred, multioutput='raw_values')
    }

# Load Experimental Data
df = pd.read_csv("Experimental_data.txt", delimiter=',')
sample_sizes = np.array([3, 4, 5, 6, 7, 8, 9, 10])
k_n = np.array([3.37, 2.63, 2.33, 2.18, 2.09, 2.00, 1.96, 1.92])

# Preprocess Data for a Single Sample (e.g., row 12)
series_index = 12
mean, cv, x_k = [], [], []

for size, k in zip(sample_sizes, k_n):
    values = df.iloc[series_index, :size]
    m, std = np.mean(values), np.std(values)
    coeff_var = std / m
    mean.append(m)
    cv.append(coeff_var)
    x_k.append(m * (1 - k * coeff_var))

# Combine into a DataFrame
data = pd.DataFrame({"Size": sample_sizes, "Mean": mean, "CV": cv, "X_k": x_k})

# Extend with Custom Sample (e.g., adding 2 new values)
new_sample = np.array(df.iloc[0, :10])
inserted = np.append(new_sample[:-1], np.random.randint(2000, 3000, 2))
new_mean = np.mean(inserted)
new_cv = np.std(inserted) / new_mean
new_xk = new_mean * (1 - 1.92 * new_cv)
extra = pd.DataFrame([[11, new_mean, new_cv, new_xk]], columns=data.columns)
data = pd.concat([data, extra], ignore_index=True)

# 1D Regression Model
def characteristic_model(x, a, b):
    return a * np.power(x, b)

x_data = data["Size"].values
y_data = data["X_k"].values
params, _ = curve_fit(characteristic_model, x_data, y_data)

def plot_target_vs_prediction(true, pred, title):
    plt.figure()
    sns.scatterplot(x=true, y=pred)
    plt.plot([min(true), max(true)], [min(true), max(true)], '--k')
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.grid(True)
    plt.show()

# 2D Linear Regression
def two_input_model(inputs, a, b, c):
    return a * inputs[0] + b * inputs[1] + c

x_2d = data[["Size", "Mean"]].values.T
y_2d = data["X_k"].values
params_2d, _ = curve_fit(two_input_model, x_2d, y_2d)

# Neural Network for 1D Data
class TQDMSingleLineProgressBar(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.tqdm = tqdm(total=self.params['epochs'], leave=True)
    def on_epoch_end(self, epoch, logs=None):
        self.tqdm.update(1)
    def on_train_end(self, logs=None):
        self.tqdm.close()

X = data[["Size"]].values
Y = data[["X_k"]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train model
model = Sequential([
    Input(shape=(1,)),
    Dense(100, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train_scaled, Y_train, epochs=1000, verbose=0, callbacks=[TQDMSingleLineProgressBar()])

# Plot Loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()

# Final Prediction Metrics
y_pred_test = model.predict(X_test_scaled)
metrics = calculate_metrics(Y_test, y_pred_test)
print("Test Metrics:", metrics)
plot_target_vs_prediction(Y_test.flatten(), y_pred_test.flatten(), "1D Neural Network Prediction")
