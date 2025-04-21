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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# ---------------------------------------------
# Outlier Detection & Characteristic Value Utils
# ---------------------------------------------

def critical_value(array):
    """Returns the Grubbs' test critical value for given sample size."""
    k_n_dict = {
        3: 3.37, 4: 2.63, 5: 2.33, 6: 2.18,
        7: 2.09, 8: 2.00, 9: 1.96, 10: 1.92
    }
    return k_n_dict.get(len(array), None)

def characteristic_value(array):
    """Computes characteristic value with correction based on critical value."""
    array = np.array(array)
    n = len(array)
    
    if n == 0:
        return 2000
    elif n == 1:
        return array[0]
    elif n == 2:
        return np.mean(array)
    
    mean = np.mean(array)
    std_dev = np.std(array)
    k_val = critical_value(array)

    if k_val is None or mean == 0:
        return mean  # fallback to mean
    
    return mean * (1 - (k_val * (std_dev / mean)))

def outlier_iqr(array, multiplier=1.5):
    """Removes outliers using the IQR method."""
    array = np.array(array)
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return array[(array >= lower) & (array <= upper)]

def outlier_t_score(array, critical_t=2.262, min_values=0):
    """Removes outliers based on t-statistics with a fallback to retain min_values."""
    array = np.array(array)
    mean = np.mean(array)
    std = np.std(array, ddof=1)
    t_scores = (array - mean) / (std / np.sqrt(len(array)))
    mask = np.abs(t_scores) <= critical_t

    while np.sum(mask) < min_values:
        critical_t += 0.1
        mask = np.abs(t_scores) <= critical_t

    return array[mask]

def outlier_autoencoder(array, threshold=0.5):
    """Removes outliers using a simple autoencoder."""
    array = np.array(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    train_data, _ = train_test_split(array, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)

    model = Sequential([
        Dense(10, activation='relu', input_shape=(array.shape[1],)),
        Dense(array.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_scaled, train_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)

    full_scaled = scaler.transform(array)
    reconstructed = model.predict(full_scaled, verbose=0)
    reconstruction_error = np.mean(np.abs(full_scaled - reconstructed), axis=1)

    return array[reconstruction_error <= threshold]

# ---------------------------------------------
# Main Data Processing & Visualization
# ---------------------------------------------

# Load Data
filename = "EC_data_10.txt"
df = pd.read_csv(filename, delimiter=',')
data = df.iloc[:, :10].to_numpy()

# Process characteristic values
original_vals, cleaned_vals = [], []

for row in data:
    original_vals.append(characteristic_value(row))
    cleaned = outlier_iqr(row)
    cleaned_vals.append(characteristic_value(cleaned))

char_vals = np.array(cleaned_vals)  # Switch between `original_vals` or `cleaned_vals`
flattened = data.flatten()
pop_mean = np.mean(flattened)
pop_std = np.std(flattened)
pop_char_val = round(pop_mean * (1 - 1.64 * (pop_std / pop_mean)), 2)

# Plot
char_sample_mean = round(np.mean(char_vals), 2)
mape = round(np.abs((pop_char_val - char_sample_mean) / pop_char_val) * 100, 2)

plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.kdeplot(data=char_vals, fill=True, color='blue')
plt.axvline(x=pop_char_val, linestyle='--', color='black',
            label=f"Population Characteristic Value\n= {pop_char_val}")
plt.axvline(x=char_sample_mean, linestyle='-', color='purple',
            label=f"Sample Mean Characteristic Value\n= {char_sample_mean}")
plt.text(x=0.65, y=0.75, s=f'MAPE = {mape}%', transform=plt.gca().transAxes)
plt.title('Probability Distribution of Characteristic Values After Outlier Removal', fontsize=14)
plt.xlabel('Characteristic Value')
plt.ylabel('')
plt.legend()
plt.tight_layout()
plt.show()
