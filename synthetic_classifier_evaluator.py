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
from ctgan import CTGAN
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()


def generate_synthetic_samples(real_data, epochs=200, sample_size=10):
    """Generate synthetic samples using CTGAN."""
    data = real_data.reshape(-1, 1)
    ctgan = CTGAN(verbose=False)
    ctgan.fit(data, epochs=epochs)
    synthetic_data = ctgan.sample(sample_size)
    return synthetic_data


def prepare_mixed_dataset(real_data, synthetic_data):
    """Combine and label real and synthetic data for classification."""
    real_labels = np.zeros(len(real_data))
    synth_labels = np.ones(len(synthetic_data))
    real_df = pd.DataFrame(real_data)
    synth_df = pd.DataFrame(synthetic_data)
    real_labeled = pd.concat([real_df, pd.DataFrame(real_labels)], axis=1)
    synth_labeled = pd.concat([synth_df, pd.DataFrame(synth_labels)], axis=1)
    mixed_df = pd.concat([real_labeled, synth_labeled], axis=0).sample(frac=1).reset_index(drop=True)
    return mixed_df


def train_classifier(features, labels):
    """Train a simple neural network classifier."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = Sequential([
        Dense(32, activation='relu', input_dim=features_scaled.shape[1]),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(features_scaled, labels, epochs=5, batch_size=32, verbose=0)

    predictions = model.predict(features_scaled, verbose=0)
    propensity_score = np.round(np.mean((predictions - 0.5) ** 2), 4)
    return propensity_score, predictions, scaler


def plot_distribution(real_data, synth_data, propensity_score, plot_title, save_path=None):
    """Plot distribution of real vs synthetic samples with the propensity score."""
    plt.figure(figsize=(8, 5))
    plt.hist(real_data, bins=5, density=True, alpha=0.5, color='blue', label='Real')
    plt.hist(synth_data, bins=5, density=True, alpha=0.5, color='orange', label='Synthetic')
    plt.title(plot_title)
    plt.text(0.02, 0.85, f'Propensity = {propensity_score}', transform=plt.gca().transAxes)
    plt.xlabel("Tensile Strength")
    plt.legend(loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def main():
    # Load your dataset
    df = pd.read_csv("Experimental_data.txt", delimiter=',')
    Strength_range = [2000, 3000]

    for i in range(14):  # Loop through rows
        real_data = df.iloc[i, 0:10].values

        # Generate synthetic samples
        synthetic_data = generate_synthetic_samples(real_data)

        # Combine and label data
        mixed_df = prepare_mixed_dataset(real_data, synthetic_data)
        D = mixed_df.iloc[:, 0].values.reshape(-1, 1)
        L = mixed_df.iloc[:, 1].values

        # Train classifier and calculate propensity score
        p_score, probs, _ = train_classifier(D, L)
        print(f"Sample {i + 1} - Propensity Score: {p_score}")

        # Plot results
        plot_distribution(
            real_data,
            synthetic_data.values.flatten(),
            p_score,
            plot_title=f"Real vs Synthetic Distribution (Sample {i + 1})",
            save_path=f"{i + 1}_prob_dist_real_synth.pdf"
        )

        # Optional: Summary statistics
        print("Real Data Stats:\n", pd.Series(real_data).describe())
        print("Synthetic Data Stats:\n", synthetic_data.describe(), "\n")


if __name__ == "__main__":
    main()
