# Machine Learning for Determining Characteristic Values Based on Material-Specific Tests

This repository contains the complete documentation and implementation of my **Master’s Thesis** work at **TU Dresden**, as part of the Master's program in *Advanced Computational and Civil Engineering Structural Studies*. The core focus is on applying **machine learning methods** to predict **characteristic material values** (such as the 5th percentile of tensile strength) from **material-specific experimental tests**, with an emphasis on **robust outlier detection** and **data augmentation** using synthetic data.

---

## 📌 Overview

Traditional methods for calculating characteristic values from material tests—especially when the sample size is small—are prone to **biases and statistical fragility**. This thesis introduces a **data-driven approach**, leveraging modern machine learning techniques to:

- **Detect and remove outliers** from small experimental datasets
- **Generate synthetic material test samples** using CTGANs
- **Predict characteristic values** such as tensile strength using regression models

---

## 🧪 Problem Background

In civil and materials engineering, characteristic values like **5th percentile tensile strength** are crucial for design safety and material certification. These values are traditionally derived from standardized statistical approaches assuming normal or log-normal data distributions.

However, in many real-world scenarios:
- Only **a handful of test samples** (e.g., 5 to 20) are available.
- The data may contain **outliers** due to human error, instrument noise, or rare structural anomalies.
- **Over-reliance on assumptions** about distributions can lead to incorrect estimates of characteristic values.

This thesis addresses these challenges by integrating **outlier filtering** and **data augmentation** into a **machine learning pipeline** for robust prediction.

---

## 🎯 Objective

The main objectives of this thesis are:

1. **Detect and eliminate outliers** in small test datasets using robust methods
2. **Augment data** using Conditional Tabular GANs (CTGAN) to simulate realistic synthetic material tests
3. **Train regression models** (e.g., neural networks or other simple regressors) to predict characteristic values
4. **Evaluate the impact of outlier filtering and synthetic data on model accuracy**

---

## 🧰 Methodology

### 1. Outlier Detection (Core Focus of This Work)

Outlier detection formed the **cornerstone of the preprocessing pipeline** in this thesis. Given the small size of experimental configurations and the high sensitivity of machine learning models to data irregularities, multiple complementary strategies were implemented to identify and remove outliers in a robust and justifiable way.

The following methods were used in combination to detect outliers:

---

#### 1.1 Interquartile Range (IQR) Method

The **IQR method** was applied as a first-pass statistical filter.

- For each set of tensile strength measurements (typically 10 values):
  - Calculate **Q1** (25th percentile) and **Q3** (75th percentile)
  - Compute the **IQR**: `IQR = Q3 - Q1`
  - Identify outliers as:
    ```text
    Values < Q1 - 1.5 × IQR  
    or  
    Values > Q3 + 1.5 × IQR
    ```

This method works well for skewed distributions and small datasets, and allowed removal of extreme values without assuming normality.

---

#### 1.2 Z-Score Method

To detect **extreme deviations from the mean**, a **Z-score approach** was also employed:

- For each configuration:
  - Compute mean `μ` and standard deviation `σ`
  - Calculate Z-score for each value:  
    `Z = (x - μ) / σ`
  - Values with |Z| > 2.5 or 3 were flagged as outliers.

This method is sensitive to symmetric deviation and complements the IQR method, especially in near-normal distributions.

---

#### 1.3 Autoencoder-Based Outlier Detection

To detect **non-linear, high-dimensional outliers**, a **neural network–based autoencoder** was trained:

- Architecture: Small, dense autoencoder
- Input: Vector of 10 tensile strength values per configuration
- Training:
  - The autoencoder learns to reconstruct normal configurations
  - Reconstruction error (e.g., MSE) is computed for each input
- Configurations with **high reconstruction error** were considered **outliers**, as the network failed to compress/reconstruct them accurately.

This approach enables detection of **contextual anomalies**—those that might appear statistically valid but deviate from learned patterns across the dataset.

---

#### 1.4 Visual Statistical Inspection

All data subsets were also inspected **manually** using plots:

- **Boxplots**: Revealed spread, skewness, and whisker outliers
- **Histograms**: Helped spot multi-modal behavior and asymmetry
- **Line plots of raw values**: Flagged unusually fluctuating test runs

This allowed for:
- Validation of automated outlier detection
- Correction of false positives/negatives
- Holistic understanding of data variability

---

#### 1.5 Domain Knowledge–Based Filtering

Statistical tools were supported by **engineering judgment**:

- **Threshold filtering**:
  - Tensile strength values below ~1000 MPa or above ~7000 MPa were discarded
- **Homogeneity flag**:
  - Configurations with **no variance** or near-identical values were removed
- **Variance-to-mean check**:
  - If standard deviation was disproportionately large (> 50% of the mean), the case was flagged for instability

This step filtered out **physically implausible data**, regardless of statistical normality.

---

#### 1.6 Post-Filtering Re-Validation

After all filtering steps:

- The cleaned data was **re-evaluated for distribution quality**:
  - Smooth unimodal histograms
  - Balanced variance
  - Retention of physical meaning
- Outlier removal was verified to avoid **biasing the population statistics**

---

### Summary

| Method                    | Detects                                     | Assumptions        | Strengths                                 |
|---------------------------|---------------------------------------------|--------------------|--------------------------------------------|
| IQR                       | Univariate outliers                         | None               | Simple, robust, non-parametric             |
| Z-Score                   | Extreme deviations (normal-like data)       | Approx. normal     | Easy to implement, interpretable           |
| Autoencoder               | Multivariate, contextual anomalies          | Learned patterns   | Captures subtle nonlinear relationships    |
| Visual Inspection         | Skewed, multi-modal, or spurious data       | Human judgment     | Qualitative verification                   |
| Domain Knowledge Filters  | Physically implausible or corrupt readings  | Expert knowledge   | Ensures physical realism                   |

> **Conclusion**: The combination of statistical, deep learning, visual, and expert-guided techniques ensured a high-confidence, well-behaved dataset for the next stages of modeling and synthetic data generation.


---

### 2. Synthetic Data Generation with CTGAN

#### **Overview**:
In cases where the data is limited (i.e., fewer than 10-20 test samples), the model utilizes **CTGANs** (Conditional Generative Adversarial Networks) to generate realistic **synthetic data**. This synthetic data can then be used to augment the training dataset, helping the model generalize better and avoid overfitting.

#### Why CTGAN?
- Designed specifically for **tabular data**
- Handles both **continuous** and **categorical** variables
- Learns complex feature correlations even in **imbalanced datasets**

#### **Data Generation Process**:
- **Training CTGAN**: The CTGAN is trained on the available **real test data** (e.g., tensile strength test results) to learn the underlying distribution and characteristics.
- **Synthetic data generation**: Once trained, the CTGAN generates synthetic test data that follows the same statistical distribution as the real data but with variations to improve generalization.
- **Data Augmentation**: The synthetic data is then combined with the real data to create an **augmented training dataset**.

#### **Importance of Synthetic Data**:
- **Data augmentation**: By using synthetic data, the model has more examples to learn from, improving its performance.
- **Avoiding overfitting**: Synthetic data helps to avoid overfitting, which occurs when the model learns too much from a small, possibly unrepresentative sample.
- **Enhanced generalization**: With more diverse data, the model can make predictions that are more robust to variations in new, unseen material test data.

#### Implementation Highlights:
- **Input**: Cleaned real material test values (e.g., tensile strength)
- **Epochs & Batch size**: Tuned to avoid overfitting or mode collapse
- **Conditional columns**: Not used in this case since all features were continuous
- **Output**: Synthetic dataset of the same dimensionality and similar distribution

---

#### 📊 Evaluation Using Propensity Score Matching

To assess how **realistic** the synthetic data is, a **propensity score classifier** was trained.

##### What is a propensity score in this context?
It is the predicted **probability that a sample belongs to the real vs. synthetic group**, estimated using a classifier (e.g., logistic regression or a shallow neural network). If the classifier **cannot distinguish** between the two, the synthetic data is considered **indistinguishable from real data**—a desirable outcome.

##### Steps:
1. **Label** real and synthetic data (e.g., 0 for real, 1 for synthetic)
2. **Train classifier** (e.g., logistic regression or MLP) on this mixed dataset
3. **Evaluate output probabilities**:
   - If predictions are close to **0.5 for all samples**, it indicates **no significant separation** between real and fake → synthetic data is realistic
   - If classifier achieves high accuracy (i.e., >70%), then synthetic data may be **too different** from real

##### Outcome:
In our case, the classifier struggled to differentiate real from synthetic, with **probabilities tightly clustered around 0.5**. This indicates that the CTGAN successfully generated **statistically similar samples** to the real-world material data.

---

#### Integration into the ML Pipeline
Once validated, synthetic samples were **merged with the original real dataset** to:
- Increase training volume
- Improve model generalization
- Stabilize learning of low-frequency edge cases

This augmented dataset was then used to train the **regression model** for characteristic value prediction.

---

### 3. Winsorizing: Validation-Based Outlier Treatment

This section presents a statistical strategy to evaluate and treat potential outliers in small-scale material test datasets (e.g., sets of 10 tensile strength values). Rather than discarding the lowest measurement arbitrarily, this approach leverages **generative modeling** and **sampling-based validation** to assess whether modifying an extreme value improves the representativeness of the dataset.

To determine whether an extreme tensile strength value (e.g., the lowest in a set of 10 measurements) should be retained or replaced, based on its effect on the overall **statistical alignment** with a synthetic population.

---

####  Step-by-Step Procedure

1. **Original Sample Set**
   - Begin with a set of 10 tensile strength test values obtained from physical experiments.

2. **Synthetic Population Generation using CTGAN**
   - Train a CTGAN model on the original sample to generate a synthetic population of samples that statistically mimic the underlying distribution of the real data.
   - The authenticity of these synthetic samples was verified using a **propensity score classifier**.
     - A binary classifier was trained to distinguish between real and synthetic data points.
     - The model returned probabilities close to 0.5, indicating that the synthetic data was statistically **indistinguishable from the original**, achieving **low propensity scores**.

3. **Monte Carlo Sampling of Means**
   - Draw a large number (e.g., 1000) of random samples of size 10 from the synthetic population.
   - For each sample, compute the **arithmetic mean**.
   - The distribution of these means reflects the expected variation of sample means drawn from the assumed population.
   - Compute the **population reference mean** by averaging all sampled means.

4. **Mean Absolute Percentage Error (MAPE) Evaluation**
   - Compute the **MAPE** between:
     - The original sample mean and the reference population mean.
     - The modified sample mean (e.g., with the lowest value replaced) and the reference population mean.

5. **Decision Criterion**
   - If the modified sample has a **lower MAPE**, then replacing the suspected outlier improves alignment with the population — suggesting the original value was an outlier.
   - If the original sample has a lower MAPE, then the lowest value is likely valid and should be retained.

---

####  Scientific Justification

| Benefit | Explanation |
|--------|-------------|
| **Avoids arbitrary deletion** | Values are flagged based on statistical validation rather than assumptions (e.g., “lowest = outlier”). |
| **Stabilizes statistical metrics** | Reducing extreme values improves the robustness of the mean and standard deviation, key inputs in characteristic value estimation. |
| **Population-aware** | Instead of assuming a fixed normal distribution, this method builds an empirical distribution using generative modeling. |
| **Tested synthetic realism** | CTGAN-generated samples passed **propensity score validation**, ensuring that the synthetic data closely mirrors the statistical properties of real data. |
| **Ensemble-friendly** | Works in tandem with IQR, z-score, and autoencoder-based outlier detection to reinforce decisions using multiple criteria. |

---

####  Summary

This Winsorizing method provides a **statistically grounded, data-driven mechanism** for handling extreme values in material property datasets. It ensures that the characteristic value estimation remains **robust, unbiased, and representative**, especially when working with limited sample sizes common in physical testing scenarios.

---

### 4. Characteristic Value Prediction

The final component involves using the augmented dataset to train a **regression model** that predicts characteristic values directly.

#### Model:
- **Simple Feedforward Neural Network**
- Input: Augmented material-specific features (e.g., tensile strength tests)
- Output: Predicted **characteristic value** (e.g., 5th percentile)

#### Training Strategy:
- **Normalized input features**
- **K-fold cross-validation** to assess generalization
- **MAPE and RMSE** used as evaluation metrics

---

## 🗃️ Dataset

Each sample in the dataset represents a **material-specific test** (e.g., tensile strength):

### 🔷 Raw Input Features:
- Test ID
- Material category (optional)
- Tensile strength value

### 🔶 Output Target:
- 5th percentile or other characteristic value

After outlier detection and synthetic data generation, the dataset is expanded to support model learning.

---

## 📈 Results Summary

### 🔹 Outlier Impact

| Dataset Stage         | RMSE ↓ | MAPE ↓ |
|------------------------|--------|--------|
| Before Outlier Removal | High   | High   |
| After Outlier Removal  | ↓↓     | ↓↓     |
| After Augmentation     | ↓↓↓    | ↓↓↓    |

- **Outlier filtering** significantly improved model stability and prediction accuracy.
- **CTGAN-augmented training** led to better generalization.

---

## 🔭 Future Work

- **Dynamic thresholding** for outlier scores depending on sample size
- **Larger synthetic datasets** with conditional variability
- **Advanced regressors** like ensemble models or Bayesian regressors for uncertainty estimation
- **Benchmarking** against traditional characteristic value estimation standards (e.g., EN1990)

---

## 📚 Tools & Libraries Used

- Python  
- Pandas / NumPy  
- SciKit-learn (Z-score, Isolation Forest)  
- TensorFlow / Keras (Regression)  
- CTGAN (SDV Library)  
- Matplotlib / Seaborn (Visualization)

---

## 🙋‍♂️ Author

**Anvar Mohamed Aslam Sha**  
Master’s Thesis – TU Dresden  
MSc in Advanced Computational and Civil Engineering Structural Studies  
Email: aslamshaw97@gmail.com  

---

## 📂 License

This work is for academic and research purposes only. Please cite or credit appropriately if you reuse this methodology or ideas.

---
