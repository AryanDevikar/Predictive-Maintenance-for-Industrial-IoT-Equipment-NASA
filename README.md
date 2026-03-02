# Forecasting Equipment Failures in Industrial IoT Environments

## Overview

This project addresses the challenge of unplanned equipment failures in Industrial IoT (IIoT) environments using machine learning to predict the **Remaining Useful Life (RUL)** of aircraft engines. Using NASA's C-MAPSS dataset, the pipeline covers data preprocessing, feature engineering, and model training with cross-validation across both regression and classification tasks.

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

Each engine begins with different degrees of initial wear and manufacturing variation, operates normally, develops a fault, and eventually fails. The dataset includes:

| Field | Description |
|---|---|
| Unit number | Unique engine identifier |
| Time cycles | Number of operational cycles completed |
| Operational settings | 3 settings influencing engine performance |
| Sensor measurements | 21 sensor readings per cycle (temperature, pressure, fan speed, etc.) |

**Splits:**
- `train_FD001` — Complete run-to-failure sensor data
- `test_FD001` — Sensor data up to a point before failure
- `RUL_FD001` — Ground truth RUL values for the test set

---

## Methodology

### Data Preprocessing
- **Feature selection:** 7 near-constant sensors (s_1, s_5, s_6, s_10, s_16, s_18, s_19) were removed after correlation analysis
- **RUL calculation:** `RUL = max_time_cycle - current_time_cycle`
- **Normalization:** MinMaxScaler applied to all features
- **RUL clipping:** Values clipped at 195 cycles to focus on degradation patterns near failure

### Feature Engineering
- **Moving averages:** 10-cycle rolling means computed per sensor to reduce noise and highlight degradation trends
- **Risk categorization** (for classification):

| Risk Zone | RUL Range |
|---|---|
| 🔴 RISK ZONE | ≤ 68 cycles |
| 🟡 MODERATED RISK | 69–137 cycles |
| 🟢 NO RISK | > 137 cycles |

### Models

**Regression** (predicting exact RUL):
- Linear Regression *(baseline)*
- Ridge Regression *(L2 regularization)*
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

**Classification** (predicting risk category):
- Random Forest Classifier
- XGBoost Classifier
- Naive Bayes
- K-Nearest Neighbors (KNN)

### Evaluation
- **Regression:** RMSE, R² Score
- **Classification:** Accuracy, Confusion Matrix, Precision/Recall/F1
- **Validation:** 4-fold cross-validation for all models

---

## Results

### Regression

| Model | RMSE (Train) | RMSE (Test) | RMSE (Validation) |
|---|---|---|---|
| LightGBM | 25.661 | **30.280** | 37.918 |
| Random Forest | 25.722 | 30.432 | 37.936 |
| XGBoost | **6.370** | 30.482 | **35.286** |
| Ridge Regression | 35.422 | 36.203 | 45.931 |
| Linear Regression | 35.422 | 36.207 | 45.982 |

**Key observations:**
- **LightGBM** achieves the lowest test RMSE (30.280), making it the best generalizer
- **XGBoost** shows significant overfitting (train RMSE 6.370 vs. validation RMSE 35.286)
- Ridge Regression offers no meaningful improvement over Linear Regression, suggesting L2 regularization doesn't address the core issue in this dataset
- Most predictive features: rolling averages of **s_2, s_3, s_4, s_7** (temperature and pressure sensors)

### Classification

| Model | Test Accuracy | Validation Accuracy |
|---|---|---|
| XGBoost | 0.722 | 0.450 |
| Random Forest | 0.719 | 0.480 |
| LightGBM | 0.714 | 0.420 |
| KNN | 0.671 | 0.600 |
| Naive Bayes | 0.645 | **0.620** |

**Key observations:**
- Tree-based models show heavy overfitting on classification tasks
- **Naive Bayes** and **KNN** generalize much better, with more consistent test/validation performance

---

## Conclusions

- Ensemble methods (LightGBM, Random Forest) outperform linear models for RUL regression, but suffer overfitting in classification
- Temperature and pressure sensors are the strongest predictors of remaining useful life
- The methodology is transferable to other IIoT predictive maintenance settings

---

## Future Scope

- **Deep learning:** LSTM networks to capture complex temporal patterns in sensor data
- **Transfer learning:** Adapting knowledge across different engine types with limited training data
- **Uncertainty quantification:** Providing confidence bounds on predictions for more robust maintenance planning

---

## References

- Saxena et al. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *PHM Conference*, IEEE.
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Breiman (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
- Heng et al. (2009). Rotating machinery prognostics: State of the art, challenges and opportunities. *Mechanical Systems and Signal Processing, 23*(3), 724–739.
