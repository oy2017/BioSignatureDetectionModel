# Deep Learning Model Tuning Journal

## Goal
To optimize a neural network model for detecting biosignatures from exoplanet spectra, aiming to match or exceed the performance of the existing XGBoost benchmark (~87% accuracy).

---

## Key Findings

1.  **PCA is Essential:** Both MLP and CNN models consistently failed to learn from raw spectral data (**E1-E4**). Applying Principal Component Analysis (PCA) was the single most critical step to enable successful training.
2.  **MLP is the Superior Architecture:** The Multi-Layer Perceptron (MLP) consistently outperformed the Convolutional Neural Network (CNN). The best MLP (**E16**, 87.63%) was significantly more accurate than the best CNN (**E10**, 83.00%).
3.  **Feature Sets are Model-Specific:** Cross-check experiments (**E18, E19**) confirmed that each model performs best on a different set of PCA components. The MLP excels with components `0-100`, while the CNN is better suited to the "cleaner" `2-102` set.
4.  **MLP Matches the Benchmark:** The final optimized MLP configuration (**E16**) successfully matched the ~87% accuracy of the XGBoost model, achieving the project's primary goal. `EarlyStopping` proved effective, as a longer 100-epoch run (**E17**) did not yield further improvement.

---

## Hyperparameters Under Investigation

*   **Model Architecture (`--model_type`):** The fundamental type of neural network used (CNN vs. MLP).
*   **Input Features (PCA vs. Raw Spectra):** The data fed into the model, controlled via `--pca_start_idx` and `--pca_end_idx`.
*   **Feature Scaling:** `StandardScaler` was applied both before PCA and on the selected PCA components. This was found to be a critical step.
*   **Epochs (`--epochs`):** The number of passes through the training dataset. `EarlyStopping` was used to prevent overfitting.
*   **Batch Size (`--batch_size`):** The number of samples per gradient update.

---

## Master Experiment Log

| ID | Model | PCA Config | Epochs | Batch | Accuracy | Command |
|----|-------|------------|--------|-------|----------|---------|
| E1 | CNN   | Raw Spectra| 5      | 32    | **50.17%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx -1 --epochs 5 --model_type cnn` |
| E2 | CNN   | Raw Spectra| 20     | 32    | **49.83%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx -1 --epochs 20 --model_type cnn` |
| E3 | CNN   | Raw Spectra| 50     | 32    | **49.83%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx -1 --epochs 50 --model_type cnn` |
| E4 | MLP   | Raw Spectra| 50     | 32    | **51.34%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx -1 --epochs 50 --model_type mlp` |
| E5 | CNN   | PCA 0-End  | 1      | 32    | **54.01%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --epochs 1 --model_type cnn` |
| E6 | CNN   | PCA 0-End  | 20     | 32    | **74.58%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --epochs 20 --model_type cnn` |
| E7 | CNN   | PCA 0-End  | 50     | 32    | **79.77%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --epochs 50 --model_type cnn` |
| E8 | CNN   | PCA 2-102  | 20     | 32    | **79.77%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 2 --pca_end_idx 102 --epochs 20 --model_type cnn` |
| E9 | CNN   | PCA 2-102  | 50     | 32    | **75.75%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 2 --pca_end_idx 102 --epochs 50 --model_type cnn` |
| E10| CNN   | PCA 2-102  | 100    | 64    | **83.00%** | **Best CNN** `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 2 --pca_end_idx 102 --epochs 100 --batch_size 64 --model_type cnn` |
| E11| CNN   | PCA 2-102  | 100    | 128   | **77.76%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 2 --pca_end_idx 102 --epochs 100 --batch_size 128 --model_type cnn` |
| E12| MLP   | PCA 2-102  | 50     | 32    | **74.92%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 2 --pca_end_idx 102 --epochs 50 --model_type mlp` |
| E13| MLP   | PCA 2-102  | 100    | 64    | **83.11%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 2 --pca_end_idx 102 --epochs 100 --batch_size 64 --model_type mlp` |
| E14| MLP   | PCA 0-100  | 50     | 32    | **83.78%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --pca_end_idx 100 --epochs 50 --model_type mlp` |
| E15| MLP   | PCA 0-100  | 50     | 256   | **85.12%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --pca_end_idx 100 --epochs 50 --batch_size 256 --model_type mlp` |
| E16| MLP   | PCA 0-100  | 50     | 128   | **87.63%** | **Best MLP** `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --pca_end_idx 100 --epochs 50 --batch_size 128 --model_type mlp` |
| E17| MLP   | PCA 0-100  | 100    | 128   | **87.63%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --pca_end_idx 100 --epochs 100 --batch_size 128 --model_type mlp` |
| E18| CNN   | PCA 0-100  | 100    | 64    | **79.26%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 0 --pca_end_idx 100 --epochs 100 --batch_size 64 --model_type cnn` |
| E19| MLP   | PCA 2-102  | 50     | 128   | **85.45%** | `./venv-fix/bin/python evaluate_deep_learning.py H2 --pca_start_idx 2 --pca_end_idx 102 --epochs 50 --batch_size 128 --model_type mlp` |

---

## Experimental Analysis

### Part 1: The Critical Role of PCA vs. Raw Spectra

The initial investigation focused on whether the models could learn from the raw spectral data. Experiments **E1, E2, E3** (CNN) and **E4** (MLP) all resulted in accuracies of ~50%, demonstrating a complete failure to learn. In contrast, as soon as PCA-transformed data was used (e.g., **E7**, **E14**), performance dramatically improved into the 80-84% range.

**Conclusion:** Raw spectral data contains high-variance noise that prevents the neural networks from learning. PCA is an essential feature engineering step.

### Part 2: Architecture & Feature Set Co-dependency

A direct comparison shows the MLP is the superior architecture. The best MLP result (**E16**, 87.63%) significantly outperformed the best CNN result (**E10**, 83%).

Crucially, the final cross-check experiments confirmed that the choice of feature set is highly dependent on the model architecture:
*   The CNN performed worse on the MLP's preferred `0-100` feature set (**E18**, 79.26%) than on its own best `2-102` set (**E10**, 83.00%). This shows the CNN is sensitive to the noisy initial components.
*   The MLP performed worse on the CNN's preferred `2-102` feature set (**E19**, 85.45%) than on its own best `0-100` set (**E16**, 87.63%). This suggests the MLP can extract useful information from the initial components that the CNN cannot.

**Conclusion:** The MLP is the more effective architecture, and it performs best when it has access to all 100 principal components.

### Part 3: Tuning - The Effect of Batch Size & Epochs

The final tuning steps revealed a critical, model-specific sensitivity to batch size and the effectiveness of early stopping.

*   **For the MLP:** A series of experiments (**E14**, **E16**, **E15**) tested batch sizes of 32, 128, and 256. Performance peaked at a batch size of **128**.
*   **For the CNN:** The best performance (**E10**) was found with a batch size of **64**.
*   **Epochs:** Extending the best MLP run to 100 epochs (**E17**) yielded no improvement over the 50-epoch run (**E16**), confirming that `EarlyStopping` is working effectively and 50 epochs is sufficient.

**Conclusion:** The optimal batch size is not universal. The MLP benefited from a larger batch size (128), while the CNN performed better with a smaller one (64).

---

## Final Conclusion & Optimal Configurations

### Optimal Configuration (MLP)
*   **Reference:** Experiment **E16**
*   **Architecture:** MLP (Dense 256 -> 128 -> 64)
*   **Input Features:** PCA Components 0-100 (Scaled)
*   **Hyperparameters:** 50 Epochs, Batch Size 128, Learning Rate 0.001
*   **Performance:** **87.63% Accuracy**

This configuration is the best-performing neural network, successfully matching the XGBoost benchmark.

### Best CNN Configuration
*   **Reference:** Experiment **E10**
*   **Architecture:** 1D CNN
*   **Input Features:** PCA Components 2-102 (Scaled, Noisy components removed)
*   **Hyperparameters:** 100 Epochs, Batch Size 64, Learning Rate 0.001
*   **Performance:** **83.00% Accuracy**

The CNN also demonstrates strong performance, proving to be a viable, albeit slightly less optimal, alternative to the MLP.