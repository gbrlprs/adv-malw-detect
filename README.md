## IoT Malware Detection (RandomForest + Keras)

Detecting malicious IoT network activity using classical ML and a small neural network. This project loads and preprocesses CTU IoT capture data, reduces dimensionality with PCA, and compares a RandomForest baseline against a Keras MLP.

### Why this project
- Showcases practical ML: robust data loading, preprocessing pipelines, dimensionality reduction, and two modeling paradigms.
- Clear evaluation: classification report and confusion matrix for actionable feedback.
- Lightweight and reproducible: minimal code to run end-to-end.

### Dataset
- CTU IoT Malware Capture (subset). Place the CSV anywhere in the repo; the script auto-discovers files matching `*CTU-IoT-Malware-Capture-1.csv`.

### Approach
- Preprocessing
  - Column typing via `pandas.DataFrame.select_dtypes`.
  - `ColumnTransformer`: `StandardScaler` for numeric; `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categorical.
  - `PCA(n_components=0.95)` to retain 95% variance with fewer features.
- Models
  - RandomForestClassifier: strong baseline, fast, interpretable.
  - Keras MLP: `relu` Dense layers; `sigmoid`+`binary_crossentropy` for binary or `softmax`+`sparse_categorical_crossentropy` for multi-class.
- Target handling
  - `LabelEncoder` for consistent label space; predictions mapped back to original labels.

### Results (what to report)
- Printouts: accuracy, precision, recall, F1 via `classification_report`.
- Confusion matrix: highlights false positives/negatives.
- Takeaway template:
  - “RandomForest achieved X% accuracy (precision Y%, recall Z%). Neural net produced similar results; RF is simpler and faster, so recommended for deployment.”

### How to run
```bash
# Optional: create a venv
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
# Install
pip install -r requirements.txt
# Run
python johns/johns_hopkins.py
```

### Key files
- `johns/johns_hopkins.py`: end-to-end pipeline (load → preprocess → PCA → RF/MLP → evaluate).
- `requirements.txt`: Python dependencies.

### Tech stack and why it matters
- pandas: CSV IO, dataframe ops.
- scikit-learn: split, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `PCA`, `RandomForest`, metrics.
  - Note: newer versions use `sparse_output` in `OneHotEncoder`.
- TensorFlow/Keras: quick MLP with `Sequential` + `Dense`.
- matplotlib/seaborn: simple training curves.

### Python notes
- Train/test split uses `stratify=y` to preserve class balance.
- Preprocessing applies only to train, reuses fitted transforms on test (avoids leakage).
- PCA downstream of preprocessing ensures uniform scaling before variance-based reduction.
- Binary vs multi-class loss/activation chosen automatically by `num_classes`.

### Possible extensions
- Add `EarlyStopping` and `Dropout` to the MLP.
- Hyperparameter tuning with `GridSearchCV` or `KerasTuner`.
- Feature importance (RF) and SHAP explanations.
- Cross-validation and stratified folds.

### Contact
- If you have questions or ideas for improvements, feel free to open an issue or PR.
