## IoT Malware Detection (RandomForest + Keras)

detecting malicious IoT network activity using classical ML and a small neural network. This project loads and preprocesses CTU IoT capture data, reduces dimensionality with PCA, and compares a RandomForest baseline against a Keras MLP.

### why this project
- showcases practical ML: robust data loading, preprocessing pipelines, dimensionality reduction, and two modeling paradigms.
- clear evaluation: classification report and confusion matrix for actionable feedback.
- lightweight and reproducible: minimal code to run end-to-end.

### dataset
- CTU IoT Malware Capture (subset). Place the CSV anywhere in the repo; the script auto-discovers files matching `*CTU-IoT-Malware-Capture-1.csv`.

### approach
- preprocessing
  - column typing via `pandas.DataFrame.select_dtypes`.
  - `ColumnTransformer`: `StandardScaler` for numeric; `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categorical.
  - `PCA(n_components=0.95)` to retain 95% variance with fewer features.
- models
  - RandomForestClassifier: strong baseline, fast, interpretable.
  - Keras MLP: `relu` Dense layers; `sigmoid`+`binary_crossentropy` for binary or `softmax`+`sparse_categorical_crossentropy` for multi-class.
- target handling
  - `LabelEncoder` for consistent label space; predictions mapped back to original labels.

 ### results (what to report)
- printouts: accuracy, precision, recall, F1 via `classification_report`.
- confusion matrix: highlights false positives/negatives.
- takeaway template:
  - “RandomForest achieved X% accuracy (precision Y%, recall Z%). neural net produced similar results; RF is simpler and faster, so recommended for deployment.”

### how to run
```bash
# Optional: create a venv
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
# Install
pip install -r requirements.txt
# Run
python johns/johns_hopkins.py
```

### key files
- `johns/johns_hopkins.py`: end-to-end pipeline (load → preprocess → PCA → RF/MLP → evaluate).
- `requirements.txt`: Python dependencies.

### tech stack and why it matters
- pandas: CSV IO, dataframe ops.
- scikit-learn: split, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `PCA`, `RandomForest`, metrics.
  - note: newer versions use `sparse_output` in `OneHotEncoder`.
- TensorFlow/Keras: quick MLP with `Sequential` + `Dense`.
- matplotlib/seaborn: simple training curves.

### python notes
- train/test split uses `stratify=y` to preserve class balance.
- preprocessing applies only to train, reuses fitted transforms on test (avoids leakage).
- PCA downstream of preprocessing ensures uniform scaling before variance-based reduction.
- binary vs multi-class loss/activation chosen automatically by `num_classes`.

### possible extensions
- Add `EarlyStopping` and `Dropout` to the MLP.
- Hyperparameter tuning with `GridSearchCV` or `KerasTuner`.
- Feature importance (RF) and SHAP explanations.
- Cross-validation and stratified folds.

### Contact
- If you have questions or ideas for improvements, feel free to open an issue or PR.
