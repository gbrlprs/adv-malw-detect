# Imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pathlib import Path
# Data
project_root = Path(__file__).resolve().parents[1]
csv_candidates = list(project_root.rglob('*CTU-IoT-Malware-Capture-1.csv'))
if not csv_candidates:
    # Fallback: CWD
    csv_candidates = list(Path.cwd().rglob('*CTU-IoT-Malware-Capture-1.csv'))
if not csv_candidates:
    raise FileNotFoundError(
        "Could not find a file matching '*CTU-IoT-Malware-Capture-1.csv'. "
        "Please place the dataset in the project or update the path."
    )
data_csv_path = csv_candidates[0]
print(f"Loading dataset from: {data_csv_path}")
data = pd.read_csv(data_csv_path)

# Preview
print(data.head())

# Missing values
print("Missing values per column:\n", data.isna().sum())
# Columns
print("Columns in the dataset:", data.columns)

# Target
possible_targets = ['label', 'Label', 'class', 'Class', 'target', 'Target', 'malicious', 'Malicious', 'y', 'Category']
target_col = None
for col in possible_targets:
    if col in data.columns:
        target_col = col
        break
if target_col is None:
    target_col = data.columns[-1]

X = data.drop(columns=[target_col])
y = data[target_col]

# Split
stratify_arg = y if y.nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)

# Shapes
print(f"Features shape: {X.shape}, Target shape: {y.shape}")
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Dtypes
categorical_columns = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Preprocess
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ],
    remainder="drop",
)

# PCA
preprocess_and_reduce = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("pca", PCA(n_components=0.95, random_state=42)),
    ]
)

# Transform
X_train_transformed = preprocess_and_reduce.fit_transform(X_train)
X_test_transformed = preprocess_and_reduce.transform(X_test)

# Transformed shapes
print(f"Transformed training features shape: {X_train_transformed.shape}")
print(f"Transformed testing features shape: {X_test_transformed.shape}")
# RF
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# RF train
rf_model.fit(X_train_transformed, y_train)

# RF predict
y_pred_rf = rf_model.predict(X_test_transformed)

# RF report
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)

# NN
input_dim = X_train_transformed.shape[1]
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))
model.add(Dense(64, activation='relu'))
if num_classes == 2:
    model.add(Dense(1, activation='sigmoid'))
else:
    model.add(Dense(num_classes, activation='softmax'))

# Compile
if num_classes == 2:
    loss_fn = 'binary_crossentropy'
    metrics = ['accuracy']
else:
    loss_fn = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

# Train
history = model.fit(
    X_train_transformed,
    y_train_encoded,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
)
# Predict
if num_classes == 2:
    y_pred_probs = model.predict(X_test_transformed, verbose=0).ravel()
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    y_pred_labels = label_encoder.inverse_transform(y_pred_binary)
else:
    y_pred_proba = model.predict(X_test_transformed, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Decoded above

# Predictions
print(y_pred_labels)
# Plots
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
from sklearn.metrics import classification_report

# Encoded above

# RF numeric labels
try:
    y_pred_rf_encoded = label_encoder.transform(y_pred_rf)
except Exception:
    # If already encoded
    y_pred_rf_encoded = y_pred_rf

# DL numeric labels
if num_classes == 2:
    y_pred_dl_binary = (model.predict(X_test_transformed, verbose=0).ravel() >= 0.5).astype(int)
else:
    y_pred_dl_binary = np.argmax(model.predict(X_test_transformed, verbose=0), axis=1)

# Back to labels
y_pred_dl_labels = label_encoder.inverse_transform(y_pred_dl_binary)

print(classification_report(y_test, y_pred_dl_labels))
