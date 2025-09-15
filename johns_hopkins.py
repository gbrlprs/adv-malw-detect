# Import necessary libraries 
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
# Load the dataset
# Try to locate the CSV by pattern anywhere under the project
project_root = Path(__file__).resolve().parents[1]
csv_candidates = list(project_root.rglob('*CTU-IoT-Malware-Capture-1.csv'))
if not csv_candidates:
    # Fallback to current working directory
    csv_candidates = list(Path.cwd().rglob('*CTU-IoT-Malware-Capture-1.csv'))
if not csv_candidates:
    raise FileNotFoundError(
        "Could not find a file matching '*CTU-IoT-Malware-Capture-1.csv'. "
        "Please place the dataset in the project or update the path."
    )
data_csv_path = csv_candidates[0]
print(f"Loading dataset from: {data_csv_path}")
data = pd.read_csv(data_csv_path)

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print("Missing values per column:\n", data.isna().sum())
# Inspect the DataFrame columns to find the target variable
print("Columns in the dataset:", data.columns)

# Set the features (X) and target variable (y)
# Heuristic: try common target names; fallback to last column
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

# Split the dataset into training and testing sets
stratify_arg = y if y.nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)

# Output the shapes of the features and target
print(f"Features shape: {X.shape}, Target shape: {y.shape}")
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Identify categorical and numerical columns
categorical_columns = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Create a column transformer to apply scaling and one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ],
    remainder="drop",
)

# Create a pipeline for preprocessing and PCA
preprocess_and_reduce = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("pca", PCA(n_components=0.95, random_state=42)),
    ]
)

# Fit and transform the training data, and transform the test data
X_train_transformed = preprocess_and_reduce.fit_transform(X_train)
X_test_transformed = preprocess_and_reduce.transform(X_test)

# Output the shapes of the transformed features
print(f"Transformed training features shape: {X_train_transformed.shape}")
print(f"Transformed testing features shape: {X_test_transformed.shape}")
# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Train the model
rf_model.fit(X_train_transformed, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_transformed)

# Evaluate the model
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Assuming the target variable is in y_train and needs encoding
# Step 1: Convert categorical labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)

# Step 2: Define the deep learning model
input_dim = X_train_transformed.shape[1]
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))
model.add(Dense(64, activation='relu'))
if num_classes == 2:
    model.add(Dense(1, activation='sigmoid'))
else:
    model.add(Dense(num_classes, activation='softmax'))

# Compile the model
if num_classes == 2:
    loss_fn = 'binary_crossentropy'
    metrics = ['accuracy']
else:
    loss_fn = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

# Step 3: Train the model
history = model.fit(
    X_train_transformed,
    y_train_encoded,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
)
# Make predictions
if num_classes == 2:
    y_pred_probs = model.predict(X_test_transformed, verbose=0).ravel()
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    y_pred_labels = label_encoder.inverse_transform(y_pred_binary)
else:
    y_pred_proba = model.predict(X_test_transformed, verbose=0)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# If you want to decode predictions back to original labels
# Already decoded above into y_pred_labels

# Check the predictions
print(y_pred_labels)
# Visualize the training history
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

# Encode y_test before evaluation
# Already encoded as y_test_encoded

# Convert predictions from Random Forest to numerical labels
# If y is categorical strings, encode to compare numerically
try:
    y_pred_rf_encoded = label_encoder.transform(y_pred_rf)
except Exception:
    # In case RF already outputs encoded labels
    y_pred_rf_encoded = y_pred_rf

# Convert Deep Learning model predictions to binary (0 or 1)
if num_classes == 2:
    y_pred_dl_binary = (model.predict(X_test_transformed, verbose=0).ravel() >= 0.5).astype(int)
else:
    y_pred_dl_binary = np.argmax(model.predict(X_test_transformed, verbose=0), axis=1)

# Map binary predictions back to original labels
# Assuming the mapping is 0 -> 'Benign' and 1 -> 'Malicious'
y_pred_dl_labels = label_encoder.inverse_transform(y_pred_dl_binary)

print(classification_report(y_test, y_pred_dl_labels))