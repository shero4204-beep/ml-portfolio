#Project--Cardiovascular Risk Factors
#Members-- Shreya, Rishi, Parth and Sheshank

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
# %%
#Loading the datasets
original_path = r"C:\Users\shero\PycharmProjects\pythonProject\venv\Scripts\OneDrive_2025-03-22\Cardio Dataset\Orignal\heart_disease_dataset.csv"
synthetic_path = r"C:\Users\shero\PycharmProjects\pythonProject\venv\Scripts\OneDrive_2025-03-22\Cardio Dataset\Synthetic\heart_disease_dataset_Synthetic_Data.csv"

df_original = pd.read_csv(original_path)
df_synthetic = pd.read_csv(synthetic_path)

print("Original Data Shape:", df_original.shape)
print("Synthetic Data Shape:", df_synthetic.shape)


# %%
# Using 'num' as target variable
target_column = 'num'

def preprocess(df):
    df.columns = df.columns.str.strip()
    df.fillna(df.median(numeric_only=True), inplace=True)

    categorical_cols = ['cp', 'restecg', 'thal', 'slope']
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded, X.columns

X_orig, y_orig, feature_names = preprocess(df_original)
X_syn, y_syn, _ = preprocess(df_synthetic)

# %%
# Spliting of both into train-test (80-20) with same random state
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)

# %%
#defining and training the ANN model
def build_ann(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

n_classes = len(np.unique(y_orig))

# %%
# Model trained on original data
model_orig = build_ann(X_train_orig.shape[1], n_classes)
model_orig.fit(X_train_orig, y_train_orig, epochs=100, batch_size=32, verbose=0)

# Model trained on synthetic data
model_syn = build_ann(X_train_syn.shape[1], n_classes)
model_syn.fit(X_train_syn, y_train_syn, epochs=100, batch_size=32, verbose=0)

#%%
#Evaluating both models
def evaluate_model(model, X_test, y_test, label="Model"):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred_classes)
    cm = confusion_matrix(y_test, y_pred_classes)
    print(f"\n {label} Accuracy: {acc:.4f}")
    print(f"{label} Classification Report:\n", classification_report(y_test, y_pred_classes))
    return acc, cm

#%%
# Evaluation of original-trained ANN on original test data
acc_orig, cm_orig = evaluate_model(model_orig, X_test_orig, y_test_orig, "Original ANN")

# Evaluation of synthetic-trained ANN on synthetic test data
acc_syn, cm_syn = evaluate_model(model_syn, X_test_syn, y_test_syn, "Synthetic ANN")
#%%
#Visulaization of confusion matrix
def plot_cm(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_cm(cm_orig, "Confusion Matrix: Original ANN")
plot_cm(cm_syn, "Confusion Matrix: Synthetic ANN")
