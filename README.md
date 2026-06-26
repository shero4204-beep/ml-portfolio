# ml-portfolio
Machine learning projects including Linear Regression, ANN, Random Forest, k-NN, optimization algorithms, and data analysis.
# Cardiovascular Risk Prediction using ANN

## Overview
This project predicts cardiovascular disease risk using Artificial Neural Networks trained on:
- Original dataset
- CTGAN-generated synthetic dataset

## Models Used
- ANN
- Random Forest (comparison)

## Techniques
- Data preprocessing
- Feature scaling
- Synthetic data generation
- Confusion matrix
- Accuracy comparison
- A. Data Loading
We loaded two separate datasets:
•	heart_disease_dataset.csv – real clinical data
•	heart_disease_dataset_Synthetic_Data.csv – synthetic records (mimicking real distributions)
Each dataset was kept separate throughout the analysis for clear comparison.
✅ B. Preprocessing
We applied standard preprocessing techniques:
•	Removed missing values using median imputation
•	Handled categorical variables (cp, thal, restecg, slope) with one-hot encoding
•	Scaled all numerical features using StandardScaler
•	Encoded the target variable num using LabelEncoder
These steps ensured that both datasets were properly formatted for training a neural network.
✅ C. Train/Test Splits
Each dataset was split independently using an 80/20 ratio.
This allows us to evaluate each model based on its performance on unseen data from the same domain.
✅ D. Model Architecture
We built an ANN model using TensorFlow, containing:
•	Input layer (matching the number of features)
•	Two hidden layers (64 and 32 units with ReLU activation)
•	Output layer (with softmax for multi-class classification)
Each model was compiled with sparse_categorical_crossentropy and trained for 50 epochs using the Adam optimizer.
✅ E. Evaluation
We evaluated both models using:
•	Accuracy score
•	Classification report (Precision, Recall, F1-score)
•	Confusion matrix (visualized using Seaborn heatmaps)
This allowed us to clearly see how well each model performed on its own test data.
________________________________________
3. Key Findings
Dataset	Model Type	Evaluation
Real	ANN (TensorFlow)	Accuracy, Classification Report, Confusion Matrix
Synthetic	ANN (TensorFlow)	Accuracy, Classification Report, Confusion Matrix
•	The original-data ANN showed relatively better performance, as expected, due to real clinical signals.
•	The synthetic-data ANN demonstrated decent generalization, indicating promise for privacy-preserving data use in training.
________________________________________
4. Alignment with Shimizu et al. (2024)
Shimizu et al. Approach	Our Implementation
Internal vs. External Validation (Brazil vs. USA hospitals)	Real vs. Synthetic Dataset comparison
Multi-model evaluation (Random Forest, MLP, etc.)	Implemented TensorFlow-based MLP (ANN); others planned
Data diversity & generalizability focus	Validated models on separate datasets without mixing
Interpretability focus (SHAP, LIME)	Will implement SHAP and LIME in the next update
________________________________________
5. Tools and Libraries Used
•	Python (v3.10+)
•	TensorFlow/Keras – for neural network modeling
•	Scikit-learn – for preprocessing, splitting, and metrics
•	Pandas, NumPy – data handling
•	Matplotlib, Seaborn – visualizing confusion matrices
________________________________________
6. Next Steps
•	Evaluate cross-domain performance (e.g., synthetic-trained model tested on real data)
•	Add ROC-AUC scoring and visualization
•	Implement SHAP and LIME for interpretability of the ANN predictions
•	Expand model types (Random Forest, XGBoost) to align further with Shimizu et al.

