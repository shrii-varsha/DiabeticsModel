# Diabetics Model
This project aims to predict whether an individual has diabetes based on various medical attributes using a Support Vector Machine (SVM) model. The dataset used for this project is the PIMA Indian Diabetes dataset.

## Requirements
To run this project, you need to have the following libraries installed:
- numpy
- pandas
- matplotlib
- scikit-learn
- imbalanced-learn
- scipy

Import the necessary libraries for data processing, visualization, and machine learning.

### Load Dataset

Load the diabetes dataset from a CSV file.

### Data Preprocessing

Split the dataset into features (x) and target (y).

Standardize the feature data.

Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

### Train-Test Split

Split the resampled data into training and testing sets.

### Hyperparameter Tuning

Perform hyperparameter tuning using RandomizedSearchCV to find the best parameters for the SVM model.
### Model Training

Train the SVM model with the best parameters found during hyperparameter tuning.

### Model Evaluation

Evaluate the model's performance using various metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.

### ROC Curve

Plot the ROC curve to visualize the model's performance.
### Prediction

Take input values for prediction, preprocess the input data, and predict whether the individual is diabetic.
