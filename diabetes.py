import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from scipy.stats import uniform

diabetes_dataset = pd.read_csv(r"C:\Shrii Varsha\python ml\diabetics.csv")
print(diabetes_dataset['Outcome'].value_counts())


x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

scaler = StandardScaler()
standized_data = scaler.fit_transform(x)

# handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(standized_data, y)

# Split data as training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=2)

# parameter distribution for Randomized Search
param_dist = {
    'C': uniform(0.1, 100), 
    'kernel': ['linear', 'sigmoid', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(svm.SVC(), param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)

random_search.fit(X_train, Y_train)

print("Best parameters found by RandomizedSearchCV: ", random_search.best_params_)

best_model = random_search.best_estimator_

X_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(Y_train, X_train_pred)
print("Training Accuracy with best parameters:", train_accuracy)

X_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_pred)
print("Test Accuracy with best parameters:", test_accuracy)

# Performance metrics
precision = precision_score(Y_test, X_test_pred)
recall = recall_score(Y_test, X_test_pred)
f1 = f1_score(Y_test, X_test_pred)
roc_auc = roc_auc_score(Y_test, best_model.decision_function(X_test))

print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}, ROC-AUC: {roc_auc}")

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, X_test_pred)
print("Confusion Matrix:\n", conf_matrix)

# ROC Curve
fpr, tpr, _ = roc_curve(Y_test, best_model.decision_function(X_test))
plt.plot(fpr, tpr, label='SVM (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Input for prediction
colunm_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

input_data = []

for i in colunm_names:
    value = float(input(f"Enter the value for {i}: "))
    input_data.append(value)

input_df = pd.DataFrame([input_data], columns=colunm_names)

std_data = scaler.transform(input_df)

prediction = best_model.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
