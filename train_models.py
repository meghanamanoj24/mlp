import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv('sample_data.csv')

# Prepare features and target variables
X = data[['age', 'experience', 'education_level']]
y_salary = data['salary']  # For regression models
y_loan = data['loan_approved']  # For classification models

# Split the data
X_train, X_test, y_salary_train, y_salary_test, y_loan_train, y_loan_test = train_test_split(
    X, y_salary, y_loan, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and save Simple Linear Regression (SLR)
slr = LinearRegression()
slr.fit(X_train[['experience']], y_salary_train)
joblib.dump(slr, 'models/slr_model.joblib')

# Train and save Multiple Linear Regression (MLR)
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_salary_train)
joblib.dump(mlr, 'models/mlr_model.joblib')

# Train and save Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_salary_train)
joblib.dump((poly, poly_reg), 'models/poly_model.joblib')

# Train and save Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_loan_train)
joblib.dump(log_reg, 'models/logistic_model.joblib')

# Train and save KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_loan_train)
joblib.dump(knn, 'models/knn_model.joblib')

# Save the scaler
joblib.dump(scaler, 'models/scaler.joblib')

print("All models have been trained and saved successfully!") 