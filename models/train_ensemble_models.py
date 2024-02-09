#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv('yourpath.csv')

# Define your numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('liquidations')  # Assuming 'liquidations' is your target variable

# Preprocessing for numerical data: imputation + scaling
numerical_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Since 'address' is the categorical column, let's create a transformer for it as well
categorical_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, ['address'])
    ])

# Define the target and features
X = df.drop('liquidations', axis=1)
y = df['liquidations']
print(df['liquidations'].value_counts())


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define alternative models to test
alternative_models = {
    'LogisticRegression': LogisticRegression(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(silent=True),  # silent=True to avoid CatBoost verbose output
    'RandomForest':RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}

# Fit the preprocessor to the training data
preprocessor.fit(X_train)

# Get feature names after preprocessing
# Use get_feature_names_out for scikit-learn 0.24 and above
onehot_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(['address'])
all_features = numerical_features + onehot_columns.tolist()

# Evaluate each alternative model
for name, model in alternative_models.items():
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)])
    # Train the model pipeline
    model_pipeline.fit(X_train, y_train)

    # Predict the test data
    predictions = model_pipeline.predict(X_test)
    print(f"{name} Classification Report:\n")
    print(classification_report(y_test, predictions))
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_test, probabilities)
    print(f"{name} roc_auc: {roc_auc}\n")

    # Compute and print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy: {accuracy}\n")

