
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

# Load dataset
df = pd.read_csv('Stat_Learning_set_v02.csv')

# Define numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('liquidations')  # 'liquidations' is the target variable

# Preprocessing for numerical data: imputation + scaling
numerical_transformer = ImbPipeline(steps=

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
                                    
#######For Hyperparameter Tuning######
                                    
# Define hyperparameter grids
xgb_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 6],
    'classifier__min_child_weight': [1, 3]
}

lgbm_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__num_leaves': [31, 62],
    'classifier__max_depth': [3, 6],
}

# Hyperparameter tuning for XGBoost and LightGBM
for name, model in [('XGBoost', XGBClassifier()), ('LightGBM', LGBMClassifier())]:
    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    param_grid = xgb_param_grid if name == 'XGBoost' else lgbm_param_grid
    grid_search = GridSearchCV(model_pipeline, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"{name} Classification Report:\n{classification_report(y_test, predictions)}")
    print(f"{name} roc_auc: {roc_auc}")
    print(f"{name} Accuracy: {accuracy}\n")

