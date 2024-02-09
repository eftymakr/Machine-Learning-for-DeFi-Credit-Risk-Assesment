#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Assuming df is your DataFrame after loading the dataset
# Assuming 'liquidations' is your target variable and 'address' is a categorical feature
# Load your dataset
df = pd.read_csv('Stat_Learning_set_v02.csv')

# Define preprocessing for numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('liquidations')  # Target variable

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, ['address'])
    ])

# Split the data
X = df.drop('liquidations', axis=1)
y = df['liquidations']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define CNN model within a function for KerasClassifier
def create_cnn_model(input_dim):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(400, activation='relu'),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model with KerasClassifier
input_dim = preprocessor.fit_transform(X_train).shape[1]
cnn_classifier = KerasClassifier(build_fn=create_cnn_model, input_dim=input_dim, epochs=100, batch_size=256, verbose=0)

# Create a pipeline with preprocessing, SMOTE, and the classifier
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', cnn_classifier)
])

# Train the CNN model
model_pipeline.fit(X_train, y_train)

# Predict the test data
predictions = model_pipeline.predict(X_test)

# Compute and print classification report and accuracy
print("CNN Model Classification Report:\n")
print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
print(f"CNN Model Accuracy: {accuracy}\n")

# Compute ROC AUC score
# For AUC, we need to predict probabilities and use the positive class's probability
probabilities = model_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, probabilities)
print(f"CNN Model ROC AUC: {roc_auc}\n")

