import os
os.chdir(os.getcwd()+'/chatgpt')
print(os.getcwd())

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import lightgbm as lgb
import pickle

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Check for missing values
print(train_df.isnull().sum())

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)

# Vectorize the data
vectorizer = CountVectorizer()
train_feature = vectorizer.fit_transform(X_train).astype('float64')
val_feature = vectorizer.transform(X_val).astype('float64')
test_feature = vectorizer.transform(test_df['text']).astype('float64')

# Set up the parameter grid for RandomizedSearchCV
params = {
    'n_estimators': [100, 500, 1000, 2000],
    'learning_rate': [0.001, 0.01, 0.1, 0.5],
    'num_leaves': [10, 20, 30, 40, 50],
    'boosting_type': ['gbdt', 'dart'],
}

# Set up the model
clf = lgb.LGBMClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=10, cv=5, scoring='f1_macro', random_state=42)

# Train the model with RandomizedSearchCV
random_search.fit(train_feature, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Save the best model as a pickle file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Predict on the validation set
y_pred = best_model.predict(val_feature)

# Get evaluation metrics
cm = confusion_matrix(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')

# Save evaluation metrics to a csv file
result_dict = {'Confusion Matrix': [cm], 'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]}
result_df = pd.DataFrame(result_dict)
result_df.to_csv('best_model_result.csv', index=False)

# Update the best model with the vectorized train data
best_model.fit(train_feature, y_train)

# Save the updated best model as a pickle file
with open('final_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Predict on the test set and create a submission file
test_pred = best_model.predict(test_feature)
submission_df = pd.DataFrame({'id': test_df['id'], 'label': test_pred})
submission_df.to_csv('sample_submission.csv', index=False)


