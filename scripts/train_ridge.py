import pickle
import numpy as np

# Read the unpreprocessed data
# with open('../data/unprocessed_data.pkl', 'rb') as f:
#     X_train, y_train, X_test, y_test = pickle.load(f)
#     print('Data Loaded Successfully!')
with open('N:/Elevvo/Student Score Prediction/Student_Score_prediction/data/unprocessed_data.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)
    print('Data Loaded Successfully!')

# Apply log on the student scores
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Load the model pipiline
# with open('../models/ridge_best_model.pkl','rb') as f:
#     model_pipeline=pickle.load(f)
#     print('Model is imported')
with open('N:/Elevvo/Student Score Prediction/Student_Score_prediction/models/ridge_best_model.pkl','rb') as f:
    model_pipeline=pickle.load(f)
    print('Model is imported')


# Fit the model pipeline
model_pipeline.fit(X_train, y_train_log)
print('Model Pipeline Fitted Successfully!')