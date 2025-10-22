import pickle 
import numpy as np
import pandas as pd

# Load the unpreprocessed data
with open('N:/Elevvo/Student Score Prediction/Student_Score_prediction/data/unprocessed_data.pkl', 'rb') as f:
    _, _, X_test, y_test = pickle.load(f)
    print('Data Loaded Successfully!')

# Random choice of a sample from the test set for prediction
idx = np.random.randint(0, X_test.shape[0])

# Sample For Prediction
X_sample = pd.DataFrame([X_test.iloc[idx]])

with open('N:/Elevvo/Student Score Prediction/Student_Score_prediction/models/ridge_best_model.pkl','rb') as f:
    model_pipeline=pickle.load(f)
    print('Model is imported')

# Fit the model pipeline
log_score = model_pipeline.predict(X_sample)
# score = np.expm1(log_score)  # Apply expm1 to reverse the log1p transformation
score=log_score
print('Predicted Score: ', score[0])
print('Actual Score: ', y_test.iloc[idx])