# Import libraries
import pandas as pd
from joblib import load
from sklearn import metrics
import os
import json


# Read data
df = pd.read_csv('./processed_data/test.csv', sep=',')

# Split data into dependent and independent variables
X_test = df.drop('sentimiento', axis=1)
Y_test = df['sentimiento']

# Load model
model = load('./model/class_model.joblib')

# Predict
Y_pred = model.predict(X_test)

# Compute test accuracy
acc = metrics.accuracy_score(Y_test, Y_pred)

# Test accuracy to JSON
test_metadata = {
    'test_acc': acc
}

# Set output path
test_results_file = 'test_metadata.json'
results_path = os.path.join('./results/', test_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(test_metadata, outfile)