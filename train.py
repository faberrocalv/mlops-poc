# Import libraries
import pandas as pd
from sklearn import tree
from joblib import dump


# Read data
df = pd.read_csv('./processed_data/train.csv', sep=',')

# Split data into dependent and independent variables
X_train = df.drop('sentimiento', axis=1)
Y_train = df['sentimiento']

# Train a classification model
model = tree.DecisionTreeClassifier(min_samples_leaf=2, max_depth=10)
model = model.fit(X_train, Y_train)

# Serialize and save model
dump(model, './model/class_model.joblib')
