import pandas as pd
from sklearn.model_selection import train_test_split

# load kidney disease data
kidney_df = pd.read_csv('kidney_disease.csv')

# handle missing values if necessary
kidney_df = kidney_df.fillna(kidney_df.median(numeric_only=True))

# create feature matrix X and label vector y
X = kidney_df.drop(columns=['classification'])
y = kidney_df['classification']

# split dataset 70% training, 30% testing with fixed random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# COMMENTS FOR QUESTION 3
# We shouldn't train and test a model on the same data because the model might just memorize the answers (overfitting) instead of actually learning the patterns.
# The purpose of the testing set is to act as unseen and real world data so we can fairly judge how well the model generalizes to new patients.