import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load dataset
kidney_df = pd.read_csv('kidney_disease.csv')

# clean up classification column since it has some weird tabs in it
# mapping ckd to 0 and notckd to 1
kidney_df['classification'] = kidney_df['classification'].replace(
    {'ckd': 0, 'ckd\t': 0, 'notckd': 1}
)

# make sure the target is numbers and drop rows where its missing
kidney_df = kidney_df.dropna(subset=['classification'])
kidney_df['classification'] = kidney_df['classification'].astype(int)

# remove id column cause we dont use it for training
if 'id' in kidney_df.columns:
    kidney_df = kidney_df.drop(columns=['id'])

# force all features to be numeric and turn ? into NaN
feature_cols = [col for col in kidney_df.columns if col != 'classification']
for col in feature_cols:
    kidney_df[col] = pd.to_numeric(kidney_df[col], errors='coerce')

# drop columns that are completely empty/NaN (this was the missing fix!)
kidney_df = kidney_df.dropna(axis=1, how='all')

# fill remaining missing values with the median so knn works
kidney_df = kidney_df.fillna(kidney_df.median())

# set up X and y
X = kidney_df.drop(columns=['classification'])
y = kidney_df['classification']

# split into training and testing 70 30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# list of k values to test
k_values = [1, 3, 5, 7, 9]
results = []
best_k = 1
best_acc = 0

print("--- Question 5 Results ---")
print(f"{'k':<5} | {'Accuracy'}")
print("-" * 15)

# loop through each k to see which one works best
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)

    accuracy_k = accuracy_score(y_test, y_pred_k)
    results.append((k, accuracy_k))

    print(f"{k:<5} | {accuracy_k:.4f}")

    # keep track of the best one
    if accuracy_k > best_acc:
        best_acc = accuracy_k
        best_k = k

print(f"\nThe value of k that gives the highest test accuracy is k={best_k} with an accuracy of {best_acc:.4f}\n")

# COMMENTS FOR QUESTION 5
# Changing k changes how the model makes decisions. A small k looks at just a few neighbors so its very sensitive to local details
# while a large k looks at a lot of neighbors and smooths things out more

# Very small values like k=1 can cause overfitting because the model memorizes the noise in the training data
# instead of learning the actual pattern

# Very large values of k can cause underfitting because it ignores the local details and just picks the most common class
# which makes the decision boundary too simple