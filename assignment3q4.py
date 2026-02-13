import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

kidney_df = pd.read_csv('kidney_disease.csv')

# clean target Column
# dataset often has 'ckd\t' (with a tab) which needs to be fixed

kidney_df['classification'] = kidney_df['classification'].replace(
    {'ckd': 0, 'ckd\t': 0, 'notckd': 1}
)

kidney_df = kidney_df.dropna(subset=['classification'])
kidney_df['classification'] = kidney_df['classification'].astype(int)

# clean Feature Columns
# remove 'id' if it exists
if 'id' in kidney_df.columns:
    kidney_df = kidney_df.drop(columns=['id'])

feature_cols = [col for col in kidney_df.columns if col != 'classification']
for col in feature_cols:
    kidney_df[col] = pd.to_numeric(kidney_df[col], errors='coerce')

# handle NaNs
# drop columns that are completely empty
kidney_df = kidney_df.dropna(axis=1, how='all')
# fill remaining missing values with the median of each column
kidney_df = kidney_df.fillna(kidney_df.median())

# define X and y
X = kidney_df.drop(columns=['classification'])
y = kidney_df['classification']

# split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# train KNN with k=5
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(X_train, y_train)

# predict on test data
y_pred = knn_5.predict(X_test)

# compute Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=0)
rec = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)

print("--- Question 4 Metrics ---")
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}\n")

# COMMENTS FOR QUESTION 4

# True Positive: We correctly predicted the patient has kidney disease (CKD/0).
# True Negative: We correctly predicted the patient is healthy (notckd/1).
# False Positive: We predicted the patient has CKD, but they are actually healthy.
# False Negative: We predicted the patient is healthy, but they actually have CKD.

# Accuracy calculates the percentage of correct predictions overall. However, if the dataset
# is imbalanced a model could simply guess "CKD" for everyone and still achieve 90% accuracy
# without actually learning to detect healthy patients.

# If missing a kidney disease case is very serious,Recall is the most important metric.
# High Recall ensures we minimize False Negatives meaning we catch as many true cases of the disease as possible,
# even if it leads to a few more false alarms.