import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Files from (https://github.com/dylan-slack/TalkToModel)
df = pd.read_csv("diabetes.csv")

# Extract input(x) and output(y)
y = df['y']
X = df.drop(columns=['y'])

# Standardize the dataset
sdf = (X - X.mean()) / X.std()

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k = 3


# Calculate the Euclidean Distance between 2 rows
def euclidean_dis(row_1, row_2):
    result = np.sqrt(np.sum((row_1 - row_2) ** 2))
    return result


# K nearest-neighbors algorithm
def knn(X_train, y_train, X_test, k):
    predictions = []

    for i, test_row in X_test.iterrows():
        distances = []

        # Calculate distances between the test data and all training data
        for j, train_row in X_train.iterrows():
            d = euclidean_dis(test_row, train_row)
            distances.append(d)

        # Sort distances and select the k nearest neighbor's index
        sorted_d = np.argsort(distances)
        nearest_neighbors = sorted_d[:k]

        # Get y labels from y_train based on neighbors' indices
        labels = []
        for x in nearest_neighbors:
            labels.append(y_train.iloc[x])

        # Classify the test point through a majority vote
        results = max(set(labels), key=labels.count)
        predictions.append(results)

    return predictions


y_pred = knn(X_train, y_train, X_test, k)

print(classification_report(y_test, y_pred))
