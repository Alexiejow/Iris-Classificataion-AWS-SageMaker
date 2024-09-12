import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['species'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed datasets
X_train.to_csv('train_features.csv', index=False)
X_test.to_csv('test_features.csv', index=False)
y_train.to_csv('train_labels.csv', index=False)
y_test.to_csv('test_labels.csv', index=False)
