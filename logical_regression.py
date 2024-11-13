import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example dataset (X: features, y: labels)
# Let's assume a binary classification problem (y = 0 or 1)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the outputs for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the predicted outputs and accuracy
print("Predicted y values:", y_pred)
print("Actual y values:", y_test)
print("Accuracy of the model:", accuracy)

# Predict for new input
new_input = np.array([[4, 6]])  # Example input
new_pred = model.predict(new_input)
new_prob = model.predict_proba(new_input)  # Probability of the prediction

print(f"Predicted class for input {new_input}: {new_pred}")
print(f"Probability for input {new_input}: {new_prob}")
