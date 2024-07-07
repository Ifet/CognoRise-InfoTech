#Import necessary libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset 
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


# Check for missing values
print(df.isnull().sum())

# Split the dataset into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Tune the hyperparameter k using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
grid_search.fit(X_train, y_train)

# Get the best parameter and best score
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print(f"Best k: {best_k}")
print(f"Best cross-validated score: {best_score}")

# Train the model with the best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Evaluate the tuned model
y_pred_best = knn_best.predict(X_test)
print("Confusion Matrix (Tuned Model):\n", confusion_matrix(y_test, y_pred_best))
print("Classification Report (Tuned Model):\n", classification_report(y_test, y_pred_best))
print("Accuracy Score (Tuned Model):", accuracy_score(y_test, y_pred_best))



import pandas as pd  # (Optional) For data exploration, if needed
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the WDBC dataset (built-in with scikit-learn)
data = load_breast_cancer()

# Extract features and target
X = data.data  # Features
y = data.target  # Target (malignant or benign)

# Split data into training and testing sets (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a KNN classifier (optimized hyperparameter based on common practice)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print predicted labels (cancerous or non-cancerous) for the first 10 samples (optional)
for i in range(10):
    predicted_label = "cancerous" if y_pred[i] == 1 else "non-cancerous"
    print(f"Sample {i+1}: Predicted: {predicted_label}")
