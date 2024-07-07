import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Load datasets (replace with your file paths)
mapping = pd.read_csv('mapping.csv')  # Assuming mapping is in CSV format
mapping_dict = dict(zip(mapping['number'], mapping['emoticons']))  # Assuming the columns are 'number' and 'emoticons'
output_format = pd.read_csv('outputformat.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
mapping_dict = dict(zip(mapping['number'], mapping['emoticons']))  # Assuming mapping is a dictionary
output_format = pd.read_csv('outputformat.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocess text (adjust based on your data)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

train_data['TEXT'] = train_data['TEXT'].apply(preprocess_text)
test_data['TEXT'] = test_data['TEXT'].apply(preprocess_text)

# Separate text and emojis
X_train = train_data['TEXT']
y_train = train_data['Label']  # Assuming 'label' is the emoji column
X_test = test_data['TEXT']

# Feature extraction (TfidfVectorizer for text representation)
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train a model (Logistic Regression for simplicity)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)  # Increase max_iter (adjust as needed)
model.fit(X_train_features, y_train)

# Function to predict emojis (including emoticon mapping)
def predict_emoji(text):
    # Check for emoticons first (if applicable)
    if text in mapping_dict:
        return mapping_dict[text]  # Return emoji from mapping if found

    # If no emoticon match, proceed with model prediction
    text_features = vectorizer.transform([preprocess_text(text)])
    predicted_emoji = model.predict(text_features)[0]
    return predicted_emoji
# After training the model:
predicted_emojis = model.predict(X_test_features)

# Option 1: Print all predictions (adjust for loop if needed)
for i in range(len(predicted_emojis)):
    print(f"Predicted emoji for text {i+1}: {predicted_emojis[i]}")

# Option 2: Print prediction for a specific text sample
sample_text = "Another win! Great job mighty mights ï¸12-6 @ PPO Football"
sample_features = vectorizer.transform([preprocess_text(sample_text)])
predicted_emoji = model.predict(sample_features)[0]
print(f"Predicted emoji for '{sample_text}': {predicted_emoji}")
