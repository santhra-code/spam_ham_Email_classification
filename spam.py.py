# Import libraries
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset from a CSV file
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Assuming the CSV has 'messages' for email content and 'category' for the labels (ham/spam)
emails = df['v1'].tolist()  # Adjust column name if it's different
labels = df['v2'].tolist()  # Adjust column name if it's different

# Convert 'ham' to 0 and 'spam' to 1
labels = [1 if label == 'spam' else 0 for label in labels]

# Text preprocessing and vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)
y = labels

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the model and vectorizer
with open('spam123.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vector123.pkl', 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)

print("Model and vectorizer saved successfully!")
