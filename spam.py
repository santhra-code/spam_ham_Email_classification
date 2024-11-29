# Import libraries
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
emails = [
    "Congratulations, you have won a lottery!", 
    "Hey, are you free tomorrow?", 
    "Win money easily by clicking this link.", 
    "Let's have lunch together this weekend.",
    "Your account is at risk. Click to secure it now.",
    "Do not miss this opportunity to earn $5000 weekly!",
    "Can we reschedule our meeting?",
    "Best discount deals just for you!"
]
labels = [1, 0, 1, 0, 1, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

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