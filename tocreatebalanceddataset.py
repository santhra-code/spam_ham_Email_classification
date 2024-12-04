# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
data = pd.read_csv('cleaned_data.csv', encoding='latin1')

# Rename columns for clarity
data = data.rename(columns={'Category': 'label', 'Message': 'text'})

# Drop unnecessary columns
data = data[['label', 'text']]  # Keep only the relevant columns

# Map labels to numerical values (spam = 1, ham = 0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Remove rows with missing or empty text
data = data.dropna(subset=['text'])  # Drop rows with NaN in 'text'
data = data[data['text'].str.strip() != '']  # Drop rows with empty or whitespace text

# Debug: Check the cleaned data
print("Sample cleaned data:")
print(data.head())

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, min_df=1)
X = vectorizer.fit_transform(data['text']).toarray()  # Convert text to numerical features
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define class weights (spam gets a higher weight)
class_weights = {0: 1, 1: 5}

# Train the model
model = RandomForestClassifier(class_weight=class_weights, random_state=42)
model.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
