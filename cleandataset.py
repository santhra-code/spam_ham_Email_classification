# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess your dataset
data = pd.read_csv('cleaned_data.csv')  # Replace with your file path
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})  # Map ham to 0, spam to 1

# Text preprocessing (optional improvements)
data['Message'] = data['Message'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove non-alphabets

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['Message']).toarray()  # Convert text to numerical features
y = data['Category']  # Corrected this line to use 'Category' as the target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define class weights for spam and ham
class_weights = {0: 1, 1: 5}

# Train Random Forest model
model = RandomForestClassifier(class_weight=class_weights, random_state=42, n_estimators=200)
model.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for spam class

# Adjust threshold
threshold = 0.2
y_pred_threshold = (y_pred_prob >= threshold).astype(int)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_threshold))
print("\nClassification Report:\n", classification_report(y_test, y_pred_threshold))
