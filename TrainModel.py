from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load data
data = pd.read_csv('cleaned_data.csv')  # Replace with your file path
data['label'] = data['Category'].map({'ham': 0, 'spam': 1})  # Map ham to 0, spam to 1
data['text'] = data['Message'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['text']).toarray()
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define models to try
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200),
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=5, random_state=42),
    "LightGBM": lgb.LGBMClassifier(class_weight='balanced', random_state=42),
    "SVM": SVC(class_weight='balanced', random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))