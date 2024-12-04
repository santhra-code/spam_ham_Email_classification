import pandas as pd
import chardet

# Step 1: Attempt to read the file with different encodings or auto-detect it

file_path = 'cleaned_data.csv'

# Try to detect encoding with chardet
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())  # Detect encoding
    print(f"Detected Encoding: {result['encoding']}")

# Step 2: Try reading the file with detected encoding
try:
    df = pd.read_csv(file_path, encoding=result['encoding'])
    print("File loaded successfully with detected encoding.")
except UnicodeDecodeError:
    print(f"Error with encoding: {result['encoding']}, trying other encodings.")
    
    # Step 3: If chardet fails, try common encodings
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print("File loaded successfully with ISO-8859-1 encoding.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
            print("File loaded successfully with latin1 encoding.")
        except UnicodeDecodeError:
            print("Unable to load file with standard encodings. Skipping malformed characters.")
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')

# Step 4: Check the data
print(f"First few rows of the dataset:\n{df.head()}")

# Optionally drop duplicates if needed
df = df.drop_duplicates()
print(f"Dataset after dropping duplicates:\n{df.head()}")
