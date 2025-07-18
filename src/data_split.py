import joblib
from sklearn.model_selection import train_test_split

# ✅ Step 3: Data Splitting

# 1️⃣ Load feature matrix and labels
X_vectorized = joblib.load('processed/X_vectorized.pkl')
y_labels = joblib.load('processed/y_labels.pkl')

# 2️⃣ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y_labels, test_size=0.2, stratify=y_labels, random_state=42
)

# 3️⃣ Save split datasets
joblib.dump(X_train, 'processed/X_train.pkl')
joblib.dump(X_test, 'processed/X_test.pkl')
joblib.dump(y_train, 'processed/y_train.pkl')
joblib.dump(y_test, 'processed/y_test.pkl')

print("Data splitting complete. Training and testing sets saved.")
