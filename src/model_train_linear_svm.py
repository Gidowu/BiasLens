import joblib
from sklearn.svm import LinearSVC

# Load the processed training data
X_train = joblib.load('processed/X_train.pkl')
y_train = joblib.load('processed/y_train.pkl')

# Initialize and train Linear SVM model
# Using default parameters for fair comparison, but can be tuned
model = LinearSVC(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/linear_svm_model.pkl')
print("Linear SVM model saved")