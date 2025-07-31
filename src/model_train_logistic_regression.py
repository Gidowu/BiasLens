import joblib
from sklearn.linear_model import LogisticRegression

# Load the processed training data
X_train = joblib.load('processed/X_train.pkl')
y_train = joblib.load('processed/y_train.pkl')

# Initialize and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/logistic_model.pkl')
print("Logistic Regression model saved")