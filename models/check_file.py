import pickle
import sys
import sklearn

print(f"Python version: {sys.version}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Try different approaches to load the model
print("\n1. Trying standard pickle.load:")
try:
    with open('models/logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✓ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n2. Trying with protocol specification:")
try:
    with open('models/logistic_model.pkl', 'rb') as file:
        model = pickle.load(file, encoding='latin1')
    print("✓ Model loaded successfully with latin1 encoding!")
    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n3. Trying with joblib (alternative to pickle):")
try:
    import joblib
    model = joblib.load('models/logistic_model.pkl')
    print("✓ Model loaded successfully with joblib!")
    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
except ImportError:
    print("✗ joblib not available")
except Exception as e:
    print(f"✗ Error: {e}") 