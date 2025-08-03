import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and vectorizer
model = joblib.load("models/best_ensemble_model.pkl")
vectorizer = joblib.load("processed/vectorizer.pkl")

# App UI
st.title("🗞️ Political Bias Classifier")
st.write("Enter a news excerpt below to classify its political bias:")

user_input = st.text_area("News Excerpt", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([user_input])
        
        # Predict class label
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Bias: **{prediction}**")

        # Predict probabilities
        try:
            probs = model.predict_proba(X_input)[0]
            labels = model.lr_model.classes_ if hasattr(model, 'lr_model') else model.classes_
            prob_df = pd.DataFrame({
                'Bias': labels,
                'Probability': [f"{p*100:.2f}%" for p in probs]
            })

            # Display as table or bar chart
            st.subheader("Prediction Confidence")
            st.dataframe(prob_df.set_index('Bias'))
        
        except AttributeError:
            st.error("This model does not support probability outputs.")
