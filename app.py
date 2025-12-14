import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# --- Configuration ---
st.set_page_config(page_title="Iris Species Predictor", layout="centered")

# --- Load Model and LabelEncoder ---
# The @st.cache_resource decorator ensures the model is loaded only once
# when the app starts, improving performance.
@st.cache_resource
def load_assets():
    """Loads the trained Decision Tree model and LabelEncoder."""
    
    # Define the path to the model and encoder files
    # These paths assume the .pkl files are saved in a folder named 'models'
    model_path = os.path.join('models', 'DecisionTree_iris.pkl')
    le_path = os.path.join('models', 'LabelEncoder_iris.pkl')
    
    try:
        # Load the Decision Tree model
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Load the LabelEncoder
        with open(le_path, 'rb') as le_file:
            le = pickle.load(le_file)
            
        return model, le
    except FileNotFoundError:
        st.error(f"Error: Model files not found. Please ensure 'DecisionTree_iris.pkl' and 'LabelEncoder_iris.pkl' are in the '{os.path.dirname(model_path)}' directory.")
        return None, None

model, le = load_assets()

# --- Application Title and Description ---
st.title("🌺 Iris Species Predictor")
st.markdown("""
    This app uses a pre-trained **Decision Tree Classifier** to predict the species of an Iris flower 
    based on its sepal and petal measurements.
""")

if model is not None and le is not None:
    # --- Sidebar for Input Features ---
    st.sidebar.header("Input Features (cm)")
    
    # Input sliders for the four features 
    sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.4, 0.1)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 1.3, 0.1)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.6, 0.2, 0.1)

    # --- Prepare Data for Prediction ---
    # The model was trained with the 'Id' column, so we must include it as a placeholder.
    input_data = {
        'Id': [151], # Arbitrary ID for the new observation
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    }
    
    # Create the DataFrame for prediction
    features_df = pd.DataFrame(input_data)
    
    st.subheader("User Input Data (Features used for model)")
    # Show the features, but hide the placeholder 'Id' column
    st.dataframe(features_df.drop('Id', axis=1), use_container_width=True, hide_index=True)
    
    # --- Prediction Logic ---
    if st.button('Predict Species', type="primary"):
        
        # 1. Make the prediction (outputs encoded label: 0, 1, or 2)
        prediction_encoded = model.predict(features_df)
        
        # 2. Decode the prediction (outputs the original species name)
        prediction_species = le.inverse_transform(prediction_encoded)
        
        st.subheader("Prediction Result")
        st.success(f"The predicted Iris species is: **{prediction_species[0]}**")
        
        # Optional: Show prediction probabilities for all classes
        prediction_proba = model.predict_proba(features_df)
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=le.classes_,
            index=['Probability']
        ).T.style.format({'Probability': "{:.2f}"})
        
        st.subheader("Confidence Score (Probability)")
        st.dataframe(proba_df, use_container_width=True)

# --- End of Streamlit app ---