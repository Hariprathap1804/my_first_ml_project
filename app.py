import streamlit as st
import pickle
import numpy as np
import json

# 1. THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Price Predictor", layout="centered")

# 2. Define loading functions with caching
@st.cache_resource
def load_model():
    with open('banglore_home_prices_model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_columns():
    with open("columns.json", "r") as f:
        data = json.load(f)
        return data['data_columns']

# 3. Load the assets
try:
    model = load_model()
    data_columns = load_columns()
except Exception as e:
    st.error(f"Missing required files: {e}")
    st.stop() # Stops the app if files aren't found

# --- UI Layout ---
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Enter property details to get an estimated market price.")
st.write("---")

# Input Fields
location_input = st.text_input("Location", placeholder="e.g. 1st Phase JP Nagar")

col1, col2, col3 = st.columns(3)

with col1:
    bhk = st.number_input("BHK", min_value=1, value=2, step=1)

with col2:
    bath = st.number_input("Bathrooms", min_value=1, value=2, step=1)

with col3:
    sqft = st.number_input("Total Sqft", min_value=100, value=1000, step=50)

st.write("---")

if st.button("Calculate Price", use_container_width=True):
    if not location_input:
        st.warning("Please type a location name.")
    else:
        # Match location logic
        try:
            loc_index = data_columns.index(location_input.lower().strip())
        except (ValueError, AttributeError):
            loc_index = -1

        # Create input array
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        
        if loc_index >= 0:
            x[loc_index] = 1

        # Predict
        prediction = model.predict([x])[0]
        
        # Display Result
        if prediction < 0:
            st.error("The model couldn't provide a valid estimate for these inputs.")
        else:
            st.balloons()
            st.success(f"### Estimated Price: ₹ {round(prediction, 2)} Lakhs")

# Helpful Info
with st.expander("Technical Details"):
    st.info(f"Model is trained on {len(data_columns)} features including specific Bangalore locations.")