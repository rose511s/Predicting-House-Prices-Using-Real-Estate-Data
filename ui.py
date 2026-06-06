import streamlit as st
import requests


st.set_page_config(page_title="Predicting House Prices Using Real Estate Data", page_icon="🏠", layout="centered")


st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 18px; color: #6B7280; text-align: center; margin-bottom: 30px; }
    .price-box { background-color: #E0F2FE; border-left: 5px solid #0284C7; padding: 20px; border-radius: 8px; text-align: center; }
    .price-text { font-size: 32px; font-weight: bold; color: #0369A1; }
    </style>
""", unsafe_allow_html=True)


st.markdown("<div class='main-title'>🏠 Real Estate Evaluation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict property market values instantly using Machine Learning</div>", unsafe_allow_html=True)
st.divider()

st.subheader("📋 Enter Property Characteristics")

# Create a clean layout with columns
col1, col2 = st.columns(2)

with col1:
    size = st.number_input("Property Size (Sq. Ft.)", min_value=500, max_value=10000, value=2000, step=50)
    bedrooms = st.selectbox("Total Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.selectbox("Total Bathrooms", [1, 2, 3])
    age = st.number_input("Age of the Property (Years)", min_value=0, max_value=150, value=10)

with col2:
    location = st.selectbox("Geographical Location", ["CityA", "CityB", "CityC", "CityD"])
    condition = st.selectbox("Property Condition", ["New", "Good", "Fair", "Poor"])
    prop_type = st.selectbox("Property Type", ["Single Family", "Condominium", "Townhouse"])
    month_sold = st.slider("Target Month of Sale", min_value=1, max_value=12, value=6)

st.divider()

# Prediction Action
if st.button("🔮 Calculate Estimated Market Price", type="primary", use_container_width=True):
 
    payload = {
        "Size": float(size),
        "Bedrooms": float(bedrooms),
        "Bathrooms": float(bathrooms),
        "Age_At_Sale": float(age),
        "Month_Sold": int(month_sold),
        "Location": location,
        "Condition": condition,
        "Type": prop_type
    }
    
    try:

        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            estimated_price = result["estimated_price"]
            
            # Display the result beautifully
            st.markdown(f"""
                <div class='price-box'>
                    <p style='margin:0; font-size:16px; color:#555;'>Predicted Valuation</p>
                    <div class='price-text'>${estimated_price:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ The API returned an error. Make sure your FastAPI server is running.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to the backend server! Please make sure app.py is running via uvicorn.")