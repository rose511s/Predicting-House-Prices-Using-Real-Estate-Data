import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 18px; color: #6B7280; text-align: center; margin-bottom: 30px; }
    .price-box { background-color: #E0F2FE; border-left: 5px solid #0284C7; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px; }
    .price-text { font-size: 32px; font-weight: bold; color: #0369A1; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Predicting House PRices USing Real Estate Data</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict property market values instantly using Machine Learning</div>", unsafe_allow_html=True)
st.divider()

@st.cache_resource  
def load_ml_model():
    model = joblib.load("house_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, model_features

try:
    model, model_features = load_ml_model()
    st.success("✅ Machine Learning Model Active on Cloud!")
except Exception as e:
    st.error(f"❌ Error loading model assets: {e}")

st.subheader("📋 Enter Property Characteristics")

col1, col2 = st.columns(2)

with col1:
    size = st.number_input("Property Size (Sq. Ft.)", min_value=500, max_value=10000, value=2000, step=50)
    bedrooms = st.selectbox("Total Bedrooms", [1, 2, 3, 4, 5], index=2)
    bathrooms = st.selectbox("Total Bathrooms", [1, 2, 3], index=1)
    age = st.number_input("Age of the Property (Years)", min_value=0, max_value=150, value=10)

with col2:
    location = st.selectbox("Geographical Location", ["CityA", "CityB", "CityC", "CityD"])
    condition = st.selectbox("Property Condition", ["New", "Good", "Fair", "Poor"], index=1)
    prop_type = st.selectbox("Property Type", ["Single Family", "Condominium", "Townhouse"])
    month_sold = st.slider("Target Month of Sale", min_value=1, max_value=12, value=6)

st.divider()

if st.button("🔮 Calculate Estimated Market Price", type="primary", use_container_width=True):
    
    input_row = {col: 0 for col in model_features}
    
    input_row['Size'] = float(size)
    input_row['Bedrooms'] = float(bedrooms)
    input_row['Bathrooms'] = float(bathrooms)
    input_row['Age_At_Sale'] = float(age)
    input_row['Month_Sold'] = int(month_sold)
    
    total_rooms = float(bedrooms) + float(bathrooms)
    input_row['Total_Rooms'] = total_rooms
    input_row['Size_Per_Room'] = float(size) / (total_rooms + 0.1)
    
    loc_col = f"Location_{location}".replace(" ", "_")
    cond_col = f"Condition_{condition}".replace(" ", "_")
    type_col = f"Type_{prop_type}".replace(" ", "_")
    
    if loc_col in input_row: input_row[loc_col] = 1
    if cond_col in input_row: input_row[cond_col] = 1
    if type_col in input_row: input_row[type_col] = 1
    
    df_input = pd.DataFrame([input_row])
    predicted_price = model.predict(df_input)[0]
    
    st.markdown(f"""
        <div class='price-box'>
            <p style='margin:0; font-size:16px; color:#555;'>Estimated Property Evaluation</p>
            <div class='price-text'>${predicted_price:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)
