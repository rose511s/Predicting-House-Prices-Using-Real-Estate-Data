from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. INITIALIZE THE APP FIRST (This fixes the NameError!)
app = FastAPI(title="Agent Mira Real Estate Prediction API")

print("Loading trained model and features from disk...")
# Load the files you just downloaded from Google Colab
model = joblib.load("house_model.pkl")
model_features = joblib.load("model_features.pkl")
print("Model assets loaded successfully!")

# Define exactly what information the API expects to receive from a user
class PropertyData(BaseModel):
    Size: float
    Bedrooms: float
    Bathrooms: float
    Age_At_Sale: float
    Month_Sold: int
    Location: str     # Expects: 'CityA', 'CityB', 'CityC', or 'CityD'
    Condition: str    # Expects: 'Good', 'New', 'Fair', or 'Poor'
    Type: str         # Expects: 'Single Family', 'Condominium', or 'Townhouse'

@app.get("/")
def home():
    return {"status": "Online", "message": "Welcome to the Property Price Prediction API"}

@app.post("/predict")
def predict(data: PropertyData):
    # 1. Create a baseline row of zeros matching our brand new training column layout
    input_row = {col: 0 for col in model_features}
    
    # 2. Fill in the standard numeric values
    input_row['Size'] = data.Size
    input_row['Bedrooms'] = data.Bedrooms
    input_row['Bathrooms'] = data.Bathrooms
    input_row['Age_At_Sale'] = data.Age_At_Sale
    input_row['Month_Sold'] = data.Month_Sold
    
    # --- SILENT MATHEMATICAL CALCULATIONS FOR THE LIGHTGBM MODEL ---
    total_rooms = data.Bedrooms + data.Bathrooms
    input_row['Total_Rooms'] = total_rooms
    input_row['Size_Per_Room'] = data.Size / (total_rooms + 0.1)
    # ---------------------------------------------------------------
    
    # 3. Match the text categories to the encoded columns
    loc_col = f"Location_{data.Location}"
    cond_col = f"Condition_{data.Condition}"
    type_col = f"Type_{data.Type}"
    
    # LightGBM converted spaces in column names to underscores, let's account for that:
    if loc_col.replace(" ", "_") in input_row: input_row[loc_col.replace(" ", "_")] = 1
    if cond_col.replace(" ", "_") in input_row: input_row[cond_col.replace(" ", "_")] = 1
    if type_col.replace(" ", "_") in input_row: input_row[type_col.replace(" ", "_")] = 1
    
    # 4. Convert into a DataFrame row and get the prediction
    df_input = pd.DataFrame([input_row])
    predicted_price = model.predict(df_input)[0]
    
    return {"estimated_price": round(predicted_price, 2)}