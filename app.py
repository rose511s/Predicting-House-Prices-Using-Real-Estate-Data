from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Agent Mira Real Estate Prediction API")

print("Loading trained model and features")
model = joblib.load("house_model.pkl")
model_features = joblib.load("model_features.pkl")
print("Model assets loaded successfully!")

class PropertyData(BaseModel):
    Size: float
    Bedrooms: float
    Bathrooms: float
    Age_At_Sale: float
    Month_Sold: int
    Location: str     
    Condition: str   
    Type: str       

@app.get("/")
def home():
    return {"status": "Online", "message": "Welcome to the Property Price Prediction API"}

@app.post("/predict")
def predict(data: PropertyData):
    input_row = {col: 0 for col in model_features}
    
    input_row['Size'] = data.Size
    input_row['Bedrooms'] = data.Bedrooms
    input_row['Bathrooms'] = data.Bathrooms
    input_row['Age_At_Sale'] = data.Age_At_Sale
    input_row['Month_Sold'] = data.Month_Sold
    
    total_rooms = data.Bedrooms + data.Bathrooms
    input_row['Total_Rooms'] = total_rooms
    input_row['Size_Per_Room'] = data.Size / (total_rooms + 0.1)
    
    loc_col = f"Location_{data.Location}"
    cond_col = f"Condition_{data.Condition}"
    type_col = f"Type_{data.Type}"
    
    if loc_col.replace(" ", "_") in input_row: input_row[loc_col.replace(" ", "_")] = 1
    if cond_col.replace(" ", "_") in input_row: input_row[cond_col.replace(" ", "_")] = 1
    if type_col.replace(" ", "_") in input_row: input_row[type_col.replace(" ", "_")] = 1
    
    df_input = pd.DataFrame([input_row])
    predicted_price = model.predict(df_input)[0]
    
    return {"estimated_price": round(predicted_price, 2)}
