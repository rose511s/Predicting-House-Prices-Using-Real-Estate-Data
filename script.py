import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import joblib


df = pd.read_excel("Data.xlsx", sheet_name='Data')

print(f"Original Data Shape: {df.shape}")

df = df.dropna(subset=['Price'])

df['Size'] = df['Size'].fillna(df['Size'].median())
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())

df['Age_At_Sale'] = df['Date Sold'].dt.year - df['Year Built']
df['Age_At_Sale'] = df['Age_At_Sale'].fillna(df['Age_At_Sale'].median())
df['Month_Sold'] = df['Date Sold'].dt.month

df['Condition'] = df['Condition'].fillna(df['Condition'].mode()[0])

df_cleaned = df.drop(columns=['Property ID', 'Date Sold', 'Year Built'])

df_final = pd.get_dummies(df_cleaned, columns=['Location', 'Condition', 'Type'], drop_first=True)

df_final.to_csv("cleaned_house_data.csv", index=False)
print("Data Cleaned!")




df = pd.read_csv("cleaned_house_data.csv")


df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
df['Size_Per_Room'] = df['Size'] / (df['Total_Rooms'] + 0.1)


X = df.drop(columns=['Price'])
y = df['Price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training an Advanced LightGBM Regressor...")
hyper_model = lgb.LGBMRegressor(
    n_estimators=300,      
    learning_rate=0.05,    
    max_depth=8,
    num_leaves=31,
    random_state=42,
    n_jobs=-1            
)

hyper_model.fit(X_train, y_train)

predictions = hyper_model.predict(X_test)
new_r2 = r2_score(y_test, predictions)
new_mae = mean_absolute_error(y_test, predictions)


print(f"New Upgraded R-squared Score: {new_r2:.4f}")
print(f"New Mean Absolute Error: ${new_mae:.2f}")

joblib.dump(hyper_model, "house_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")