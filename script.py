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

"""
#Random Forest Model Evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv("cleaned_house_data.csv")

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f}")

joblib.dump(model, "house_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
"""






# LightGBM Model Evaluation
df = pd.read_csv("cleaned_house_data.csv")


df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
df['Size_Per_Room'] = df['Size'] / (df['Total_Rooms'] + 0.1)


X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training an Advanced LightGBM Regressor")
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


print(f"R-squared Score: {new_r2:.4f}")
print(f"Mean Absolute Error: ${new_mae:.2f}")

joblib.dump(hyper_model, "house_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")



"""
#Plotting Predictive Features
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys 

model_path = "house_model.pkl"
features_path = "model_features.pkl"


model = joblib.load(model_path)
feature_names = joblib.load(features_path)

importances = model.feature_importances_

feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_imp_df, palette='Blues_r', legend=False)
plt.title('Top 10 Most Predictive Features in Real Estate Valuation', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance Score', fontsize=12)
plt.ylabel('Property Features', fontsize=12)
plt.tight_layout()

plt.savefig('feature_importance.png', dpi=300)
plt.show()




#Model Prediction Accuracy using 2000 Sample points
import matplotlib.ticker as mticker

plt.figure(figsize=(8, 8))

sample_idx = np.random.choice(len(y_test), 2000, replace=False)
y_test_sample = np.array(y_test)[sample_idx]
pred_sample = np.array(predictions)[sample_idx]

plt.scatter(y_test_sample, pred_sample, alpha=0.4, color='#0284C7', edgecolors='none')

ideal_line = [min(y_test_sample), max(y_test_sample)]
plt.plot(ideal_line, ideal_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction Identity')

plt.title('Model Prediction Accuracy: Actual vs. Predicted Prices', fontsize=14, fontweight='bold')
plt.xlabel('Actual Historical Price ($)', fontsize=12)
plt.ylabel('Model Estimated Price ($)', fontsize=12)
plt.legend()

ax = plt.gca()

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()

plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()
