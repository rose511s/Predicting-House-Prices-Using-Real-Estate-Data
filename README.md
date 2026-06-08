[README.md](https://github.com/user-attachments/files/28671447/README.md)
Predicting House Prices Using Real Estate Data

So basically an end-to-end Machine Learning system built to forecast house prices , from real estate transaction data. The project goes through a dataset with more than 247,000 records , then tunes the prediction behavior using a fast and solid **LightGBM Regressor**. After that it brings in a production-ready **FastAPI backend**, and there’s also a *Streamlit interface*, so you can get real-time predictions, in a more interactive kind of way.

---

##  Key Features & Architectural Highlights
* **Gradient Boosted Intelligence:** Uses LightGBM in a step-by-step way to patch model mistakes one after another, beating the usual linear baselines or even a default random forest set up.
* **Multi-Core Scaling:** Explicitly configured so the heavy matrix computations get spread across all the available CPU cores, relies on built-in parallelization (`n_jobs=-1`).  
* **Feature Synergy:** Built layered interaction variables like `Age_At_Sale`, `Total_Rooms`, `Size_Per_Room` , aiming to describe the space layout more realistically and reflect asset depreciation patterns.
* **Decoupled Full-Stack Architecture:** Keeps the training-heavy side apart from real-time prediction delivery, using a lightweight HTTP endpoint plus a reactive frontend, so it all stays responsive and clean.
---

##  Project Structure
```
├── script.py                # Cleans raw data, engineers features, and saves the trained model
├── app.py                   # FastAPI backend server exposing the prediction endpoint
├── ui.py                    # Streamlit frontend user interface
├── house_model.pkl          # Saved LightGBM serialization weights
├── model_features.pkl       # Saved tracking index of exact feature schemas
└── requirements.txt         # Project dependencies environment file

```

Local API Setup:
---
*->python -m uvicorn app:app --reload
*->Navigate to http://127.0.0.1:8000/docs access the interactive Swagger UI
Local Streamlit Frontend:
---
**->python -m streamlit run ui.py
**->Navigate to http://localhost:8501 
---
Architecture Option 2: Unified Cloud Deployment
Live public Application link --> https://predicting-house-prices-using-real-estate-data.streamlit.app/

