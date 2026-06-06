[README.md](https://github.com/user-attachments/files/28671447/README.md)
Predicting House Prices Using Real Estate Data

An end-to-end Machine Learning system engineered to predict house prices using real estate transaction data. This project processes a dataset of over 247,000 records, optimizes prediction patterns utilizing a high-performance **LightGBM Regressor**, exposes a production-ready **FastAPI** backend, and provides an interactive **Streamlit** user interface for seamless, real-time predictions.

---

## 🚀 Key Features & Architectural Highlights
* **Gradient Boosted Intelligence:** Implements LightGBM to sequentially correct model errors, surpassing standard linear or baseline random forest techniques.
* **Multi-Core Scaling:** Configured explicitly to distribute heavy matrix computations across all available CPU cores using native parallelization.
* **Feature Synergy:** Engineered complex interaction variables (`Age_At_Sale`, `Total_Rooms`, `Size_Per_Room`) to capture structural space layout and asset depreciation trends.
* **Decoupled Full-Stack Architecture:** Separates the heavy model training workload from real-time prediction serving via a lightweight HTTP API and reactive frontend.

---

## 📂 Project Structure
```text
├── script.py                # Cleans raw data, engineers features, and saves the trained model
├── app.py                   # FastAPI backend server exposing the prediction endpoint
├── ui.py                    # Streamlit frontend user interface
├── house_model.pkl          # Saved LightGBM serialization weights
├── model_features.pkl       # Saved tracking index of exact feature schemas
└── requirements.txt         # Project dependencies environment file
