# 🏠 Group 6 — House Price Predictor
**Ames, Iowa Housing Dataset | XGBoost | RMSLE: 0.1108 | R²: 93.1%**

---

## 📁 Files Needed in This Repository

| File | Where to get it |
|------|----------------|
| `app.py` | Already here ✅ |
| `requirements.txt` | Already here ✅ |
| `xgb_model.pkl` | Export from Colab (see below) |
| `label_encoders.pkl` | Export from Colab (see below) |
| `feature_cols.pkl` | Export from Colab (see below) |
| `test_clean.csv` | Export from Colab (see below) |

---

## 🔧 Step 1 — Export Files from Google Colab

Add this cell at the **end** of your notebook and run it:

```python
import joblib
from google.colab import files

# Save model artifacts
joblib.dump(xgb, 'xgb_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(feature_cols.tolist() if hasattr(feature_cols, 'tolist') else list(feature_cols), 'feature_cols.pkl')

# Save cleaned test set (with original string columns, before encoding)
test_clean.to_csv('test_clean.csv', index=False)

# Download all files
files.download('xgb_model.pkl')
files.download('label_encoders.pkl')
files.download('feature_cols.pkl')
files.download('test_clean.csv')
```

---

## 🐙 Step 2 — Push to GitHub

1. Create a **new public repository** on GitHub (e.g. `group6-house-prices`)
2. Upload all 6 files:
   - `app.py`
   - `requirements.txt`
   - `xgb_model.pkl`
   - `label_encoders.pkl`
   - `feature_cols.pkl`
   - `test_clean.csv`

---

## 🚀 Step 3 — Deploy on Streamlit Cloud (Free)

1. Go to **https://streamlit.io/cloud**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select:
   - Repository: `your-username/group6-house-prices`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **"Deploy"**
6. Wait ~2 minutes — you'll get a live URL like:
   `https://your-username-group6-house-prices-app-xxxxx.streamlit.app`

---

## ✅ What the App Does

**Tab 1 — Predict by House ID**
- Enter any House ID from 1461–2919
- App loads all 80+ features automatically
- XGBoost predicts the price
- Shows neighborhood, quality, size, and more

**Tab 2 — Predict by Features**
- Manually enter quality, size, neighborhood, etc.
- App computes all 6 engineered features automatically
- Shows the predicted price with a confidence range

---

## 🛠 Tech Stack
Python · XGBoost · Scikit-learn · Streamlit · Pandas · NumPy · Joblib
