# train_models_classical.py
import os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "ai_financial_market_daily_realistic_synthetic.csv"
FEATURE_COLS = ["R&D_Spending_USD_Mn", "AI_Revenue_USD_Mn", "AI_Revenue_Growth_%", "Company"]
TARGET_COL = "Stock_Impact_%"
TEST_SIZE = 0.2
RANDOM_STATE = 42

if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found")

df = pd.read_csv(CSV)
df = df.dropna(subset=[TARGET_COL])
# Save a historical aggregated file for the UI (group by Company and date if available)
# If CSV has a date column use it; else use index as pseudo-date
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    hist = df.sort_values('date').groupby(['Company','date']).agg({
        'AI_Revenue_USD_Mn':'mean',
        'AI_Revenue_Growth_%':'mean',
        'Stock_Impact_%':'mean'
    }).reset_index()
else:
    # create a synthetic time index so charts can display a trend
    df['_time_index'] = df.groupby('Company').cumcount()
    hist = df.groupby(['Company','_time_index']).agg({
        'AI_Revenue_USD_Mn':'mean',
        'AI_Revenue_Growth_%':'mean',
        'Stock_Impact_%':'mean'
    }).reset_index()

hist_json = {}
for c, g in hist.groupby('Company'):
    g_sorted = g.sort_values('date') if 'date' in g.columns else g.sort_values('_time_index')
    hist_json[c] = {
        "x": (g_sorted['date'].dt.strftime('%Y-%m-%d').tolist() if 'date' in g_sorted.columns else g_sorted['_time_index'].astype(str).tolist()),
        "AI_Revenue_USD_Mn": g_sorted['AI_Revenue_USD_Mn'].fillna(0).tolist(),
        "AI_Revenue_Growth_%": g_sorted['AI_Revenue_Growth_%'].fillna(0).tolist(),
        "Stock_Impact_%": g_sorted['Stock_Impact_%'].fillna(0).tolist()
    }

with open("historical_data.json", "w", encoding="utf-8") as f:
    json.dump(hist_json, f, indent=2)
print("Saved historical_data.json")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

le = LabelEncoder()
X["Company"] = le.fit_transform(X["Company"].astype(str))
joblib.dump(le, "label_encoder.pkl")
print("Saved label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")
print("Saved scaler.pkl")

models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(kernel="rbf"),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=RANDOM_STATE)
}

results = {}
residuals_for_plot = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    results[name] = {"R2": float(r2), "MSE": float(mse)}
    joblib.dump(model, name.lower().replace(" ", "_") + ".pkl")
    # residuals
    residuals_for_plot[name] = (y_test.values - preds).tolist()
    print(f" -> {name}: R2={r2:.4f}, MSE={mse:.4f}")

# Save best model
best_name = max(results, key=lambda k: results[k]["R2"])
joblib.dump(models[best_name], "best_model.pkl")
print("Saved best_model.pkl ->", best_name)

# Feature importance
fi = None
if "Gradient Boosting Regressor" in models:
    try:
        fi = models["Gradient Boosting Regressor"].feature_importances_.tolist()
    except Exception:
        fi = None

out = {
    "results": results,
    "best_model": best_name,
    "feature_columns": FEATURE_COLS,
    "feature_importance": fi
}
with open("model_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Saved model_results.json")

# make residual plots into static/residuals/
os.makedirs("static/residuals", exist_ok=True)
for name, res in residuals_for_plot.items():
    plt.figure(figsize=(6,3))
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.scatter(range(len(res)), res, s=8, alpha=0.6)
    plt.title(f"Residuals — {name}")
    plt.xlabel("Test sample index")
    plt.ylabel("y_true - y_pred")
    plt.tight_layout()
    filename = f"static/residuals/{name.lower().replace(' ','_')}_residuals.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print("Saved", filename)

print("Training complete.")
