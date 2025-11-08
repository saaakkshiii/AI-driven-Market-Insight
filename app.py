# app.py
"""
Flask AI Market Dashboard (full app.py)
- Login / Register (Flask-Login)
- Predict (loads trained models)
- History (per-user, saved to SQLite)
- Diagnostics, Retrain, Export CSV
- Automatically opens browser when server starts
"""

import os
import json
import csv
import datetime
import traceback
import subprocess
import sqlite3
import webbrowser
import threading

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, g, jsonify
)
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
import joblib
import pandas as pd

# -------------------- Config --------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace_this_with_a_random_secret"  # CHANGE for production

# SQLite DB file
DATABASE = "database.db"

# ML artifact filenames
SCALER_FILE = "scaler.pkl"
LE_FILE = "label_encoder.pkl"
METRICS_FILE = "model_results.json"
TRAIN_SCRIPT = "train_models_classical.py"

# map of model display name -> filename saved by training script
models_map = {
    "Linear Regression": "linear_regression.pkl",
    "Support Vector Regressor": "support_vector_regressor.pkl",
    "Gradient Boosting Regressor": "gradient_boosting_regressor.pkl"
}

# -------------------- Database utilities --------------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            rd REAL,
            revenue REAL,
            growth REAL,
            company TEXT,
            lr_pred REAL,
            svr_pred REAL,
            gbr_pred REAL
        )
    """)
    db.commit()

@app.teardown_appcontext
def close_connection(exc):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# -------------------- Flask-Login setup --------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

def query_user_by_username(username):
    db = get_db()
    cur = db.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cur.fetchone()

def create_user(username, password):
    db = get_db()
    try:
        db.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, password))
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    cur = db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if row:
        return User(row["id"], row["username"])
    return None

# -------------------- ML artifacts loading --------------------
# load scaler & label encoder if present
scaler = joblib.load(SCALER_FILE) if os.path.exists(SCALER_FILE) else None
le = joblib.load(LE_FILE) if os.path.exists(LE_FILE) else None

def load_models():
    models = {}
    for name, path in models_map.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    # fallback: if models missing, try best_model.pkl
    if len(models) < len(models_map) and os.path.exists("best_model.pkl"):
        best = joblib.load("best_model.pkl")
        for k in models_map.keys():
            if k not in models:
                models[k] = best
    return models

models = load_models()
companies = list(le.classes_) if le is not None else []

def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# -------------------- Auth routes --------------------
@app.route("/register", methods=["GET","POST"])
def register():
    init_db()
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        if not username or not password:
            flash("Provide username & password")
            return redirect(url_for("register"))
        if create_user(username, password):
            flash("Account created. Login now.")
            return redirect(url_for("login"))
        else:
            flash("Username already exists.")
            return redirect(url_for("register"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    init_db()
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        row = query_user_by_username(username)
        if row and row["password"] == password:
            user = User(row["id"], row["username"])
            login_user(user)
            flash("Logged in")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out")
    return redirect(url_for("login"))

# -------------------- Main app routes --------------------
@app.route("/", methods=["GET","POST"])
@login_required
def home():
    init_db()
    global models, scaler, le, companies
    result_dict = None
    best_model = None
    metrics = load_metrics()
    recent = []

    if request.method == "POST":
        if scaler is None or le is None or len(models) == 0:
            return "<h3>Server not ready</h3><p>Run training first.</p>", 500
        try:
            rd = float(request.form.get("rd"))
            revenue = float(request.form.get("revenue"))
            growth = float(request.form.get("growth"))
            company = request.form.get("company")
            if company not in companies:
                return "Invalid company", 400

            company_encoded = int(le.transform([company])[0])
            X = pd.DataFrame([{
                "R&D_Spending_USD_Mn": rd,
                "AI_Revenue_USD_Mn": revenue,
                "AI_Revenue_Growth_%": growth,
                "Company": company_encoded
            }])
            X_scaled = scaler.transform(X)

            # reload models
            models = load_models()
            result_dict = {}
            lr = svr = gbr = None
            for name, model in models.items():
                p = float(model.predict(X_scaled)[0])
                result_dict[name] = round(p, 4)
                if "Linear Regression" in name:
                    lr = p
                elif "Support Vector" in name:
                    svr = p
                elif "Gradient Boosting" in name:
                    gbr = p

            if metrics and metrics.get("best_model"):
                best_model = metrics.get("best_model")
            else:
                best_model = max(result_dict, key=result_dict.get)

            # save user-specific history
            db = get_db()
            db.execute(
                "INSERT INTO predictions (user_id, timestamp, rd, revenue, growth, company, lr_pred, svr_pred, gbr_pred) VALUES (?,?,?,?,?,?,?,?,?)",
                (int(current_user.id), datetime.datetime.now().isoformat(), rd, revenue, growth, company, lr, svr, gbr)
            )
            db.commit()

        except Exception:
            return f"<h3>Error during prediction</h3><pre>{traceback.format_exc()}</pre>", 500

    # load recent predictions for user
    db = get_db()
    cur = db.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 8", (int(current_user.id),))
    rows = cur.fetchall()
    recent = [dict(r) for r in rows]

    return render_template("index.html", companies=companies, result_dict=result_dict, best_model=best_model, metrics=metrics, recent=recent)

@app.route("/export_history")
@login_required
def export_history():
    db = get_db()
    cur = db.execute("SELECT timestamp, rd, revenue, growth, company, lr_pred, svr_pred, gbr_pred FROM predictions WHERE user_id = ? ORDER BY id DESC", (int(current_user.id),))
    rows = cur.fetchall()
    if not rows:
        flash("No history to export")
        return redirect(url_for("home"))
    csv_path = f"predictions_user_{current_user.id}.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","rd","revenue","growth","company","lr_pred","svr_pred","gbr_pred"])
        for r in rows:
            writer.writerow([r["timestamp"], r["rd"], r["revenue"], r["growth"], r["company"], r["lr_pred"], r["svr_pred"], r["gbr_pred"]])
    return send_file(csv_path, as_attachment=True)

@app.route("/retrain", methods=["GET","POST"])
@login_required
def retrain():
    if not os.path.exists(TRAIN_SCRIPT):
        flash("Train script missing")
        return redirect(url_for("home"))
    proc = subprocess.run(["python", TRAIN_SCRIPT], capture_output=True, text=True)
    # reload artifacts
    global scaler, le, models, companies
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    if os.path.exists("label_encoder.pkl"):
        le = joblib.load("label_encoder.pkl")
        companies = list(le.classes_)
    models = load_models()
    stdout = proc.stdout or "(no stdout)"
    stderr = proc.stderr or ""
    return f"<h3>Retrain finished</h3><pre>{stdout}</pre><p style='color:red'>{stderr}</p><p><a href='/'>Back</a> | <a href='/diagnostics'>Diagnostics</a></p>"

@app.route("/diagnostics")
@login_required
def diagnostics():
    metrics = load_metrics()
    return render_template("diagnostics.html", metrics=metrics)

@app.route("/historical_data")
@login_required
def historical_data():
    fname = "historical_data.json"
    if not os.path.exists(fname):
        return jsonify({"error":"no historical file"}), 404
    with open(fname, "r", encoding="utf-8") as f:
        d = json.load(f)
    company = request.args.get("company")
    if not company:
        return jsonify({"companies": list(d.keys())})
    if company not in d:
        return jsonify({"error":"company not found", "companies": list(d.keys())}), 404
    return jsonify(d[company])

@app.route("/history")
@login_required
def history():
    db = get_db()
    cur = db.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC", (int(current_user.id),))
    rows = cur.fetchall()
    return render_template("history.html", rows=[dict(r) for r in rows])

# -------------------- Main entry --------------------
if __name__ == "__main__":
    # Open default browser automatically shortly after server starts
    def open_browser():
        try:
            webbrowser.open_new("http://127.0.0.1:5000")
        except Exception:
            pass

    # Start a timer to open browser after 1 second
    threading.Timer(1, open_browser).start()

    # initialize DB inside app context
    with app.app_context():
        init_db()

    # start Flask
    app.run(debug=True)
