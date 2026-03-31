"""
LEGACY FLASK BACKEND - FOR REFERENCE ONLY
==========================================

For Streamlit deployment on Streamlit Cloud or locally, use: app.py

This file is the original Flask REST API backend.
To run this Flask server locally (requires Flask and Flask-CORS):
  
  pip install flask flask-cors
  python app2.py

Then access at: http://localhost:5000

For production Streamlit deployment, use:
  streamlit run app.py
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import time

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------
# 1. Load ALL Trained Machine Learning Models on Startup
# -------------------------------------------------------------
MODELS_DIR = "models"

models = {}
scaler = None

def load_all_models():
    """Load all .pkl models and the keras ANN model from the models directory."""
    global models, scaler
    
    # Load scaler
    scaler_path = os.path.join(MODELS_DIR, "robust_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("RobustScaler loaded successfully.")
    
    # Load sklearn/xgboost/lightgbm models
    model_files = {
        "logistic_regression": "logistic_regression.pkl",
        "random_forest": "random_forest.pkl",
        "xgboost": "xgboost.pkl",
        "lightgbm": "lightgbm.pkl",
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                print(f"Model '{name}' loaded from {filename}")
            except Exception as e:
                print(f"WARNING: Failed to load {filename}: {e}")
    
    # Load Keras ANN model
    ann_path = os.path.join(MODELS_DIR, "ann_model.keras")
    if os.path.exists(ann_path):
        try:
            import tensorflow as tf
            models["ann"] = tf.keras.models.load_model(ann_path)
            print("ANN (Keras) model loaded successfully.")
        except ImportError:
            print("WARNING: TensorFlow not installed. ANN model will not be available.")
        except Exception as e:
            print(f"WARNING: Failed to load ANN model: {e}")
    
    # Load tuned XGBoost if available
    tuned_path = os.path.join(MODELS_DIR, "best_xgb_tuned.pkl")
    if os.path.exists(tuned_path):
        try:
            models["xgboost_tuned"] = joblib.load(tuned_path)
            print("Tuned XGBoost model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Failed to load tuned XGBoost: {e}")

    print(f"\nTotal models loaded: {len(models)}")
    print(f"Available models: {list(models.keys())}")

load_all_models()

# -------------------------------------------------------------
# 2. Serve the Frontend
# -------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -------------------------------------------------------------
# 3. Get Available Models
# -------------------------------------------------------------
@app.route("/api/models", methods=["GET"])
def get_models():
    """Return list of available models."""
    model_info = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "description": "Linear classifier with sigmoid activation",
            "icon": "📊",
            "color": "#38bdf8"
        },
        "random_forest": {
            "name": "Random Forest",
            "description": "Ensemble of decision trees with bagging",
            "icon": "🌲",
            "color": "#10b981"
        },
        "xgboost": {
            "name": "XGBoost",
            "description": "Gradient boosted trees with regularization",
            "icon": "⚡",
            "color": "#f59e0b"
        },
        "lightgbm": {
            "name": "LightGBM",
            "description": "Light gradient boosting with leaf-wise growth",
            "icon": "💡",
            "color": "#8b5cf6"
        },
        "ann": {
            "name": "Neural Network (ANN)",
            "description": "Deep learning with dense layers",
            "icon": "🧠",
            "color": "#ec4899"
        },
        "xgboost_tuned": {
            "name": "XGBoost (Tuned)",
            "description": "Hyperparameter-optimized XGBoost",
            "icon": "🎯",
            "color": "#ef4444"
        }
    }
    
    available = {}
    for key in models:
        if key in model_info:
            available[key] = model_info[key]
    
    return jsonify({"models": available})

# -------------------------------------------------------------
# 4. Predict with ALL Models (Compare)
# -------------------------------------------------------------
@app.route("/predict/all", methods=["POST"])
def predict_all():
    """Run prediction through ALL loaded models and return comparison."""
    try:
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Invalid Input. Expected: {'input': [feat1, feat2, ...]}"}), 400
        
        features = data["input"]
        
        # User explicitly passes: [Time, V1..V28, Amount, Class] (31 items)
        # We need to map it to Model expecting: [scaled_time, scaled_amount, V1..V28, Hour]
        if len(features) == 31:
            time_val = features[0]
            v_features = features[1:29] # 28 elements: V1 to V28
            amount_val = features[29]
            # Class is features[30]
            
            # Feature extraction
            hour_val = int(time_val / 3600) % 24
            
            if scaler is not None:
                try:
                    scaled = scaler.transform(np.array([[time_val, amount_val]]))
                    scaled_time = scaled[0][0]
                    scaled_amount = scaled[0][1]
                except:
                    scaled_time = time_val
                    scaled_amount = amount_val
            else:
                scaled_time = time_val
                scaled_amount = amount_val
                
            model_input = [scaled_time, scaled_amount] + v_features + [hour_val]
            input_array = np.array(model_input, dtype=float).reshape(1, -1)
        else:
            input_array = np.array(features, dtype=float).reshape(1, -1)
        
        results = {}
        for name, model in models.items():
            start = time.time()
            try:
                if name == "ann":
                    # Keras model
                    prob = float(model.predict(input_array, verbose=0)[0][0])
                    pred = 1 if prob >= 0.5 else 0
                    elapsed = round((time.time() - start) * 1000, 2)
                    results[name] = {
                        "prediction": pred,
                        "probability": round(prob, 6),
                        "status": "Safe" if pred == 0 else "Fraud",
                        "inference_time_ms": elapsed
                    }
                else:
                    prediction = model.predict(input_array)
                    pred_val = int(prediction[0])
                    prob = None
                    if hasattr(model, "predict_proba"):
                        prob = round(float(model.predict_proba(input_array)[0][1]), 6)
                    elapsed = round((time.time() - start) * 1000, 2)
                    results[name] = {
                        "prediction": pred_val,
                        "probability": prob,
                        "status": "Safe" if pred_val == 0 else "Fraud",
                        "inference_time_ms": elapsed
                    }
            except Exception as e:
                results[name] = {
                    "prediction": None,
                    "probability": None,
                    "status": f"Error: {str(e)}",
                    "inference_time_ms": 0
                }
        
        # Compute consensus
        predictions = [r["prediction"] for r in results.values() if r["prediction"] is not None]
        fraud_votes = sum(predictions)
        total_votes = len(predictions)
        consensus = {
            "fraud_votes": fraud_votes,
            "normal_votes": total_votes - fraud_votes,
            "total_models": total_votes,
            "verdict": "FRAUD" if fraud_votes > total_votes / 2 else "SAFE",
            "confidence": round(max(fraud_votes, total_votes - fraud_votes) / total_votes * 100, 1) if total_votes > 0 else 0
        }
        
        return jsonify({"results": results, "consensus": consensus}), 200
    
    except ValueError as ve:
        return jsonify({"error": f"Value Error: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# -------------------------------------------------------------
# 5. Predict with Single Model
# -------------------------------------------------------------
@app.route("/predict/<model_name>", methods=["POST"])
def predict_single(model_name):
    """Run prediction through a specific model."""
    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found. Available: {list(models.keys())}"}), 404
    
    try:
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Invalid Input. Expected: {'input': [feat1, feat2, ...]}"}), 400
        
        features = data["input"]
        
        # Mapping User Array [Time, V1..V28, Amount, Class] to Model expectations
        if len(features) == 31:
            time_val = features[0]
            v_features = features[1:29]
            amount_val = features[29]
            
            hour_val = int(time_val / 3600) % 24
            
            if scaler is not None:
                try:
                    scaled = scaler.transform(np.array([[time_val, amount_val]]))
                    scaled_time = scaled[0][0]
                    scaled_amount = scaled[0][1]
                except:
                    scaled_time = time_val
                    scaled_amount = amount_val
            else:
                scaled_time = time_val
                scaled_amount = amount_val
                
            model_input = [scaled_time, scaled_amount] + v_features + [hour_val]
            input_array = np.array(model_input, dtype=float).reshape(1, -1)
        else:
            input_array = np.array(features, dtype=float).reshape(1, -1)
        
        model = models[model_name]
        start = time.time()
        
        if model_name == "ann":
            prob = float(model.predict(input_array, verbose=0)[0][0])
            pred = 1 if prob >= 0.5 else 0
        else:
            prediction = model.predict(input_array)
            pred = int(prediction[0])
            prob = None
            if hasattr(model, "predict_proba"):
                prob = round(float(model.predict_proba(input_array)[0][1]), 6)
        
        elapsed = round((time.time() - start) * 1000, 2)
        
        return jsonify({
            "model": model_name,
            "prediction": pred,
            "probability": prob if prob is not None else (prob if model_name != "ann" else round(prob, 6)),
            "status": "Normal Transaction" if pred == 0 else "Fraudulent Transaction",
            "inference_time_ms": elapsed
        }), 200
    
    except ValueError as ve:
        return jsonify({"error": f"Value Error: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# -------------------------------------------------------------
# 6. Sample Data Endpoint
# -------------------------------------------------------------
@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return 3 sample transaction data for testing."""
    samples = [
        {
            "name": "🟢 Normal Transaction #1",
            "description": "Low-amount legitimate purchase — typical consumer behavior",
            "label": "Normal",
            "data": [-0.528, -0.181, -1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239, 0.098,
                     0.363, 0.090, -0.551, -0.617, -0.991, -0.311, 1.468, -0.470, 0.207, 0.025,
                     0.251, -0.018, 0.277, -0.110, 0.066, 0.128, -0.189, 0.133, -0.021, 0.014, 14.0]
        },
        {
            "name": "🔴 Fraudulent Transaction #1",
            "description": "Suspicious high-amount transaction with anomalous PCA features",
            "label": "Fraud",
            "data": [1.213, 2.451, 1.191, 0.266, 0.166, 0.448, 0.059, -0.082, -0.078, 0.085,
                     -0.255, -0.166, 1.612, 1.065, 0.489, -0.143, 0.635, 0.463, -0.114, -0.183,
                     -0.145, -0.069, -0.225, -0.638, 0.101, -0.339, 0.167, 0.125, -0.008, 0.014, 3.0]
        },
        {
            "name": "🟡 Borderline Case",
            "description": "Ambiguous transaction — tests model disagreement scenarios",
            "label": "Unknown",
            "data": [0.105, 0.359, -0.425, 0.960, 1.141, -0.168, 0.420, -0.029, 0.476, 0.260,
                     -0.568, -0.371, 1.341, 0.359, -0.358, 0.459, -0.210, 0.870, 0.140, 0.299,
                     -0.145, 0.040, 0.110, 0.260, -0.070, -0.060, -0.590, 0.340, 0.080, 0.020, 23.0]
        }
    ]
    return jsonify({"samples": samples})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
