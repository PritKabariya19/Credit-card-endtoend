import streamlit as st
import joblib
import numpy as np
import os
import time
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 1. Model Loading with Caching
# =====================================================================
MODELS_DIR = "models"

@st.cache_resource
def load_all_models():
    """Load all .pkl models and the keras ANN model from the models directory."""
    models = {}
    scaler = None
    
    # Load scaler
    scaler_path = os.path.join(MODELS_DIR, "robust_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
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
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")
    
    # Load Keras ANN model
    ann_path = os.path.join(MODELS_DIR, "ann_model.keras")
    if os.path.exists(ann_path):
        try:
            import tensorflow as tf
            models["ann"] = tf.keras.models.load_model(ann_path)
        except ImportError:
            st.warning("TensorFlow not installed. ANN model will not be available.")
        except Exception as e:
            st.warning(f"Failed to load ANN model: {e}")
    
    # Load tuned XGBoost if available
    tuned_path = os.path.join(MODELS_DIR, "best_xgb_tuned.pkl")
    if os.path.exists(tuned_path):
        try:
            models["xgboost_tuned"] = joblib.load(tuned_path)
        except Exception as e:
            st.warning(f"Failed to load tuned XGBoost: {e}")
    
    return models, scaler

# Load models
models, scaler = load_all_models()

# Model information
MODEL_INFO = {
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

# =====================================================================
# 2. Prediction Functions
# =====================================================================

def preprocess_features(features):
    """Convert raw features to model input format."""
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
        return np.array(model_input, dtype=float).reshape(1, -1)
    else:
        return np.array(features, dtype=float).reshape(1, -1)

def predict_single(model_name, features):
    """Predict using a single model."""
    if model_name not in models:
        return None, None, None, 0
    
    try:
        input_array = preprocess_features(features)
        model = models[model_name]
        start = time.time()
        
        if model_name == "ann":
            prob = float(model.predict(input_array, verbose=0)[0][0])
            prediction = 1 if prob >= 0.5 else 0
        else:
            prediction = int(model.predict(input_array)[0])
            prob = None
            if hasattr(model, "predict_proba"):
                prob = round(float(model.predict_proba(input_array)[0][1]), 6)
        
        elapsed = round((time.time() - start) * 1000, 2)
        status = "🔴 Fraud" if prediction == 1 else "🟢 Safe"
        
        return prediction, prob, status, elapsed
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}", 0

def predict_all(features):
    """Predict using all available models."""
    results = {}
    
    for name in models.keys():
        prediction, prob, status, elapsed = predict_single(name, features)
        results[name] = {
            "prediction": prediction,
            "probability": prob,
            "status": status,
            "inference_time_ms": elapsed
        }
    
    # Consensus voting
    predictions = [r["prediction"] for r in results.values() if r["prediction"] is not None]
    fraud_votes = sum(predictions)
    total_votes = len(predictions)
    
    consensus = {
        "fraud_votes": fraud_votes,
        "normal_votes": total_votes - fraud_votes,
        "total_models": total_votes,
        "verdict": "🔴 FRAUD" if fraud_votes > total_votes / 2 else "🟢 SAFE",
        "confidence": round(max(fraud_votes, total_votes - fraud_votes) / total_votes * 100, 1) if total_votes > 0 else 0
    }
    
    return results, consensus

# =====================================================================
# 3. Sample Data
# =====================================================================

SAMPLE_DATA = {
    "normal_1": {
        "name": "🟢 Normal Transaction #1",
        "description": "Low-amount legitimate purchase — typical consumer behavior",
        "label": "Normal",
        "data": [-0.528, -0.181, -1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239, 0.098,
                 0.363, 0.090, -0.551, -0.617, -0.991, -0.311, 1.468, -0.470, 0.207, 0.025,
                 0.251, -0.018, 0.277, -0.110, 0.066, 0.128, -0.189, 0.133, -0.021, 0.014, 14.0]
    },
    "fraud_1": {
        "name": "🔴 Fraudulent Transaction #1",
        "description": "Suspicious high-amount transaction with anomalous PCA features",
        "label": "Fraud",
        "data": [1.213, 2.451, 1.191, 0.266, 0.166, 0.448, 0.059, -0.082, -0.078, 0.085,
                 -0.255, -0.166, 1.612, 1.065, 0.489, -0.143, 0.635, 0.463, -0.114, -0.183,
                 -0.145, -0.069, -0.225, -0.638, 0.101, -0.339, 0.167, 0.125, -0.008, 0.014, 3.0]
    },
    "borderline": {
        "name": "🟡 Borderline Case",
        "description": "Ambiguous transaction — tests model disagreement scenarios",
        "label": "Unknown",
        "data": [0.105, 0.359, -0.425, 0.960, 1.141, -0.168, 0.420, -0.029, 0.476, 0.260,
                 -0.568, -0.371, 1.341, 0.359, -0.358, 0.459, -0.210, 0.870, 0.140, 0.299,
                 -0.145, 0.040, 0.110, 0.260, -0.070, -0.060, -0.590, 0.340, 0.080, 0.020, 23.0]
    }
}

# =====================================================================
# 4. UI Components
# =====================================================================

def display_header():
    """Display main header."""
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("### Advanced Multi-Model Ensemble for Real-Time Fraud Detection")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Loaded Models", len(models))
    with col2:
        st.metric("Total Features", "31")
    with col3:
        st.metric("Prediction Method", "Ensemble Voting")

def display_home():
    """Display home page."""
    st.header("📊 System Overview")
    
    st.markdown("""
    This system uses **ensemble learning** with multiple ML models to detect fraudulent credit card transactions.
    
    #### 🤖 Available Models:
    """)
    
    available_models = {k: v for k, v in MODEL_INFO.items() if k in models}
    
    cols = st.columns(min(3, len(available_models)))
    for idx, (key, info) in enumerate(available_models.items()):
        with cols[idx % len(cols)]:
            st.markdown(f"""
            **{info['icon']} {info['name']}**  
            {info['description']}
            """)
    
    st.markdown("---")
    
    st.markdown("""
    #### 📝 How to Use:
    
    1. **Single Model**: Test predictions with individual models
    2. **Compare All Models**: Get consensus voting across all models
    3. **Sample Data**: Test with pre-configured sample transactions
    
    #### 📌 Input Format:
    Provide 31 features:
    - Time (seconds since start of day)
    - V1-V28 (PCA-transformed features)
    - Amount (transaction amount)
    """)

def display_single_model():
    """Display single model prediction interface."""
    st.header("🔍 Single Model Prediction")
    
    # Model selection
    available_models = list(models.keys())
    selected_model = st.selectbox("Select Model", available_models)
    
    model_info = MODEL_INFO.get(selected_model, {})
    st.info(f"{model_info.get('icon', '🤖')} {model_info.get('description', '')}")
    
    # Input method selection
    input_method = st.radio("Input Method", ["Manual Input", "Sample Data"])
    
    if input_method == "Sample Data":
        sample_key = st.selectbox("Select Sample", SAMPLE_DATA.keys(), 
                                  format_func=lambda k: SAMPLE_DATA[k]["name"])
        features = SAMPLE_DATA[sample_key]["data"]
        st.info(SAMPLE_DATA[sample_key]["description"])
    else:
        # Manual input
        st.subheader("Enter Features")
        col1, col2 = st.columns(2)
        
        time_val = col1.number_input("Time (seconds)", value=0.0, step=100.0)
        amount_val = col2.number_input("Amount ($)", value=0.0, step=1.0, format="%.2f")
        
        st.markdown("**PCA Features (V1-V28)**")
        v_features = []
        for i in range(0, 28, 7):
            cols = st.columns(7)
            for j in range(7):
                if i + j < 28:
                    v_features.append(cols[j].number_input(f"V{i+j+1}", value=0.0, step=0.1, key=f"v_{i+j}"))
        
        features = [time_val] + v_features + [amount_val]
    
    # Make prediction
    if st.button("🚀 Predict", use_container_width=True, type="primary"):
        prediction, prob, status, elapsed = predict_single(selected_model, features)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", status)
        with col2:
            if prob is not None:
                st.metric("Probability", f"{prob:.4f}")
        with col3:
            st.metric("Inference Time", f"{elapsed}ms")

def display_all_models():
    """Display all models comparison."""
    st.header("⚖️ Compare All Models");
    
    # Input method selection
    input_method = st.radio("Input Method", ["Manual Input", "Sample Data"], key="all_models_input")
    
    if input_method == "Sample Data":
        sample_key = st.selectbox("Select Sample", SAMPLE_DATA.keys(), 
                                  format_func=lambda k: SAMPLE_DATA[k]["name"],
                                  key="all_models_sample")
        features = SAMPLE_DATA[sample_key]["data"]
        st.info(SAMPLE_DATA[sample_key]["description"])
    else:
        # Manual input
        st.subheader("Enter Features")
        col1, col2 = st.columns(2)
        
        time_val = col1.number_input("Time (seconds)", value=0.0, step=100.0, key="all_time")
        amount_val = col2.number_input("Amount ($)", value=0.0, step=1.0, format="%.2f", key="all_amount")
        
        st.markdown("**PCA Features (V1-V28)**")
        v_features = []
        for i in range(0, 28, 7):
            cols = st.columns(7)
            for j in range(7):
                if i + j < 28:
                    v_features.append(cols[j].number_input(f"V{i+j+1}", value=0.0, step=0.1, key=f"all_v_{i+j}"))
        
        features = [time_val] + v_features + [amount_val]
    
    # Make predictions
    if st.button("🚀 Predict with All Models", use_container_width=True, type="primary"):
        results, consensus = predict_all(features)
        
        st.markdown("---")
        st.subheader("🎯 Consensus Verdict")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Verdict", consensus["verdict"])
        with col2:
            st.metric("Confidence", f"{consensus['confidence']}%")
        with col3:
            st.metric("Fraud Votes", consensus["fraud_votes"])
        with col4:
            st.metric("Normal Votes", consensus["normal_votes"])
        
        st.markdown("---")
        st.subheader("📊 Individual Model Results")
        
        for model_name, result in results.items():
            model_info = MODEL_INFO.get(model_name, {})
            with st.expander(f"{model_info.get('icon', '🤖')} {model_info.get('name', model_name)}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", result["status"])
                with col2:
                    if result["probability"] is not None:
                        st.metric("Probability", f"{result['probability']:.4f}")
                with col3:
                    st.metric("Time", f"{result['inference_time_ms']}ms")

def display_sample_data():
    """Display sample data interface."""
    st.header("📋 Sample Transactions")
    
    st.markdown("Pre-configured test cases to evaluate model performance:")
    
    for key, sample in SAMPLE_DATA.items():
        with st.expander(f"{sample['name']}"):
            st.markdown(f"**Description:** {sample['description']}")
            st.markdown(f"**Label:** {sample['label']}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(f"Features: {len(sample['data'])} values", language="python")
            with col2:
                if st.button("Test Now", key=f"test_{key}"):
                    results, consensus = predict_all(sample["data"])
                    st.session_state.last_results = results
                    st.session_state.last_consensus = consensus
                    st.session_state.show_results = True
    
    if st.session_state.get("show_results"):
        st.markdown("---")
        st.subheader("Results")
        results = st.session_state.get("last_results", {})
        consensus = st.session_state.get("last_consensus", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Verdict", consensus.get("verdict", "N/A"))
        with col2:
            st.metric("Confidence", f"{consensus.get('confidence', 0)}%")
        with col3:
            st.metric("Models Agree", f"{max(consensus.get('fraud_votes', 0), consensus.get('normal_votes', 0))}/{consensus.get('total_models', 0)}")

# =====================================================================
# 5. Main App
# =====================================================================

def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Single Model", "Compare All Models", "Sample Data"])
    st.session_state.page = page
    
    # Display header
    display_header()
    
    st.markdown("---")
    
    # Route to appropriate page
    if page == "Home":
        display_home()
    elif page == "Single Model":
        display_single_model()
    elif page == "Compare All Models":
        display_all_models()
    elif page == "Sample Data":
        display_sample_data()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <small>Credit Card Fraud Detection System | Powered by Streamlit | Multi-Model Ensemble</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
