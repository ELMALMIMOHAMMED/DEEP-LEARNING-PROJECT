import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# ===============================
# 0. CONFIG STREAMLIT + THEME CSS
# ===============================
st.set_page_config(
    page_title="NEO Rarity – Deep Learning",
    page_icon="🛰️",
    layout="wide"
)

# --- Thème "aerospace" simple via CSS ---
SPACE_CSS = """
<style>
/* Background général */
.stApp {
    background: radial-gradient(circle at top, #0b1020 0, #050814 40%, #02030a 100%);
    color: #f5f7ff;
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Cartes style "cockpit" */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Titres */
h1, h2, h3, h4 {
    color: #f5f7ff;
}

/* Panel / box custom */
.aero-card {
    background: rgba(7, 16, 40, 0.95);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    border: 1px solid rgba(109, 173, 255, 0.3);
    box-shadow: 0 0 25px rgba(0, 140, 255, 0.12);
}

/* Petits badges */
.badge {
    display: inline-block;
    padding: 0.1rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #1f6feb, #a371f7);
    color: #fff;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(7, 16, 40, 0.95);
    border-radius: 10px;
    padding: 0.8rem;
    border: 1px solid rgba(109, 173, 255, 0.25);
}

/* Inputs */
.stNumberInput > div > div > input {
    background-color: #050814;
    color: #f5f7ff;
    border-radius: 8px;
    border: 1px solid rgba(109, 173, 255, 0.4);
}

/* Download button */
.stDownloadButton button, .stButton button {
    border-radius: 999px;
    padding: 0.4rem 1.4rem;
    font-weight: 600;
    border: none;
}
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)

# ===============================
# 1. CONSTANTES FICHIERS
# ===============================
DATA_PATH = Path("neo_daily_lags (1).csv.gz")
CONFIG_PATH = Path("features_config.json")
SCALER_PATH = Path("scaler.pkl")

MODEL_PATHS = {
    "MLP": Path("model_MLP_neo.h5"),
    "GRU": Path("model_GRU_neo.h5"),
    "LSTM": Path("model_LSTM_neo.h5"),
    "Best (model_neo)": Path("model_neo.h5"),  # optionnel
}

# ===============================
# 2. FONCTIONS UTILITAIRES
# ===============================
@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
        compression="gzip",
    )
    return df

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_dl_model(path: Path):
    return load_model(path, compile=False, safe_mode=False)

def make_sequences(X_2d: np.ndarray, y_1d: np.ndarray, window: int):
    X_seqs, y_seqs = [], []
    for i in range(len(X_2d) - window):
        X_seqs.append(X_2d[i:i + window])
        y_seqs.append(y_1d[i + window])
    return np.array(X_seqs), np.array(y_seqs)

def build_train_test_sequences(df, features, target, split_date, scaler, window):
    df = df.sort_index()

    train = df.loc[df.index < split_date].copy()
    test  = df.loc[df.index >= split_date].copy()

    X_train = train[features].values
    y_train = train[target].values

    X_test = test[features].values
    y_test = test[target].values

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train, window)
    X_test_seq,  y_test_seq  = make_sequences(X_test_scaled,  y_test,  window)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, train, test

def categorize_rarity(value: float) -> str:
    if value is None:
        return "n/a (pas d'estimation)"
    try:
        v = float(value)
    except Exception:
        return "n/a (valeur non numérique)"

    if np.isnan(v):
        return "n/a (pas d'estimation)"

    r = int(round(v))

    if r <= 0:
        return "0 : très fréquent (~100 fois par an, tous les quelques jours)"
    elif r == 1:
        return "1 : fréquent (~1 fois par mois)"
    elif r == 2:
        return "2 : modéré (~1 fois par an)"
    elif r == 3:
        return "3 : rare (~1 fois par décennie)"
    elif r == 4:
        return "4 : très rare (~1 fois par siècle, extrapolé)"
    else:  # r >= 5
        return f"{r} : extrêmement rare (bien moins fréquent qu'une fois par siècle, extrapolé)"

# ===============================
# 3. HEADER
# ===============================
st.markdown(
    """
    <div class="aero-card">
      <span class="badge">NEO • Deep Learning</span>
      <h1>🛰️ NEO Rarity Prediction Dashboard</h1>
      <p>
        Visualisation et prédiction de la <strong>rareté des Near-Earth Objects</strong> 
        à partir d'un modèle de Deep Learning (MLP / GRU / LSTM) et de features temporelles (lags, rolls).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===============================
# 4. CHECK FICHIERS
# ===============================
missing_files = []
if not DATA_PATH.exists():
    missing_files.append(str(DATA_PATH))
if not CONFIG_PATH.exists():
    missing_files.append(str(CONFIG_PATH))
if not SCALER_PATH.exists():
    missing_files.append(str(SCALER_PATH))

if missing_files:
    st.error("❌ Fichiers manquants : " + ", ".join(missing_files))
    st.stop()

# ===============================
# 5. CHARGEMENT CONFIG + DATA + SCALER
# ===============================
cfg = load_config()
df = load_data()
scaler = load_scaler()

features_from_config = cfg["features"]
target = cfg["target"]
window = cfg.get("seq_length", 30)
split_date = cfg.get("split_date", "2025-01-01")

# ===============================
# 6. SIDEBAR : PARAMS & MODELE
# ===============================
st.sidebar.title("🧭 Mission Control")

debug_cols = st.sidebar.checkbox("Afficher le debug des colonnes", value=False)

available_models = {name: path for name, path in MODEL_PATHS.items() if path.exists()}
if not available_models:
    st.sidebar.error("Aucun modèle .h5 trouvé.")
    st.stop()

model_name = st.sidebar.selectbox(
    "Modèle Deep Learning",
    options=list(available_models.keys()),
    index=0,
)

model_path = available_models[model_name]
model = load_dl_model(model_path)

st.sidebar.markdown("---")
st.sidebar.markdown("**Split date**")
st.sidebar.code(str(split_date))
st.sidebar.markdown(f"**Fenêtre temporelle** : `{window}` jours")

st.sidebar.markdown("---")
st.sidebar.markdown("**Features utilisées :**")
st.sidebar.write(", ".join(features_from_config))

# ===============================
# 7. TABS PRINCIPAUX
# ===============================
tab_data, tab_eval, tab_single = st.tabs([
    "📊 Dataset & Config",
    "🧠 Model Evaluation",
    "🚀 Scenario Simulation",
])

# ====== TAB 1 : DATASET & CONFIG ======
with tab_data:
    st.markdown("### 📊 Dataset & Configuration")

    if debug_cols:
        st.markdown("#### 🔍 Colonnes du dataset (debug)")
        st.write("Nombre de colonnes :", len(df.columns))
        st.write("Premières colonnes :", list(df.columns)[:40])

    # Vérifier les features
    missing = [c for c in features_from_config + [target] if c not in df.columns]
    if missing:
        st.error(
            "⛔ Colonnes manquantes dans le dataset (corrige ton CSV ou features_config.json) : "
            + ", ".join(missing)
        )
        st.stop()

    features = features_from_config

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Aperçu du dataset (head)")
        st.dataframe(df.head())

    with col_right:
        st.markdown("#### Infos rapides")
        st.write(f"- Nombre de lignes : `{len(df)}`")
        st.write(f"- Nombre de features : `{len(features)}`")
        st.write(f"- Target : `{target}`")
        st.write(f"- Index temporel min : `{df.index.min()}`")
        st.write(f"- Index temporel max : `{df.index.max()}`")

# ====== TAB 2 : MODEL EVALUATION ======
with tab_eval:
    st.markdown("### 🧠 Évaluation du modèle sur le jeu de test")

    with st.spinner("Construction des séquences et prédiction en cours..."):
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, train_df, test_df = build_train_test_sequences(
            df, features, target, split_date, scaler, window
        )

        y_pred_test = model.predict(X_test_seq).flatten()

        mae = mean_absolute_error(y_test_seq, y_pred_test)
        mse = mean_squared_error(y_test_seq, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_seq, y_pred_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{mae:.4f}")
    m2.metric("MSE", f"{mse:.4f}")
    m3.metric("RMSE", f"{rmse:.4f}")
    m4.metric("R²", f"{r2:.4f}")

    st.markdown("#### 📈 Rarity réelle vs prédite (test set fenêtré)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test_seq, label="Rarity réelle")
    ax.plot(y_pred_test, label="Rarity prédite")
    ax.set_xlabel("Index séquentiel (fenêtrage)")
    ax.set_ylabel("Rarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.markdown("#### 📥 Télécharger les prédictions du test set")
    results_df = pd.DataFrame({
        "Rarity_true": y_test_seq,
        "Rarity_pred": y_pred_test,
    })
    st.dataframe(results_df.head())

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger les prédictions (CSV)",
        data=csv_bytes,
        file_name=f"neo_rarity_predictions_{model_name}.csv",
        mime="text/csv",
    )

# ====== TAB 3 : SINGLE SCENARIO ======
with tab_single:
    st.markdown("### 🚀 Simulation d'un scénario (1 seule prédiction)")
    st.markdown(
        "Modifie le **diamètre maximum**, la **vitesse relative** et la **magnitude H** "
        "pour estimer la rareté d'un NEO, en gardant le contexte temporel (lags) du dernier segment."
    )

    needed_inputs = ["Diameter_Max", "V relative(km/s)", "H(mag)"]
    for col in needed_inputs:
        if col not in features:
            st.warning(
                f"La feature `{col}` n'est pas dans la liste des features utilisées. "
                "Vérifie ton features_config.json."
            )
            st.stop()

    last_row = df.iloc[-1]
    default_diam = float(last_row["Diameter_Max"])
    default_vrel = float(last_row["V relative(km/s)"])
    default_Hmag = float(last_row["H(mag)"])

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        input_diam = st.number_input(
            "Diameter_Max",
            value=default_diam,
            format="%.6f"
        )
    with col_b:
        input_vrel = st.number_input(
            "V relative(km/s)",
            value=default_vrel,
            format="%.6f"
        )
    with col_c:
        input_Hmag = st.number_input(
            "H(mag)",
            value=default_Hmag,
            format="%.3f"
        )

    if st.button("🔮 Prédire Rarity pour ce scénario"):
        context = df[features].tail(window).copy()

        last_idx = context.index[-1]
        context.loc[last_idx, "Diameter_Max"] = input_diam
        context.loc[last_idx, "V relative(km/s)"] = input_vrel
        context.loc[last_idx, "H(mag)"] = input_Hmag

        context_scaled = scaler.transform(context.values)
        X_single = context_scaled.reshape(1, window, len(features))

        y_single_pred = model.predict(X_single).flatten()[0]
        cat = categorize_rarity(y_single_pred)

        st.markdown(
            f"""
            <div class="aero-card">
              <h3>Résultat de la simulation</h3>
              <p><strong>Rarity prédite :</strong> {y_single_pred:.4f}</p>
              <p><strong>Interprétation :</strong> {cat}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
