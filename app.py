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
# 0. Chemins des fichiers (tout à la racine du repo)
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
# 1. Fonctions utilitaires
# ===============================
@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

@st.cache_data
def load_data():
    """
    Charge le CSV compressé en gzip.
    On suppose que la première colonne est l'index (date).
    """
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
    """
    Charge un modèle Keras sans le recompiler
    pour éviter les problèmes de compatibilité de loss / metrics.
    """
    return load_model(path, compile=False, safe_mode=False)

def make_sequences(X_2d: np.ndarray, y_1d: np.ndarray, window: int):
    """
    Crée des séquences temporelles :
    X_seq -> (n_samples - window, window, n_features)
    y_seq -> (n_samples - window,)
    """
    X_seqs, y_seqs = [], []
    for i in range(len(X_2d) - window):
        X_seqs.append(X_2d[i:i + window])
        y_seqs.append(y_1d[i + window])
    return np.array(X_seqs), np.array(y_seqs)

def build_train_test_sequences(df, features, target, split_date, scaler, window):
    """
    - Trie par date
    - Split temporel train / test selon split_date
    - Applique le scaler
    - Construit les séquences (fenêtrage) pour les modèles DL
    """
    df = df.sort_index()

    train = df.loc[df.index < split_date].copy()
    test  = df.loc[df.index >= split_date].copy()

    X_train = train[features].values
    y_train = train[target].values

    X_test = test[features].values
    y_test = test[target].values

    # scaling avec le scaler déjà entraîné
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # fenêtrage
    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train, window)
    X_test_seq,  y_test_seq  = make_sequences(X_test_scaled,  y_test,  window)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, train, test

def categorize_rarity(value: float) -> str:
    """
    Mappe une valeur de Rarity → catégorie texte compréhensible.
    D'après la description :
      0 ~ 100/an, 1 ~ 1/mois, 2 ~ 1/an, 3 ~ 1/décennie, etc.
    On arrondit à l'entier le plus proche pour l'explication.
    """
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
# 2. UI Streamlit
# ===============================
st.set_page_config(page_title="NEO Rarity – Deep Learning", layout="wide")

st.title("🛰️ Prédiction de la *Rarity* des NEO avec Deep Learning")

st.markdown(
    """
    Cette interface utilise :
    - la **dernière data daily avec lags** (`neo_daily_lags (1).csv.gz`),
    - un fichier **JSON de configuration des features** (`features_config.json`),
    - des modèles **Deep Learning** déjà entraînés (`.h5`),
    - un `scaler.pkl` (MinMaxScaler) pour reproduire le pré-traitement.
    """
)

# ===============================
# 3. Vérification de la présence des fichiers
# ===============================
if not DATA_PATH.exists():
    st.error(f"❌ Fichier data introuvable : `{DATA_PATH}`")
    st.stop()

if not CONFIG_PATH.exists():
    st.error(f"❌ Fichier config introuvable : `{CONFIG_PATH}`")
    st.stop()

if not SCALER_PATH.exists():
    st.error(f"❌ Fichier scaler introuvable : `{SCALER_PATH}`")
    st.stop()

# ===============================
# 4. Chargement des objets (config, data, scaler)
# ===============================
cfg = load_config()
df = load_data()
scaler = load_scaler()

features_from_config = cfg["features"]
target = cfg["target"]
window = cfg.get("seq_length", 30)
split_date = cfg.get("split_date", "2025-01-01")

# ===============================
# 5. Debug columns / vérification des features
# ===============================
st.subheader("🔍 Debug colonnes du dataset")

st.write("**Nombre de colonnes dans df :**", len(df.columns))
st.write("**Quelques colonnes :**", list(df.columns)[:40])

missing = [c for c in features_from_config + [target] if c not in df.columns]

if missing:
    st.error(f"⛔ Colonnes manquantes dans le dataset (vérifie le CSV ou le JSON) : {missing}")
    st.stop()

# si tout est ok, on utilise les features du JSON
features = features_from_config

# ===============================
# 6. Sidebar config
# ===============================
st.sidebar.header("⚙️ Configuration")
st.sidebar.write(f"**Features utilisées :** {len(features)}")
st.sidebar.write(", ".join(features))
st.sidebar.write(f"**Target :** `{target}`")
st.sidebar.write(f"**Fenêtre temporelle :** {window} jours")
st.sidebar.write(f"**Split date :** {split_date}")

st.subheader("👀 Aperçu de la data daily avec lags")
st.dataframe(df.head())

# ===============================
# 7. Choix du modèle DL
# ===============================
st.subheader("🧠 Choisir un modèle Deep Learning")

available_models = {name: path for name, path in MODEL_PATHS.items() if path.exists()}

if not available_models:
    st.error("❌ Aucun fichier modèle .h5 trouvé à la racine du repo.")
    st.stop()

model_name = st.selectbox(
    "Modèle à utiliser :",
    options=list(available_models.keys()),
    index=0,
)

model_path = available_models[model_name]
st.info(f"📂 Modèle sélectionné : `{model_name}` → `{model_path}`")

model = load_dl_model(model_path)

# ===============================
# 8. Construction des séquences & prédictions (test set)
# ===============================
st.subheader("📊 Évaluation sur le jeu de test")

with st.spinner("Construction des séquences et prédiction en cours..."):
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, train_df, test_df = build_train_test_sequences(
        df, features, target, split_date, scaler, window
    )

    y_pred_test = model.predict(X_test_seq).flatten()

    mae = mean_absolute_error(y_test_seq, y_pred_test)
    mse = mean_squared_error(y_test_seq, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_seq, y_pred_test)

st.write("### 📌 Métriques Test")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("RMSE", f"{rmse:.4f}")
col4.metric("R²", f"{r2:.4f}")

# ===============================
# 9. Graphique Réel vs Prédit
# ===============================
st.write("### 📈 Réel vs Prédit (jeu de test fenêtré)")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test_seq, label="Rarity réelle")
ax.plot(y_pred_test, label="Rarity prédite")
ax.set_xlabel("Index séquentiel (fenêtrage)")
ax.set_ylabel("Rarity")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# ===============================
# 9 bis. Prédiction pour UNE seule combinaison d'entrées
# ===============================
st.write("### 🎛 Prédiction pour une combinaison personnalisée (1 seul cas)")

needed_inputs = ["Diameter_Max", "V relative(km/s)", "H(mag)"]
for col in needed_inputs:
    if col not in features:
        st.warning(
            f"La feature `{col}` n'est pas dans la liste des features utilisées. "
            "Vérifie ton features_config.json."
        )
        st.stop()

# Valeurs par défaut = dernière ligne de df (dernier jour)
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

if st.button("🔮 Prédire Rarity pour ces valeurs"):
    # On prend la dernière fenêtre temporelle sur toutes les features
    context = df[features].tail(window).copy()

    # On modifie la dernière ligne avec les nouvelles entrées
    # (les autres features de lags restent cohérentes avec l'historique réel)
    last_idx = context.index[-1]
    context.loc[last_idx, "Diameter_Max"] = input_diam
    context.loc[last_idx, "V relative(km/s)"] = input_vrel
    context.loc[last_idx, "H(mag)"] = input_Hmag

    # Scaling (le scaler attend du 2D)
    context_scaled = scaler.transform(context.values)  # (window, n_features)

    # Reshape pour le modèle DL : (1, window, n_features)
    X_single = context_scaled.reshape(1, window, len(features))

    # Prédiction
    y_single_pred = model.predict(X_single).flatten()[0]

    # Catégorisation
    cat = categorize_rarity(y_single_pred)

    st.success(
        f"✨ Rarity prédite pour ces entrées : **{y_single_pred:.4f}**\n\n"
        f"📎 Catégorie : **{cat}**"
    )

# ===============================
# 10. Export des prédictions (test set complet, SANS catégorie supplémentaire)
# ===============================
st.write("### 📥 Télécharger les prédictions (test set fenêtré)")

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
