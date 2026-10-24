"""
Automobile Fuel Efficiency (MPG) Predictor
------------------------------------------
End-to-end Streamlit app:
  1. Data extraction   -> bundled auto-mpg.csv (fallback: UCI URL / CSV upload)
  2. Data cleaning     -> type fixing, '?' -> NaN, duplicate removal
  3. Preprocessing     -> missing value imputation, feature engineering,
                          scaling (numeric), one-hot encoding (categorical)
  4. Modelling         -> compares several regressors with cross-validation,
                          keeps the best one
  5. Deployment        -> interactive prediction UI + model diagnostics

Run locally:  streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="wide")

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
SEABORN_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
)
LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto-mpg.csv")
MODEL_PATH = "mpg_model.joblib"
RANDOM_STATE = 42

TARGET = "mpg"
NUMERIC_BASE = ["cylinders", "displacement", "horsepower", "weight",
                "acceleration", "model_year"]
CATEGORICAL = ["origin"]
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}
ENGINEERED = ["power_to_weight", "disp_per_cyl", "vehicle_age"]


# --------------------------------------------------------------------------
# 1. DATA EXTRACTION
# --------------------------------------------------------------------------
def _from_uci() -> pd.DataFrame:
    cols = ["mpg", "cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year", "origin", "car_name"]
    df = pd.read_csv(UCI_URL, names=cols, na_values="?",
                     comment="\t", sep=" ", skipinitialspace=True)
    df["origin"] = df["origin"].map(ORIGIN_MAP)
    return df


def _from_seaborn() -> pd.DataFrame:
    df = pd.read_csv(SEABORN_URL)
    df = df.rename(columns={"name": "car_name"})
    df["model_year"] = df["model_year"].astype(int)
    df["origin"] = df["origin"].str.upper().replace({"EUROPE": "Europe",
                                                       "JAPAN": "Japan"})
    return df


def _from_local() -> pd.DataFrame:
    df = pd.read_csv(LOCAL_CSV, na_values=["?", "NA", ""])
    return standardise_columns(df)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names / origin coding of any Auto-MPG style CSV."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"name": "car_name", "year": "model_year",
                            "model": "model_year"})
    if "origin" in df.columns and pd.api.types.is_numeric_dtype(df["origin"]):
        df["origin"] = df["origin"].map(ORIGIN_MAP)
    elif "origin" in df.columns:
        df["origin"] = (df["origin"].astype(str).str.strip().str.title()
                        .replace({"Usa": "USA"}))
    missing = [c for c in [TARGET] + NUMERIC_BASE + CATEGORICAL
               if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


@st.cache_data(show_spinner="Loading dataset…")
def load_data(uploaded_bytes: bytes | None = None) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_description)."""
    if uploaded_bytes is not None:
        import io
        df = pd.read_csv(io.BytesIO(uploaded_bytes), na_values=["?", "NA", ""])
        return standardise_columns(df), "uploaded CSV"

    # bundled dataset first (fast, works offline), online sources as fallback
    for loader, label in [(_from_local, f"bundled dataset ({LOCAL_CSV})"),
                          (_from_uci, "UCI repository"),
                          (_from_seaborn, "seaborn-data mirror")]:
        try:
            if label.startswith("bundled") and not os.path.exists(LOCAL_CSV):
                continue
            return loader(), label
        except Exception:
            continue
    raise RuntimeError(
        "Could not load the dataset from any source. Place 'auto-mpg.csv' "
        "next to app.py or upload a CSV from the sidebar."
    )


# --------------------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # make sure numeric columns really are numeric ('?' etc. -> NaN)
    for col in [TARGET] + NUMERIC_BASE:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # model year such as 70 -> 1970
    df.loc[df["model_year"] < 100, "model_year"] += 1900

    # target cannot be imputed -> drop rows without it
    df = df.dropna(subset=[TARGET])

    # remove duplicates and physically impossible values
    df = df.drop_duplicates()
    df = df[(df[TARGET] > 0) & (df["weight"].fillna(1) > 0)
            & (df["cylinders"].fillna(4) > 0)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. PREPROCESSING PIPELINE
# --------------------------------------------------------------------------
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering (NaN-safe; imputation happens right afterwards)."""
    X = X.copy()
    X["power_to_weight"] = X["horsepower"] / X["weight"]
    X["disp_per_cyl"] = X["displacement"] / X["cylinders"]
    X["vehicle_age"] = 2000 - X["model_year"]  # relative age; scale-friendly
    return X


def build_pipeline(model) -> Pipeline:
    numeric_cols = NUMERIC_BASE + ENGINEERED

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # missing values
        ("scale", StandardScaler()),                    # scaling
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),  # encoding
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, CATEGORICAL),
    ])

    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", preprocessor),
        ("model", model),
    ])


# --------------------------------------------------------------------------
# 4. MODEL TRAINING & SELECTION
# --------------------------------------------------------------------------
def candidate_models() -> dict:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE),
    }


@st.cache_resource(show_spinner="Training models (first run only)…")
def train(df: pd.DataFrame):
    X = df[NUMERIC_BASE + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    # cross-validate each candidate on the training split
    cv_rows = []
    for name, mdl in candidate_models().items():
        scores = cross_val_score(build_pipeline(mdl), X_train, y_train,
                                 cv=5, scoring="r2")
        cv_rows.append({"Model": name, "CV R² (mean)": scores.mean(),
                        "CV R² (std)": scores.std()})
    cv_df = (pd.DataFrame(cv_rows)
             .sort_values("CV R² (mean)", ascending=False)
             .reset_index(drop=True))

    best_name = cv_df.loc[0, "Model"]
    best_pipe = build_pipeline(candidate_models()[best_name])
    best_pipe.fit(X_train, y_train)

    preds = best_pipe.predict(X_test)
    metrics = {
        "R²": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    test_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})

    # final model is refit on ALL data for the best possible predictions
    final_pipe = build_pipeline(candidate_models()[best_name]).fit(X, y)
    joblib.dump(final_pipe, MODEL_PATH)

    return final_pipe, best_name, cv_df, metrics, test_df


# --------------------------------------------------------------------------
# 5. UI
# --------------------------------------------------------------------------
def main():
    st.title("🚗 Automobile Fuel Efficiency (MPG) Predictor")
    st.caption("Predict a car's miles-per-gallon from its specifications.")

    # ---- sidebar: optional data upload -----------------------------------
    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Use your own Auto-MPG CSV (optional)",
                              type="csv")
        st.caption("Required columns: mpg, cylinders, displacement, "
                   "horsepower, weight, acceleration, model_year, origin")

    try:
        raw, source = load_data(up.getvalue() if up else None)
        df = clean_data(raw)
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    model, best_name, cv_df, metrics, test_df = train(df)

    tab_pred, tab_data, tab_model = st.tabs(
        ["🔮 Predict", "📊 Data", "🧪 Model performance"])

    # ---- Predict ---------------------------------------------------------
    with tab_pred:
        st.subheader("Enter vehicle specifications")
        c1, c2, c3 = st.columns(3)
        lo, hi = df.min(numeric_only=True), df.max(numeric_only=True)

        with c1:
            cylinders = st.selectbox(
                "Cylinders", sorted(df["cylinders"].dropna().astype(int).unique()),
                index=1)
            displacement = st.number_input(
                "Displacement (cu. in.)", float(lo["displacement"]),
                float(hi["displacement"]), float(df["displacement"].median()))
        with c2:
            horsepower = st.number_input(
                "Horsepower", float(lo["horsepower"]), float(hi["horsepower"]),
                float(df["horsepower"].median()))
            weight = st.number_input(
                "Weight (lbs)", float(lo["weight"]), float(hi["weight"]),
                float(df["weight"].median()))
        with c3:
            acceleration = st.number_input(
                "Acceleration (0–60 mph, sec)", float(lo["acceleration"]),
                float(hi["acceleration"]), float(df["acceleration"].median()))
            model_year = st.slider(
                "Model year", int(lo["model_year"]), int(hi["model_year"]),
                int(df["model_year"].median()))
        origin = st.selectbox("Origin", sorted(df["origin"].dropna().unique()))

        if st.button("Predict MPG", type="primary"):
            row = pd.DataFrame([{
                "cylinders": cylinders, "displacement": displacement,
                "horsepower": horsepower, "weight": weight,
                "acceleration": acceleration, "model_year": model_year,
                "origin": origin,
            }])
            mpg = float(model.predict(row)[0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted fuel efficiency", f"{mpg:.1f} MPG")
            m2.metric("In km per litre", f"{mpg * 0.4251:.1f} km/L")
            m3.metric("In L / 100 km", f"{235.215 / mpg:.1f}")

            avg = df[TARGET].mean()
            st.progress(min(max(mpg / df[TARGET].max(), 0.0), 1.0))
            st.caption(f"Dataset average: {avg:.1f} MPG — this car is "
                       f"{'above' if mpg >= avg else 'below'} average.")

    # ---- Data ------------------------------------------------------------
    with tab_data:
        st.write(f"**Source:** {source} · **Rows after cleaning:** {len(df)}")
        a, b = st.columns(2)
        with a:
            st.write("Missing values in raw data")
            st.dataframe(raw.isna().sum().rename("missing").to_frame())
        with b:
            st.write("Summary statistics")
            st.dataframe(df.describe().T)
        st.write("Sample rows")
        st.dataframe(df.head(25), use_container_width=True)
        st.write("MPG vs weight")
        st.scatter_chart(df, x="weight", y=TARGET, color="origin")

    # ---- Model -----------------------------------------------------------
    with tab_model:
        st.write(f"**Selected model:** {best_name}")
        a, b, c = st.columns(3)
        a.metric("Test R²", f"{metrics['R²']:.3f}")
        b.metric("Test MAE", f"{metrics['MAE']:.2f} MPG")
        c.metric("Test RMSE", f"{metrics['RMSE']:.2f} MPG")
        st.write("Cross-validation comparison (5-fold, training split)")
        st.dataframe(cv_df, use_container_width=True)
        st.write("Actual vs predicted (held-out test set)")
        st.scatter_chart(test_df, x="Actual", y="Predicted")


if __name__ == "__main__":
    main()
