
import argparse
import polars as pl
import numpy as np
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import DATA_PROCESSED, TIMEZONE

#Colonnes cible / baseline et ensuite toutes features exclues.
TARGET = "load"
BASELINE = "forecast_load"

EXCLUDE_ALWAYS = {
    "timestamp", "load", "forecast_load",
}

FORECAST_EXCLUDE = {
    "load_lag_1", "load_lag_4",
    "load_lag_96",
    "pv_total_lag_96",
    "load_rmean_1h", "load_rstd_1h",
    "load_rmean_3h", "load_rstd_3h",
    "load_rmean_24h", "load_rstd_24h",
    "temp_rmean_1h", "temp_rstd_1h",
    "temp_rmean_3h", "temp_rstd_3h",
    "temp_rmean_24h", "temp_rstd_24h",
    "temp_gradient_1h", "temp_gradient_3h",
    "meteo_temperature_2m", "meteo_pressure", "meteo_global_radiation",
    "meteo_gust_peak", "meteo_precipitation", "meteo_humidity",
    "meteo_sunshine_duration", "meteo_wind_direction", "meteo_wind_speed",
    "pv_total", "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote",
    "pv_normalized",
    "temp_x_hour_sin", "temp_x_hour_cos",
    "pred_glob_uncertainty_x_pv",
    "day_of_year",
}
#hyperparametre
SLIDING_WINDOW_MONTHS = 18
PV_HOURS = list(range(10, 17))#heures pv 
PV_WEIGHT_DEFAULT = 1.6
N_OPTUNA_TRIALS = 100
MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50

#vecteur de poids pour que lightgbm minimise la loss
def compute_sample_weights(df: pl.DataFrame,
                           pv_hours: list[int],
                           pv_weight: float) -> np.ndarray | None:
    if pv_weight <= 1.0:
        return None
    hours = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TIMEZONE).dt.hour().alias("_h")
    )["_h"].to_numpy()
    weights = np.ones(len(hours), dtype=np.float64)
    for h in pv_hours:
        weights[hours == h] = pv_weight
    return weights

#Selectionne les features autorisees pour le forecast J+1.
def get_forecast_features(df: pl.DataFrame) -> list[str]:
    exclude = EXCLUDE_ALWAYS | FORECAST_EXCLUDE
    numeric_types = {
        pl.Float64, pl.Float32,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    }
    features = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in numeric_types
    ]
    return sorted(features)

"""Decoupe le dataset en train / validation / test.

    Coupures :
      Train : tout avant le 30 sept 2024 22h UTC
      Val   : 30 sept 2024 22h -> 30 mars 2025 22h UTC (6 mois)
      Test  : apres le 30 mars 2025 22h UTC (~6 mois)
"""

def temporal_split(df: pl.DataFrame,
                   sliding_window_months: int | None = None):
    cut_train = datetime(2024, 9, 30, 22, 0, tzinfo=timezone.utc)
    cut_val = datetime(2025, 3, 30, 22, 0, tzinfo=timezone.utc)

    train = df.filter(pl.col("timestamp") < cut_train)
    val = df.filter(
        (pl.col("timestamp") >= cut_train)
        & (pl.col("timestamp") < cut_val)
    )
    test = df.filter(pl.col("timestamp") >= cut_val)
# Application de la fenetre glissante sur le train pour limiter les donnees anciennes et moins eprtinentes

    if sliding_window_months is not None:
        window_start = cut_train - relativedelta(months=sliding_window_months)
        window_start = window_start.replace(tzinfo=timezone.utc)
        n_full = train.shape[0]
        train = train.filter(pl.col("timestamp") >= window_start)
        print(f"  Fenetre glissante : {sliding_window_months} mois")
        print(f"  Train reduit : {n_full:,} -> {train.shape[0]:,} lignes")

    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:5s} : {part.shape[0]:>7,} lignes "
              f"({part['timestamp'].min()} -> {part['timestamp'].max()})")

    return train, val, test

#convertit un DataFrame Polars en arrays NumPy (X, y) X matrice y vecteur
def to_numpy(df: pl.DataFrame, feature_cols: list[str]):
    X = df.select(feature_cols).to_numpy()
    y = df[TARGET].to_numpy()
    return X, y

# régression linéaire si LightGBM ne fait pas mieux que Ridge,
      #quelque chose ne va pas
def train_ridge(X_train, y_train, X_val, y_val, w_train=None):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train, sample_weight=w_train)
    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    print(f"  Ridge Forecast v8 - Val MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return model, pred_val

"""construit des arbres de decision sequentiellement
    - Chaque arbre corrige les erreurs residuelles du precedent
    - Le resultat final = somme ponderee de tous les arbres
    Et optuna optimise les hyperparametres
    HUber loss (en test car il pénalise moins les grosses erreurs que MAE)
    """
def train_lightgbm_optuna(X_train, y_train, X_val, y_val,
                          feature_names, w_train=None,
                          n_trials=150, max_boost_rounds=2000):
    dtrain = lgb.Dataset(X_train, y_train, weight=w_train,
                         feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(X_val, y_val, feature_name=feature_names,
                       free_raw_data=False)
#A chaque trial, Optuna propose une combinaison de parametres
    def objective(trial):
        params = {
            "objective": "huber",
            "alpha": trial.suggest_float("huber_alpha", 0.5, 2.0),
            "metric": "mae",
            "verbosity": -1,
            "n_jobs": -1,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True),
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS,
                               verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params, dtrain, num_boost_round=max_boost_rounds,
            valid_sets=[dval], callbacks=callbacks,
        )
        y_pred = model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)
# Lancer l'optimisation Optuna

    print(f"  Optuna : {n_trials} trials (Huber loss)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Meilleure MAE validation : {study.best_value:.4f}")
    print(f"  Meilleurs parametres : {study.best_params}")
# Optuna nomme le parametre "huber_alpha" mais LightGBM attend
    # "alpha" directement. On renomme.
    best_params = study.best_params.copy() # Re-entrainer le modele final avec les meilleurs parametres.
    huber_alpha = best_params.pop("huber_alpha")
    best_params.update({
        "objective": "huber",
        "alpha": huber_alpha,
        "metric": "mae",
        "verbosity": -1,
        "n_jobs": -1,
        "feature_pre_filter": False,
    })

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS,
                           verbose=False),
        lgb.log_evaluation(period=0),
    ]

    best_model = lgb.train(
        best_params, dtrain, num_boost_round=max_boost_rounds,
        valid_sets=[dval], callbacks=callbacks,
    )

    y_pred_val = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"  LightGBM v8 - Val MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return best_model, y_pred_val, study

#Affiche la MAE par heure et par mois dans la console
def evaluate_detailed(df, y_true, y_pred, model_name):
    hours = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TIMEZONE).dt.hour().alias("_h")
    )["_h"].to_numpy()
    months = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TIMEZONE).dt.month().alias("_m")
    )["_m"].to_numpy()

    mae_global = mean_absolute_error(y_true, y_pred)
    rmse_global = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n  {model_name} - MAE: {mae_global:.4f}, RMSE: {rmse_global:.4f}")

    print("  MAE par heure :")
    month_names = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Avr", 5: "Mai",
                   6: "Jun", 7: "Jul", 8: "Aou", 9: "Sep", 10: "Oct",
                   11: "Nov", 12: "Dec"}

    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            mae_h = mean_absolute_error(y_true[mask], y_pred[mask])
            bar = "#" * int(mae_h * 50)
            print(f"    {h:2d}h : {mae_h:.4f} {bar}")

    print("  MAE par mois :")
    for m in sorted(set(months)):
        mask = months == m
        if mask.sum() > 0:
            mae_m = mean_absolute_error(y_true[mask], y_pred[mask])
            name = month_names.get(m, str(m))
            print(f"    {name:3s} : {mae_m:.4f}")

    return mae_global

#Compare notre modele avec la baseline OIKEN. dans la console
def compare_with_baseline(df, y_pred, model_name="Forecast v8"):
    y_true = df[TARGET].to_numpy()
    y_baseline = df[BASELINE].to_numpy()

    valid = ~np.isnan(y_baseline) & ~np.isnan(y_true)

    ts = df["timestamp"]
    corrupt = (
        (ts >= datetime(2025, 9, 13, tzinfo=timezone.utc))
        & (ts < datetime(2025, 9, 17, tzinfo=timezone.utc))
    ).to_numpy()
    valid &= ~corrupt

    y_true_c = y_true[valid]
    y_baseline_c = y_baseline[valid]
    y_pred_c = y_pred[valid]

    mae_bl = mean_absolute_error(y_true_c, y_baseline_c)
    mae_md = mean_absolute_error(y_true_c, y_pred_c)
    rmse_bl = np.sqrt(mean_squared_error(y_true_c, y_baseline_c))
    rmse_md = np.sqrt(mean_squared_error(y_true_c, y_pred_c))
    imp_mae = (mae_bl - mae_md) / mae_bl * 100
    imp_rmse = (rmse_bl - rmse_md) / rmse_bl * 100

    hours = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TIMEZONE).dt.hour().alias("_h")
    )["_h"].to_numpy()[valid]
    pv_mask = np.isin(hours, PV_HOURS)

    mae_bl_pv = mean_absolute_error(y_true_c[pv_mask], y_baseline_c[pv_mask])
    mae_md_pv = mean_absolute_error(y_true_c[pv_mask], y_pred_c[pv_mask])
    rmse_bl_pv = np.sqrt(mean_squared_error(y_true_c[pv_mask], y_baseline_c[pv_mask]))
    rmse_md_pv = np.sqrt(mean_squared_error(y_true_c[pv_mask], y_pred_c[pv_mask]))
    mae_bl_night = mean_absolute_error(y_true_c[~pv_mask], y_baseline_c[~pv_mask])
    mae_md_night = mean_absolute_error(y_true_c[~pv_mask], y_pred_c[~pv_mask])
    rmse_bl_night = np.sqrt(mean_squared_error(y_true_c[~pv_mask], y_baseline_c[~pv_mask]))
    rmse_md_night = np.sqrt(mean_squared_error(y_true_c[~pv_mask], y_pred_c[~pv_mask]))

    imp_pv = (mae_bl_pv - mae_md_pv) / mae_bl_pv * 100
    imp_night = (mae_bl_night - mae_md_night) / mae_bl_night * 100

    print(f"\n  {'=' * 70}")
    print(f"  COMPARAISON {model_name.upper()} vs BASELINE OIKEN")
    print(f"  {'=' * 70}")
    print(f"  {'':30s} {'Baseline':>10s} {model_name:>12s} {'Delta':>10s}")
    print(f"  {'-' * 70}")
    print(f"  {'GLOBAL MAE':30s} {mae_bl:10.4f} {mae_md:12.4f} {imp_mae:+9.1f}%")
    print(f"  {'GLOBAL RMSE':30s} {rmse_bl:10.4f} {rmse_md:12.4f} {imp_rmse:+9.1f}%")
    print(f"  {'-' * 70}")
    print(f"  {'JOUR-PV (10-16h) MAE':30s} {mae_bl_pv:10.4f} {mae_md_pv:12.4f} {imp_pv:+9.1f}%")
    print(f"  {'JOUR-PV (10-16h) RMSE':30s} {rmse_bl_pv:10.4f} {rmse_md_pv:12.4f}")
    print(f"  {'NUIT (reste) MAE':30s} {mae_bl_night:10.4f} {mae_md_night:12.4f} {imp_night:+9.1f}%")
    print(f"  {'NUIT (reste) RMSE':30s} {rmse_bl_night:10.4f} {rmse_md_night:12.4f}")
    print(f"  (13-16 sept 2025 exclus : baseline corrompue)")
    print(f"  {'=' * 70}")

    return mae_bl, mae_md


def plot_predictions(df, y_true, y_pred, model_name, save_dir):
    n = len(y_true)
    start = n // 3
    end = start + 96 * 7

    ts = df["timestamp"][start:end].to_list()
    yt = y_true[start:end]
    yp = y_pred[start:end]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])
    fig.suptitle(f"Prediction vs Realite - {model_name} (Forecast v8 J+1)",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(ts, yt, label="Reel", color="#2980b9", linewidth=1.2)
    ax.plot(ts, yp, label="Predit (J+1)", color="#8e44ad",
            linewidth=1.2, alpha=0.8)
    ax.fill_between(ts, yt, yp, alpha=0.15, color="#8e44ad")
    ax.set_ylabel("Charge standardisee [-]")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    ax = axes[1]
    ax.bar(ts, yt - yp, width=0.01, color="#8e44ad", alpha=0.6)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Erreur [-]")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

    plt.tight_layout()
    path = save_dir / "forecast_v8_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure : {path}")


def plot_feature_importance(model, feature_cols, save_dir):
    importances = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importances)[::-1][:25]

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [feature_cols[i] for i in sorted_idx]
    values = importances[sorted_idx]
    ax.barh(range(len(names)), values[::-1], color="#2980b9")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Gain")
    ax.set_title("Top 25 Features - Forecast v8 (LightGBM)")
    plt.tight_layout()
    path = save_dir / "forecast_v8_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure : {path}")


def plot_shap_analysis(model, X_val, feature_cols, save_dir):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val[:2000])
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_val[:2000],
                          feature_names=feature_cols, show=False,
                          max_display=25)
        plt.title("SHAP Summary - Forecast v8")
        plt.tight_layout()
        path = save_dir / "forecast_v8_shap_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  SHAP : {path}")
    except ImportError:
        print("  Warning: shap non installe, analyse SHAP ignoree")


def plot_hourly_comparison(df, y_pred, save_dir):
    y_true = df[TARGET].to_numpy()
    y_baseline = df[BASELINE].to_numpy()
    hours = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TIMEZONE).dt.hour().alias("_h")
    )["_h"].to_numpy()

    valid = ~np.isnan(y_baseline)
    h_range = range(24)
    mae_bl_h = []
    mae_v8_h = []
    for h in h_range:
        mask = (hours == h) & valid
        mae_bl_h.append(mean_absolute_error(y_true[mask], y_baseline[mask]))
        mae_v8_h.append(mean_absolute_error(y_true[mask], y_pred[mask]))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)
    w = 0.35
    ax.bar(x - w / 2, mae_bl_h, w, label="Baseline OIKEN", color="#95a5a6")
    ax.bar(x + w / 2, mae_v8_h, w, label="Forecast v8", color="#8e44ad")
    ax.set_xlabel("Heure")
    ax.set_ylabel("MAE [-]")
    ax.set_title("MAE par heure - Forecast v8 vs Baseline OIKEN")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}h" for h in h_range], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = save_dir / "forecast_v8_hourly_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure : {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pv-weight", type=float, default=PV_WEIGHT_DEFAULT)
    parser.add_argument("--trials", type=int, default=N_OPTUNA_TRIALS)
    parser.add_argument("--window", type=int, default=SLIDING_WINDOW_MONTHS)
    args = parser.parse_args()

    pv_weight = args.pv_weight
    n_trials = args.trials
    window = args.window

    pv_label = f"x{pv_weight}" if pv_weight > 1.0 else "AUCUNE (uniforme)"

    print("=" * 60)
    print("MODELISATION FORECAST v8 (J+1) - STRICT NO-LEAK")
    print(f"  Ponderation heures PV : {pv_label}")
    print(f"  Fenetre d'entrainement : {window} mois")
    print(f"  Optuna : {n_trials} trials, max {MAX_BOOST_ROUNDS} rounds")
    print(f"  Objectif : Huber (robuste aux outliers)")
    print("=" * 60)

    print("\nChargement du dataset...")
    df = pl.read_parquet(DATA_PROCESSED / "dataset_features.parquet")
    print(f"  {df.shape[0]} lignes, {df.shape[1]} colonnes")

    feature_cols = get_forecast_features(df)
    print(f"\n  Features forecast no-leak : {len(feature_cols)}")
    for f in feature_cols:
        print(f"    {f}")

    print("\nSplit temporel...")
    df_train, df_val, df_test = temporal_split(
        df, sliding_window_months=window
    )

    w_train = compute_sample_weights(df_train, PV_HOURS, pv_weight)
    if w_train is not None:
        pv_pct = (w_train > 1).sum() / len(w_train) * 100
        print(f"  Echantillons PV ponderes : {pv_pct:.1f}% (x{pv_weight})")

    X_train, y_train = to_numpy(df_train, feature_cols)
    X_val, y_val = to_numpy(df_val, feature_cols)
    X_test, y_test = to_numpy(df_test, feature_cols)

    print("\n" + "-" * 40)
    print("BASELINE INTERNE - Ridge Regression")
    print("-" * 40)
    ridge_model, ridge_pred_val = train_ridge(
        X_train, y_train, X_val, y_val, w_train=w_train
    )
    ridge_pred_test = ridge_model.predict(X_test)
    evaluate_detailed(df_test, y_test, ridge_pred_test, "Ridge v8")
    compare_with_baseline(df_test, ridge_pred_test, "Ridge v8")

    print("\n" + "-" * 40)
    print("MODELE PRINCIPAL - LightGBM (Forecast v8)")
    print("-" * 40)
    lgb_model, lgb_pred_val, study = train_lightgbm_optuna(
        X_train, y_train, X_val, y_val,
        feature_names=feature_cols,
        w_train=w_train,
        n_trials=n_trials,
        max_boost_rounds=MAX_BOOST_ROUNDS,
    )

    lgb_pred_test = lgb_model.predict(X_test)
    mae_v8 = evaluate_detailed(
        df_test, y_test, lgb_pred_test, "LightGBM Forecast v8"
    )
    compare_with_baseline(df_test, lgb_pred_test, "Forecast v8")

    print("\nGeneration des visualisations...")
    save_dir = DATA_PROCESSED
    plot_predictions(df_test, y_test, lgb_pred_test, "LightGBM", save_dir)
    plot_feature_importance(lgb_model, feature_cols, save_dir)
    plot_shap_analysis(lgb_model, X_val, feature_cols, save_dir)
    plot_hourly_comparison(df_test, lgb_pred_test, save_dir)

    import joblib
    model_path = DATA_PROCESSED / "forecast_v8_lgb_model.joblib"
    joblib.dump(lgb_model, model_path)
    print(f"\n  Modele sauvegarde : {model_path}")

    print("\n" + "=" * 60)
    print("RESUME FORECAST v8 (STRICT NO-LEAK)")
    print("=" * 60)
    print(f"  Test MAE global   : {mae_v8:.4f}")
    print(f"  Historique        : v1=0.2174, v2=0.1977* (*fuite), v7=0.2092")
    print(f"  Baseline OIKEN    : 0.2030")
    print(f"  Ponderation PV    : {pv_label}")
    print(f"  Fenetre train     : {window} mois")
    print(f"  Optuna trials     : {n_trials}")
    print(f"  Objectif          : Huber")
    print(f"  Features          : {len(feature_cols)} (no-leak)")
    print("=" * 60)