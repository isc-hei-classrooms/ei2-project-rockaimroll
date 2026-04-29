"""
Modele XGBoost pour forecast J+1 OIKEN ML - V5 RESIDU PV (multi-dataset).

DIFFERENCE FONDAMENTALE VS V4 (qui etait dans ce fichier avant) :
  La cible n'est plus `load` mais `target = load - gamma * pv_predicted_kwh`,
  avec gamma estime par OLS sur les pas diurnes du training de chaque fold.
  Le modele apprend la composante NON-PV de la charge, plus stationnaire,
  pas de saturation. A l'inference, on reconstruit :
    load_pred(t) = target_pred(t) + gamma * pv_predicted_kwh(t)

Pourquoi cette approche fonctionne (validation empirique sur OIKEN test
ete 2025, voir conversation f06ecbc5) :
  - V3 (load brut)    : MAE diurne ete = 0.68, min pred -2.20 (sature)
  - V4 (post-hoc PV)  : MAE diurne ete = 0.66 (-2.8%)
  - V5 (residu PV)    : MAE diurne ete = 0.58 (-14.3%),
                        min pred -3.15 (saturation eliminee),
                        MAE sur pas extremes y<-1.5 : -42.7%

  La saturation est absorbee analytiquement par le terme gamma*pv,
  qui n'a pas de borne (lineaire) la ou les arbres saturaient.

NO-LEAK :
  pv_predicted_kwh(t) = pred_glob_ctrl(t) * pv_capacity_factor_30d(t)
                        * is_daylight(t)
  Tous calculables a J 11h pour T = J+1. Gamma estime UNIQUEMENT sur
  le training de chaque fold, jamais sur le test.

EXCLUSION DE FEATURES :
  pv_predicted_kwh est exclu des features du modele car il sert a
  construire la cible. Sinon, le modele pourrait trivialement annuler
  la transformation. Les autres features PV (capacity_factor,
  pred_glob_ctrl) restent disponibles : signal climatique distinct.

PV CORRECTION POST-HOC (heritage V4) :
  Conservee en option desactivable via --with-pv-correction.
  Par defaut OFF en V5 car le residu absorbe deja le PV linearement.
  Activable si on observe un biais residuel sur certains folds.

CONSERVATION DU SCHEMA DASHBOARD :
  Le rapport JSON expose maintenant le bloc `residual_target`
  (detecte automatiquement par dashboard.py V3/V4/V5).
  Les blocs `pv_correction` et `drift_correction` restent presents
  pour retrocompatibilite stricte du dashboard.
  Le parquet de sortie garde les colonnes load_pred_raw,
  load_pred_corrected (= meme valeur si --with-pv-correction off)
  et drift_correction_applied (= 0 si V5 pure).

MULTI-DATASET (heritage V4) :
  Preservation totale de --dataset, --all, get_cv_params, suffixes
  de sortie par dataset, cv_overrides pour les datasets courts (golden).

Pre-requis :
  uv add xgboost scikit-learn optuna pvlib

Sorties (suffixees par dataset, compatibles dashboard.py) :
  data/models/xgboost_j1_final_<dataset>.json
  data/processed/predictions_xgboost_cv_<dataset>.parquet
  data/reports/model_xgboost_report_<dataset>.json

CLI :
  python -m src.model_XGBoost                              # V5 pure
  python -m src.model_XGBoost --with-pv-correction         # V5 + post-hoc
  python -m src.model_XGBoost --dataset golden             # autre dataset
  python -m src.model_XGBoost --all                        # tous les datasets
  python -m src.model_XGBoost --quick                      # 10 trials Optuna
  python -m src.model_XGBoost --no-tune                    # params par defaut
  python -m src.model_XGBoost --n-trials 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.config import (
    DATA_PROCESSED,
    DATA_REPORTS,
    DATASETS,
    DEFAULT_DATASET,
    PROJECT_ROOT,
    get_dataset_config,
    get_features_path,
    get_model_path,
    get_predictions_path,
    get_model_report_path,
)


# ============================================================
# Constantes
# ============================================================

LOAD_COL = "load"  # cible originale (sera transformee en residu PV)

# Time-series CV
N_FOLDS = 5
TEST_SIZE_DAYS = 60
MIN_TRAIN_DAYS = 720
SAMPLES_PER_DAY = 96


def get_cv_params(dataset_name: str | None) -> dict:
    """
    Retourne les parametres CV effectifs pour un dataset donne.
    Defauts globaux eventuellement surcharges par 'cv_overrides' dans
    config.DATASETS[dataset_name] (ex : golden trop court pour les
    defauts 720+5*60=1020j).
    """
    params = {
        "n_folds": N_FOLDS,
        "test_size_days": TEST_SIZE_DAYS,
        "min_train_days": MIN_TRAIN_DAYS,
        "samples_per_day": SAMPLES_PER_DAY,
    }
    if dataset_name is None:
        return params
    try:
        cfg = get_dataset_config(dataset_name)
    except (ValueError, KeyError):
        return params
    overrides = cfg.get("cv_overrides", {}) or {}
    unknown = set(overrides) - set(params)
    if unknown:
        raise ValueError(
            f"cv_overrides du dataset '{dataset_name}' contient des "
            f"cles inconnues : {unknown}. Cles autorisees : {set(params)}."
        )
    params.update(overrides)
    return params


# === V5 : proxy PV pour la cible residu ===
# Ordre de priorite (la 1ere colonne presente dans le dataset est utilisee).
# IMPORTANT : la colonne choisie est EXCLUE des features (sinon, le modele
# peut trivialement annuler la transformation de cible).
PV_PROXY_CANDIDATES = [
    "pv_predicted_kwh",
    "pred_glob_ctrl",
    "sun_clearsky_ghi",
]

# Calibration de gamma (V5)
GAMMA_DIURNAL_QUANTILE = 0.50      # filtre : pv > Q50(pv>0) sur le train
GAMMA_TRIM_QUANTILES = (0.02, 0.98)  # trim outliers de load avant OLS
GAMMA_MIN_DIURNAL_SAMPLES = 200    # min de pas diurnes pour estimer gamma

# === V4 (heritage) : PV correction post-hoc, optionnelle ===
PV_CORR_TAIL_DAYS = 180
PV_CORR_AMPLITUDE_CLIP = 0.5
PV_CORR_MIN_DIURNAL_SAMPLES = 200
PV_CORR_THRESHOLD_QUANTILE = 0.50
PV_CORR_TRIM_QUANTILES = (0.05, 0.95)

# Optuna
OPTUNA_N_TRIALS_DEFAULT = 20
OPTUNA_N_FOLDS_FOR_TUNING = 3
OPTUNA_RANDOM_SEED = 42

# Colonnes a exclure des features.
# pv_predicted_kwh est exclu DYNAMIQUEMENT (cf select_feature_cols) :
# si c'est la colonne choisie comme proxy de cible, elle ne doit pas
# etre vue par le modele.
EXCLUDED_FROM_FEATURES = frozenset({
    "timestamp",
    "load",
    "forecast_load",            # baseline OIKEN, reservee a la metrique
    "load_residual",
    "pv_total", "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote",
    "meteo_temperature_2m", "meteo_pressure", "meteo_humidity",
    "meteo_wind_speed", "meteo_global_radiation", "meteo_gust_peak",
    "meteo_precipitation", "meteo_sunshine_duration",
    "meteo_wind_direction",
})

XGB_PARAMS_DEFAULT = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 1500,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

DATA_MODELS = PROJECT_ROOT / "data" / "models"


# ============================================================
# Utilitaires affichage
# ============================================================

def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def print_section(title: str, char: str = "=", width: int = 70) -> None:
    print("\n" + char * width)
    print(title)
    print(char * width)


# ============================================================
# Garde-fou anti-leak
# ============================================================

def assert_standalone(feature_cols: list[str]) -> None:
    forbidden = []
    for c in feature_cols:
        if c == "forecast_load":
            forbidden.append(c)
        elif "forecast_" in c or "residual" in c:
            forbidden.append(c)
    if forbidden:
        raise RuntimeError(
            "Mode standalone viole : features interdites detectees : "
            f"{forbidden}. Verifie EXCLUDED_FROM_FEATURES et "
            "le pipeline features.py."
        )


# ============================================================
# Selection PV proxy + features
# ============================================================

def select_pv_proxy_column(df: pl.DataFrame) -> str | None:
    """1ere colonne de PV_PROXY_CANDIDATES presente dans df."""
    for c in PV_PROXY_CANDIDATES:
        if c in df.columns:
            return c
    return None


def select_feature_cols(df: pl.DataFrame,
                         pv_proxy_col: str | None) -> list[str]:
    """
    Selectionne les features en excluant :
      - EXCLUDED_FROM_FEATURES (statique : load, forecast_load, etc.)
      - pv_proxy_col si present : il sert a construire la cible V5.
    """
    excluded = set(EXCLUDED_FROM_FEATURES)
    if pv_proxy_col is not None:
        excluded.add(pv_proxy_col)
    cols = [c for c in df.columns if c not in excluded]
    assert_standalone(cols)
    return cols


def split_X(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    X uniquement (la cible est calculee differemment en V5).
    Sanitisation NaN/inf -> 0 par securite (XGBoost accepte les NaN
    en X mais on prefere etre explicite).
    """
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# Time-series CV (expanding window)
# ============================================================

def make_ts_folds(
    n_samples: int,
    n_folds: int = N_FOLDS,
    test_size_days: int = TEST_SIZE_DAYS,
    min_train_days: int = MIN_TRAIN_DAYS,
    samples_per_day: int = SAMPLES_PER_DAY,
) -> list[tuple[np.ndarray, np.ndarray]]:
    test_size = test_size_days * samples_per_day
    min_train = min_train_days * samples_per_day
    needed = min_train + n_folds * test_size

    if n_samples < needed:
        raise ValueError(
            f"Pas assez de donnees pour {n_folds} folds : besoin de "
            f"{needed:,} samples ({needed//samples_per_day} jours), "
            f"dispo {n_samples:,} ({n_samples//samples_per_day} jours)."
        )

    folds = []
    for k in range(n_folds):
        train_end = min_train + k * test_size
        test_start = train_end
        test_end = test_start + test_size
        train_idx = np.arange(0, train_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        folds.append((train_idx, test_idx))

    return folds


# ============================================================
# === V5 : Transformation de cible (residu PV) ===
# ============================================================

def estimate_gamma(
    load_train: np.ndarray,
    pv_proxy_train: np.ndarray,
    diurnal_quantile: float = GAMMA_DIURNAL_QUANTILE,
    trim_q: tuple[float, float] = GAMMA_TRIM_QUANTILES,
    min_samples: int = GAMMA_MIN_DIURNAL_SAMPLES,
) -> tuple[float, dict]:
    """
    Estime gamma par OLS : load = const + gamma * pv_proxy + residu
    sur les pas DIURNES du train uniquement (pv > Q50(pv>0)).

    Pourquoi diurne uniquement : le couplage load <-> PV n'existe que
    le jour. Inclure la nuit (pv=0) ferait coller gamma a la moyenne
    nocturne, sans rapport avec la dynamique PV qu'on veut absorber.

    Retourne : gamma (signe attendu : negatif), info_dict avec R^2,
    n_diurnal, threshold, etc. Si echec (proxy quasi nul, trop peu
    de points), gamma = 0.0 (V5 degenere en V3).
    """
    n = len(load_train)
    if n != len(pv_proxy_train):
        raise ValueError(
            f"Tailles incompatibles : load={n}, pv={len(pv_proxy_train)}"
        )

    # Sanitisation : on travaille sur les pas finis uniquement
    finite_mask = np.isfinite(load_train) & np.isfinite(pv_proxy_train)
    load_clean = load_train[finite_mask]
    pv_clean = pv_proxy_train[finite_mask]
    n_finite = len(load_clean)

    if n_finite < 100:
        return 0.0, {
            "warning": "trop peu de pas finis dans (load, pv_proxy)",
            "n_finite": int(n_finite),
        }

    # Seuil diurne : Q50 des pv strictement positifs
    pv_pos = pv_clean[pv_clean > 1e-6]
    if len(pv_pos) < 10:
        return 0.0, {
            "warning": "pv_proxy quasi nul (hiver pur ?)",
            "n_pv_positive": int(len(pv_pos)),
        }
    threshold = float(np.quantile(pv_pos, diurnal_quantile))

    mask = pv_clean > threshold
    n_diurnal = int(mask.sum())
    if n_diurnal < min_samples:
        return 0.0, {
            "warning": (
                f"Trop peu de pas diurnes (< {min_samples}), gamma=0"
            ),
            "n_diurnal": n_diurnal,
            "threshold_pv": threshold,
        }

    load_d = load_clean[mask]
    pv_d = pv_clean[mask]

    # Trim outliers sur load (les jours atypiques : pannes, jours feries
    # mal etiquetes, etc.) avant OLS pour robustesse.
    q_lo, q_hi = np.quantile(load_d, list(trim_q))
    keep = (load_d >= q_lo) & (load_d <= q_hi)
    load_fit = load_d[keep]
    pv_fit = pv_d[keep]

    if len(load_fit) < min_samples // 2:
        return 0.0, {
            "warning": "trop peu de pas apres trim outliers",
            "n_clean": int(len(load_fit)),
        }

    # OLS : load = const + gamma * pv (np.polyfit retourne [pente, intercept])
    gamma, const = np.polyfit(pv_fit, load_fit, deg=1)

    # R^2
    pred_load = gamma * pv_fit + const
    ss_res = float(np.sum((load_fit - pred_load) ** 2))
    ss_tot = float(np.sum((load_fit - load_fit.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    info = {
        "n_diurnal": n_diurnal,
        "n_clean": int(len(load_fit)),
        "threshold_pv": threshold,
        "gamma": float(gamma),
        "const": float(const),
        "r2": float(r2),
        "load_mean_diurnal": float(load_fit.mean()),
        "load_std_diurnal": float(load_fit.std()),
        "pv_mean_diurnal": float(pv_fit.mean()),
        "pv_max_diurnal": float(pv_fit.max()),
    }
    return float(gamma), info


def make_target_residual(
    load: np.ndarray,
    pv_proxy: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    target = load - gamma * pv_proxy
    Sanitisation des NaN/inf (proviennent de pv_proxy en debut de serie
    a cause du rolling 30j) -> 0 sur la difference.
    XGBoost refuse les NaN dans le label, contrairement a X.
    """
    target = load - gamma * pv_proxy
    return np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)


def reconstruct_load(
    target_pred: np.ndarray,
    pv_proxy: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    load_pred = target_pred + gamma * pv_proxy
    Sanitisation symetrique : si pv_proxy est NaN a l'inference,
    on contribue 0 (= retour a load brut sur ce pas).
    """
    pv_safe = np.nan_to_num(pv_proxy, nan=0.0, posinf=0.0, neginf=0.0)
    return target_pred + gamma * pv_safe


# ============================================================
# === V4 (heritage) : PV correction post-hoc, optionnelle ===
# ============================================================

def estimate_pv_correction(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    pv_proxy_train: np.ndarray,
    tail_days: int = PV_CORR_TAIL_DAYS,
    samples_per_day: int = SAMPLES_PER_DAY,
    threshold_quantile: float = PV_CORR_THRESHOLD_QUANTILE,
    trim_q: tuple[float, float] = PV_CORR_TRIM_QUANTILES,
    min_samples: int = PV_CORR_MIN_DIURNAL_SAMPLES,
) -> tuple[float, float, float, dict]:
    """
    Heritage V4. Apprend err = beta * pv_proxy + alpha sur la queue
    du train, restreint aux pas diurnes. Utilise UNIQUEMENT si
    --with-pv-correction est passe sur la CLI.
    """
    n = len(y_true_train)
    tail_samples = min(tail_days * samples_per_day, n)
    sl = slice(n - tail_samples, n)

    err = y_true_train[sl] - y_pred_train[sl]
    pv = pv_proxy_train[sl]

    pv_pos = pv[pv > 1e-6]
    if len(pv_pos) < 10:
        return 0.0, 0.0, 0.0, {
            "warning": "pv_proxy quasi nul sur tail",
            "n_pv_positive": int(len(pv_pos)),
        }
    threshold = float(np.quantile(pv_pos, threshold_quantile))

    mask = pv > threshold
    n_diurnal = int(mask.sum())
    if n_diurnal < min_samples:
        return 0.0, 0.0, threshold, {
            "warning": f"Trop peu de pas diurnes (< {min_samples})",
            "n_diurnal": n_diurnal,
            "threshold_pv": threshold,
        }

    err_d = err[mask]
    pv_d = pv[mask]

    q_lo, q_hi = np.quantile(err_d, list(trim_q))
    keep = (err_d >= q_lo) & (err_d <= q_hi)
    err_clean = err_d[keep]
    pv_clean = pv_d[keep]

    if len(err_clean) < min_samples // 2:
        return 0.0, 0.0, threshold, {
            "warning": "trop peu de pas apres trim",
            "n_clean": int(len(err_clean)),
        }

    beta, alpha = np.polyfit(pv_clean, err_clean, deg=1)
    pred_err = beta * pv_clean + alpha
    ss_res = float(np.sum((err_clean - pred_err) ** 2))
    ss_tot = float(np.sum((err_clean - err_clean.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    info = {
        "n_diurnal": n_diurnal,
        "n_clean": int(len(err_clean)),
        "threshold_pv": threshold,
        "beta": float(beta),
        "alpha": float(alpha),
        "r2": float(r2),
        "tail_days_used": int(tail_days),
    }
    return float(beta), float(alpha), float(threshold), info


def apply_pv_correction(
    y_pred: np.ndarray,
    pv_proxy_test: np.ndarray,
    beta: float,
    alpha: float,
    threshold: float,
    amplitude_clip: float = PV_CORR_AMPLITUDE_CLIP,
) -> tuple[np.ndarray, np.ndarray]:
    """Correction conditionnelle (heritage V4)."""
    pv_safe = np.nan_to_num(pv_proxy_test, nan=0.0)
    raw_correction = beta * pv_safe + alpha
    correction = np.where(pv_safe > threshold, raw_correction, 0.0)
    correction = np.clip(correction, -amplitude_clip, amplitude_clip)
    return y_pred + correction, correction


# ============================================================
# Metriques (cles alignees sur dashboard.py)
# ============================================================

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray,
              threshold: float = 0.1) -> float:
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])
                         / np.abs(y_true[mask])) * 100)


def compute_metrics(
    load_true: np.ndarray,
    load_pred: np.ndarray,
    forecast_baseline: np.ndarray,
    drift_correction: np.ndarray | None = None,
) -> dict:
    """
    Cles alignees dashboard. `drift_correction` (nom legacy) contient
    soit None (V5 pure), soit la correction post-hoc V4 si activee.
    """
    if drift_correction is not None:
        load_pred_used = load_pred + drift_correction
    else:
        load_pred_used = load_pred

    mae_load = float(np.mean(np.abs(load_true - load_pred_used)))
    rmse_load = float(np.sqrt(np.mean((load_true - load_pred_used) ** 2)))
    mape_load = safe_mape(load_true, load_pred_used)

    mae_baseline = float(np.mean(np.abs(load_true - forecast_baseline)))
    rmse_baseline = float(np.sqrt(np.mean(
        (load_true - forecast_baseline) ** 2)))
    mape_baseline = safe_mape(load_true, forecast_baseline)

    gain_mae = (mae_load / mae_baseline if mae_baseline > 0
                else float("nan"))
    gain_rmse = (rmse_load / rmse_baseline if rmse_baseline > 0
                 else float("nan"))

    return {
        "mae_load": mae_load,
        "rmse_load": rmse_load,
        "mape_load_pct": mape_load,
        "mae_baseline": mae_baseline,
        "rmse_baseline": rmse_baseline,
        "mape_baseline_pct": mape_baseline,
        "gain_mae_vs_baseline": gain_mae,
        "gain_rmse_vs_baseline": gain_rmse,
    }


def compute_diurnal_metrics(
    timestamps: list,
    load_true: np.ndarray,
    load_pred: np.ndarray,
    forecast_baseline: np.ndarray,
    hour_min: int = 10,
    hour_max: int = 16,
) -> dict:
    """MAE / gain restreints aux pas 10h-16h (cible saturation V3/V4)."""
    hours = np.array([t.hour for t in timestamps])
    mask = (hours >= hour_min) & (hours < hour_max)
    if mask.sum() == 0:
        return {"warning": "no diurnal samples in range"}

    yt = load_true[mask]
    yp = load_pred[mask]
    yb = forecast_baseline[mask]

    mae_model = float(np.mean(np.abs(yt - yp)))
    mae_base = float(np.mean(np.abs(yt - yb)))
    return {
        "n_diurnal_steps": int(mask.sum()),
        "mae_diurnal_model": mae_model,
        "mae_diurnal_baseline": mae_base,
        "ratio_diurnal_model_over_baseline": (
            mae_model / mae_base if mae_base > 0 else float("nan")
        ),
        "hour_window": [hour_min, hour_max],
    }


def compute_extreme_metrics(
    load_true: np.ndarray,
    load_pred: np.ndarray,
    forecast_baseline: np.ndarray,
    threshold: float = -1.5,
) -> dict:
    """
    Diagnostic V5 specifique : MAE sur les pas y_true < threshold
    (les creux profonds que V3/V4 ne savaient pas atteindre).
    Validation directe de la suppression de saturation.
    """
    mask = load_true < threshold
    if mask.sum() < 5:
        return {
            "n_extreme_steps": int(mask.sum()),
            "warning": "trop peu de pas extremes, metrique non fiable",
        }
    return {
        "n_extreme_steps": int(mask.sum()),
        "threshold_load": threshold,
        "mae_extreme_model": float(np.mean(np.abs(
            load_true[mask] - load_pred[mask]))),
        "mae_extreme_baseline": float(np.mean(np.abs(
            load_true[mask] - forecast_baseline[mask]))),
        "min_load_true": float(load_true.min()),
        "min_load_pred": float(load_pred.min()),
        "deficit_min": float(load_true.min() - load_pred.min()),
    }


# ============================================================
# Optuna tuning (V5 : MAE evaluee sur load reconstruit)
# ============================================================

def make_optuna_objective(
    df: pl.DataFrame,
    feature_cols: list[str],
    pv_proxy_col: str | None,
    folds_for_tuning: list[tuple[np.ndarray, np.ndarray]],
):
    import xgboost as xgb

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int(
                "n_estimators", 500, 2500, step=100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 5, 50),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        maes = []
        for fold_k, (tr, te) in enumerate(folds_for_tuning):
            df_tr = df[tr]
            df_te = df[te]

            X_tr = split_X(df_tr, feature_cols)
            X_te = split_X(df_te, feature_cols)
            load_tr = df_tr[LOAD_COL].to_numpy().astype(np.float32)
            load_te = df_te[LOAD_COL].to_numpy().astype(np.float32)

            # === V5 : transformation de cible ===
            if pv_proxy_col is not None:
                pv_tr = df_tr[pv_proxy_col].to_numpy().astype(np.float64)
                pv_te = df_te[pv_proxy_col].to_numpy().astype(np.float64)
                gamma, _ = estimate_gamma(load_tr, pv_tr)
                target_tr = make_target_residual(
                    load_tr, pv_tr, gamma).astype(np.float32)
            else:
                # Pas de proxy PV : V5 degenere en V3 (cible = load)
                gamma = 0.0
                pv_te = np.zeros_like(load_te, dtype=np.float64)
                target_tr = load_tr.astype(np.float32)

            # Garde-fou : si target a encore des NaN/inf, penalise le trial
            if not np.isfinite(target_tr).all():
                maes.append(1e6)
                continue

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, target_tr, verbose=False)
            target_pred = model.predict(X_te)
            load_pred = reconstruct_load(target_pred, pv_te, gamma)

            # MAE sur LOAD reconstruit (la metrique business)
            mae = float(np.mean(np.abs(load_te - load_pred)))
            maes.append(mae)

            trial.report(mae, fold_k)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        return float(np.mean(maes))

    return objective


def make_progress_callback(n_trials: int, start_time: float):
    import optuna

    def callback(study, trial):
        n_done = len(study.trials)
        n_completed = sum(1 for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned = sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.PRUNED)

        elapsed = time.time() - start_time
        avg_per_trial = elapsed / max(1, n_done)
        remaining_trials = max(0, n_trials - n_done)
        eta = remaining_trials * avg_per_trial

        if trial.state == optuna.trial.TrialState.COMPLETE:
            state_str = "OK    "
            val_str = f"{trial.value:.4f}"
        elif trial.state == optuna.trial.TrialState.PRUNED:
            state_str = "PRUNED"
            val_str = "  -   "
        else:
            state_str = "FAIL  "
            val_str = "  -   "

        try:
            best_str = f"{study.best_value:.4f}"
        except ValueError:
            best_str = "  -   "

        progress = n_done / n_trials
        bar_n = int(progress * 20)
        bar = "#" * bar_n + "-" * (20 - bar_n)

        print(f"  [{bar}] {n_done:>2}/{n_trials} {state_str} "
              f"MAE={val_str} | best={best_str} | "
              f"elapsed={fmt_time(elapsed)} | ETA={fmt_time(eta)} | "
              f"OK={n_completed} prune={n_pruned}",
              flush=True)

    return callback


def run_optuna_tuning(
    df: pl.DataFrame,
    feature_cols: list[str],
    pv_proxy_col: str | None,
    n_trials: int,
    n_folds_for_tuning: int = OPTUNA_N_FOLDS_FOR_TUNING,
    cv_params: dict | None = None,
) -> tuple[dict, dict]:
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ImportError:
        print("\nERREUR: optuna non installe. Lance:")
        print("  uv add optuna  (ou: pip install optuna)")
        raise

    print_section(
        f"OPTUNA TUNING ({n_trials} trials, "
        f"{n_folds_for_tuning} derniers folds)",
        char="-"
    )
    print(f"  TPE sampler + MedianPruner (n_startup=5)")
    print(f"  Espace de recherche : 8 hyperparams XGBoost")
    print(f"  Objectif : MAE sur LOAD reconstruit (V5 = target + gamma*pv)")
    print(f"")

    if cv_params is None:
        cv_params = {}
    all_folds = make_ts_folds(df.height, **cv_params)
    folds_for_tuning = all_folds[-n_folds_for_tuning:]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=OPTUNA_RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_optuna_objective(
        df, feature_cols, pv_proxy_col, folds_for_tuning)

    t0 = time.time()
    callback = make_progress_callback(n_trials, t0)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        catch=(Exception,),
        callbacks=[callback],
    )
    tuning_time = time.time() - t0

    print(f"\n  Tuning termine en {fmt_time(tuning_time)} "
          f"({tuning_time/60:.1f} min)")
    print(f"  Best MAE load : {study.best_value:.4f}")
    print(f"  Best params :")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k:20s}: {v:.5f}")
        else:
            print(f"    {k:20s}: {v}")

    best_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        **study.best_params,
    }

    info = {
        "n_trials": n_trials,
        "n_folds_for_tuning": n_folds_for_tuning,
        "tuning_time_s": round(tuning_time, 1),
        "best_value_mae": float(study.best_value),
        "n_trials_completed": len(study.trials),
        "n_trials_pruned": sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED),
    }
    return best_params, info


# ============================================================
# === V5 : Entrainement d'un fold avec residu PV ===
# ============================================================

def train_fold(
    fold_idx: int,
    df: pl.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    pv_proxy_col: str | None,
    xgb_params: dict,
    with_pv_correction: bool,
    fit_times_history: list[float] | None = None,
    n_folds_total: int = N_FOLDS,
) -> dict:
    """
    V5 : entraine sur target = load - gamma*pv, predit, reconstruit.
    PV correction post-hoc V4 cumulative si with_pv_correction=True.
    """
    import xgboost as xgb

    if fit_times_history is not None and len(fit_times_history) > 0:
        avg_fit = sum(fit_times_history) / len(fit_times_history)
        folds_remaining = n_folds_total - fold_idx
        eta = folds_remaining * (avg_fit + 5)
        eta_str = f", ETA {fmt_time(eta)}"
    else:
        eta_str = ""

    n_train_days = len(train_idx) // SAMPLES_PER_DAY
    n_test_days = len(test_idx) // SAMPLES_PER_DAY
    print_section(
        f"FOLD {fold_idx + 1}/{n_folds_total} "
        f"(train {n_train_days}j, test {n_test_days}j{eta_str})",
        char="-"
    )

    df_train = df[train_idx]
    df_test = df[test_idx]

    ts_test_start = str(df_test["timestamp"][0])[:16]
    ts_test_end = str(df_test["timestamp"][-1])[:16]
    print(f"  Periode test : {ts_test_start} -> {ts_test_end}")

    if "forecast_load" not in df_test.columns:
        raise RuntimeError(
            "forecast_load absent : impossible de calculer la metrique "
            "vs baseline OIKEN."
        )
    forecast_baseline = df_test["forecast_load"].to_numpy()
    load_test_true = df_test[LOAD_COL].to_numpy()
    load_train = df_train[LOAD_COL].to_numpy().astype(np.float32)

    X_train = split_X(df_train, feature_cols)
    X_test = split_X(df_test, feature_cols)

    # === V5 : transformation de cible ===
    if pv_proxy_col is not None:
        pv_train = df_train[pv_proxy_col].to_numpy().astype(np.float64)
        pv_test = df_test[pv_proxy_col].to_numpy().astype(np.float64)
        gamma, gamma_info = estimate_gamma(load_train, pv_train)
        print(f"  Cible V5 : target = load - gamma * {pv_proxy_col}")
        print(f"    gamma={gamma:+.6f}, R^2={gamma_info.get('r2', 0):.3f}, "
              f"n_diurnal={gamma_info.get('n_diurnal', 0)}")
        if gamma == 0.0 and "warning" in gamma_info:
            print(f"    [WARN gamma=0] : {gamma_info['warning']}")
        target_train = make_target_residual(load_train, pv_train, gamma)
    else:
        # Pas de proxy : V5 degenere en V3 (cible = load brut)
        print(f"  [WARN] Pas de proxy PV : V5 degenere en V3 (cible=load)")
        gamma = 0.0
        gamma_info = {"warning": "no pv_proxy column"}
        pv_train = np.zeros_like(load_train, dtype=np.float64)
        pv_test = np.zeros_like(load_test_true, dtype=np.float64)
        target_train = load_train.copy()

    # Garde-fou : target doit etre finie partout
    if not np.isfinite(target_train).all():
        n_nan = int(np.sum(~np.isfinite(target_train)))
        print(f"    [WARN] target_train avait {n_nan} NaN/inf, "
              "deja sanitise par make_target_residual")

    print(f"  Entrainement XGBoost (V5 cible residu)...", flush=True)
    t0 = time.time()
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, target_train, verbose=False)
    fit_time = time.time() - t0
    print(f"    Fit time : {fit_time:.1f}s")
    if fit_times_history is not None:
        fit_times_history.append(fit_time)

    # Predictions V5 : reconstruire load
    target_pred_train = model.predict(X_train)
    target_pred_test = model.predict(X_test)
    load_pred_train = reconstruct_load(target_pred_train, pv_train, gamma)
    load_pred_test = reconstruct_load(target_pred_test, pv_test, gamma)

    # === Heritage V4 : PV correction post-hoc CUMULATIVE (optionnelle) ===
    if with_pv_correction and pv_proxy_col is not None:
        beta, alpha, threshold, corr_info = estimate_pv_correction(
            y_true_train=load_train,
            y_pred_train=load_pred_train,
            pv_proxy_train=pv_train,
        )
        _, correction = apply_pv_correction(
            y_pred=load_pred_test,
            pv_proxy_test=pv_test,
            beta=beta, alpha=alpha, threshold=threshold,
        )
        print(f"  PV correction post-hoc cumulee :")
        print(f"    beta={beta:+.6f}, alpha={alpha:+.4f}, "
              f"R^2={corr_info.get('r2', 0):.3f}")
        n_corrected = int(np.sum(np.abs(correction) > 1e-6))
        print(f"    n_corrected={n_corrected} "
              f"({100.0 * n_corrected / len(correction):.1f}%), "
              f"range=[{correction.min():+.4f}, {correction.max():+.4f}]")
    else:
        beta, alpha, threshold = 0.0, 0.0, 0.0
        corr_info = {"info": "pv_correction post-hoc disabled (V5 pure)"}
        correction = np.zeros_like(load_pred_test)

    # Metriques
    metrics_v5 = compute_metrics(
        load_true=load_test_true,
        load_pred=load_pred_test,
        forecast_baseline=forecast_baseline,
        drift_correction=None,
    )
    metrics_v5_corr = compute_metrics(
        load_true=load_test_true,
        load_pred=load_pred_test,
        forecast_baseline=forecast_baseline,
        drift_correction=correction,
    )

    # Diagnostic diurne
    timestamps_test = df_test["timestamp"].to_list()
    diurnal_v5 = compute_diurnal_metrics(
        timestamps_test, load_test_true, load_pred_test, forecast_baseline,
    )
    diurnal_v5_corr = compute_diurnal_metrics(
        timestamps_test, load_test_true,
        load_pred_test + correction, forecast_baseline,
    )

    # Diagnostic extreme (signature V5 : suppression de saturation)
    extreme_v5 = compute_extreme_metrics(
        load_test_true, load_pred_test, forecast_baseline,
    )

    gain_mae_v5 = (1 - metrics_v5["gain_mae_vs_baseline"]) * 100
    gain_rmse_v5 = (1 - metrics_v5["gain_rmse_vs_baseline"]) * 100

    print(f"")
    print(f"  {'':22s}  {'MAE':>8s}  {'RMSE':>8s}  {'MAPE':>7s}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*7}")
    print(f"  Baseline OIKEN        : "
          f"{metrics_v5['mae_baseline']:>8.4f}  "
          f"{metrics_v5['rmse_baseline']:>8.4f}  "
          f"{metrics_v5['mape_baseline_pct']:>6.1f}%")
    print(f"  V5 (residu PV)        : "
          f"{metrics_v5['mae_load']:>8.4f}  "
          f"{metrics_v5['rmse_load']:>8.4f}  "
          f"{metrics_v5['mape_load_pct']:>6.1f}%")
    print(f"  Gain V5 vs baseline   : "
          f"{gain_mae_v5:>+7.1f}%  "
          f"{gain_rmse_v5:>+7.1f}%")

    if with_pv_correction:
        gain_mae_v5_corr = (
            1 - metrics_v5_corr["gain_mae_vs_baseline"]) * 100
        gain_rmse_v5_corr = (
            1 - metrics_v5_corr["gain_rmse_vs_baseline"]) * 100
        print(f"  V5 + PV corr post-hoc : "
              f"{metrics_v5_corr['mae_load']:>8.4f}  "
              f"{metrics_v5_corr['rmse_load']:>8.4f}  "
              f"{metrics_v5_corr['mape_load_pct']:>6.1f}%")
        print(f"  Gain corr vs baseline : "
              f"{gain_mae_v5_corr:>+7.1f}%  "
              f"{gain_rmse_v5_corr:>+7.1f}%")

    # Diagnostics cibles
    print(f"")
    print(f"  Diagnostic diurne (10h-16h) :")
    print(f"    Baseline : MAE={diurnal_v5.get('mae_diurnal_baseline', 0):.4f}")
    print(f"    V5       : MAE={diurnal_v5.get('mae_diurnal_model', 0):.4f}")
    if with_pv_correction:
        print(f"    V5+corr  : MAE="
              f"{diurnal_v5_corr.get('mae_diurnal_model', 0):.4f}")

    print(f"  Diagnostic extreme (load_true < -1.5) :")
    if "warning" in extreme_v5:
        print(f"    {extreme_v5.get('warning', '')}")
    else:
        print(f"    n_extreme_steps : {extreme_v5['n_extreme_steps']}")
        print(f"    MAE baseline    : "
              f"{extreme_v5['mae_extreme_baseline']:.4f}")
        print(f"    MAE V5          : {extreme_v5['mae_extreme_model']:.4f}")
        print(f"    min load_true   : {extreme_v5['min_load_true']:+.3f}")
        print(f"    min load_pred   : {extreme_v5['min_load_pred']:+.3f}  "
              f"(deficit {extreme_v5['deficit_min']:+.3f})")
        if abs(extreme_v5['deficit_min']) < 0.20:
            print(f"    -> saturation eliminee (deficit |.| < 0.20)")
        else:
            print(f"    -> saturation residuelle, possiblement amplifier "
                  "le proxy PV")

    load_pred_corrected = load_pred_test + correction

    return {
        "fold": fold_idx + 1,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "test_start": str(df_test["timestamp"][0]),
        "test_end": str(df_test["timestamp"][-1]),
        "fit_time_s": round(fit_time, 1),
        # === V5 : nouveaux champs ===
        "gamma": float(gamma),
        "gamma_info": gamma_info,
        "pv_proxy_col": pv_proxy_col,
        # === V4 heritage (None si --with-pv-correction off) ===
        "pv_correction_info": corr_info,
        "pv_correction_beta": float(beta),
        "pv_correction_alpha": float(alpha),
        "pv_correction_threshold": float(threshold),
        "pv_correction_proxy_col": pv_proxy_col if with_pv_correction
                                                  else None,
        # === Compat dashboard legacy V3 ===
        "drift_slope_a_per_day": 0.0,
        "drift_intercept_b": float(alpha),
        "drift_info": gamma_info,
        # Metriques
        "metrics_no_correction": metrics_v5,
        "metrics_with_correction": metrics_v5_corr,
        "diurnal_metrics_no_correction": diurnal_v5,
        "diurnal_metrics_with_correction": diurnal_v5_corr,
        "extreme_metrics": extreme_v5,
        "_predictions": {
            "timestamp": df_test["timestamp"].to_list(),
            "load_true": load_test_true.tolist(),
            "load_pred_raw": load_pred_test.tolist(),
            "load_pred_corrected": load_pred_corrected.tolist(),
            "drift_correction_applied": correction.tolist(),
            "forecast_baseline": forecast_baseline.tolist(),
            "fold": [fold_idx + 1] * len(test_idx),
        },
    }


# ============================================================
# CV pipeline
# ============================================================

def run_cv(df: pl.DataFrame, xgb_params: dict,
           with_pv_correction: bool,
           cv_params: dict | None = None) -> dict:
    if cv_params is None:
        cv_params = {}
    n_folds_eff = cv_params.get("n_folds", N_FOLDS)
    print_section(
        f"CROSS-VALIDATION {n_folds_eff} folds (expanding window) - V5"
    )

    pv_proxy_col = select_pv_proxy_column(df)
    if pv_proxy_col is None:
        print(f"  [WARN CRITIQUE] Aucun proxy PV trouve parmi "
              f"{PV_PROXY_CANDIDATES}.")
        print(f"  V5 va degenerer en V3 (cible=load) sur tous les folds.")
    else:
        print(f"  Proxy PV pour cible V5 : {pv_proxy_col}")
        print(f"    (exclu des features pour eviter l'annulation triviale)")

    feature_cols = select_feature_cols(df, pv_proxy_col)
    print(f"  {len(feature_cols)} features (sur {df.shape[1]} colonnes)")
    print(f"  Cible V5 : target = load - gamma * {pv_proxy_col or 'N/A'}")
    print(f"  Reference externe : forecast_load (baseline OIKEN)")
    print(f"  PV correction post-hoc cumulative : "
          f"{'OUI' if with_pv_correction else 'NON (V5 pure)'}")

    if cv_params:
        defaults = {"n_folds": N_FOLDS, "test_size_days": TEST_SIZE_DAYS,
                    "min_train_days": MIN_TRAIN_DAYS,
                    "samples_per_day": SAMPLES_PER_DAY}
        diffs = {k: v for k, v in cv_params.items()
                 if defaults.get(k) != v}
        if diffs:
            print(f"  CV overrides actifs : {diffs}")

    # Audit NaN sur colonnes critiques
    print(f"\n  Audit NaN sur colonnes critiques :")
    cols_to_audit = [LOAD_COL, "forecast_load"]
    if pv_proxy_col:
        cols_to_audit.append(pv_proxy_col)
    for col in cols_to_audit:
        if col in df.columns:
            n_nan = df[col].is_null().sum()
            n_total = df.height
            pct = 100.0 * n_nan / n_total if n_total > 0 else 0
            flag = " [ATTENTION]" if n_nan > n_total * 0.05 else ""
            print(f"    {col:30s}: {n_nan:>6,} / {n_total:,} NaN "
                  f"({pct:.2f}%){flag}")

    # Filtre cible NaN
    n_before = df.height
    df = df.filter(pl.col(LOAD_COL).is_not_null())
    n_after = df.height
    if n_after < n_before:
        print(f"  Filtre NaN cible load : {n_before - n_after} lignes "
              "retirees")

    folds = make_ts_folds(df.height, **cv_params)
    print(f"\n  Plan de validation :")
    for k, (tr, te) in enumerate(folds):
        ts_test_start = str(df["timestamp"][int(te[0])])[:10]
        ts_test_end = str(df["timestamp"][int(te[-1])])[:10]
        print(f"    Fold {k+1}: train {len(tr)//SAMPLES_PER_DAY:>4}j | "
              f"test [{ts_test_start} -> {ts_test_end}] "
              f"({len(te)//SAMPLES_PER_DAY}j)")

    cv_start = time.time()
    fit_times_history: list[float] = []
    fold_results = []
    all_predictions = []
    for k, (tr, te) in enumerate(folds):
        res = train_fold(
            k, df, tr, te, feature_cols, pv_proxy_col, xgb_params,
            with_pv_correction=with_pv_correction,
            fit_times_history=fit_times_history,
            n_folds_total=n_folds_eff,
        )
        all_predictions.append(res.pop("_predictions"))
        fold_results.append(res)

    cv_time = time.time() - cv_start
    print_section(f"RESULTATS AGREGES {n_folds_eff} folds "
                  f"({fmt_time(cv_time)})")

    # Stabilite de gamma entre folds (signal de qualite V5)
    gammas = [r["gamma"] for r in fold_results]
    print(f"")
    print(f"  Stabilite de gamma entre folds :")
    print(f"    mean={np.mean(gammas):+.6f}, std={np.std(gammas):.6f}, "
          f"range=[{min(gammas):+.6f}, {max(gammas):+.6f}]")
    if np.std(gammas) > abs(np.mean(gammas)):
        print(f"    [WARN] gamma instable (std > |mean|), proxy PV "
              "probablement bruite")

    print(f"")
    print(f"  {'Metrique':30s}  {'V5 pure':>11s}  {'V5+corr':>11s}  "
          f"{'Std (V5)':>11s}")
    print(f"  {'-'*30}  {'-'*11}  {'-'*11}  {'-'*11}")

    keys_pretty = [
        ("mae_load", "MAE load (V5)"),
        ("rmse_load", "RMSE load (V5)"),
        ("mape_load_pct", "MAPE load (%) (V5)"),
        ("mae_baseline", "MAE baseline OIKEN"),
        ("rmse_baseline", "RMSE baseline OIKEN"),
        ("mape_baseline_pct", "MAPE baseline OIKEN (%)"),
        ("gain_mae_vs_baseline", "Ratio MAE V5/baseline"),
        ("gain_rmse_vs_baseline", "Ratio RMSE V5/baseline"),
    ]
    for key, label in keys_pretty:
        vals_raw = [r["metrics_no_correction"][key] for r in fold_results]
        vals_corr = [r["metrics_with_correction"][key]
                     for r in fold_results]
        mean_raw = float(np.mean(vals_raw))
        mean_corr = float(np.mean(vals_corr))
        std_raw = float(np.std(vals_raw))

        if "pct" in key:
            print(f"  {label:30s}  {mean_raw:>10.2f}%  "
                  f"{mean_corr:>10.2f}%  {std_raw:>10.2f}%")
        else:
            print(f"  {label:30s}  {mean_raw:>11.4f}  "
                  f"{mean_corr:>11.4f}  {std_raw:>11.4f}")

    avg_gain_mae_v5 = (1 - np.mean(
        [r["metrics_no_correction"]["gain_mae_vs_baseline"]
         for r in fold_results])) * 100
    avg_gain_rmse_v5 = (1 - np.mean(
        [r["metrics_no_correction"]["gain_rmse_vs_baseline"]
         for r in fold_results])) * 100
    print(f"")
    print(f"  >>> Gain MAE  (V5 pure)   : {avg_gain_mae_v5:+.1f}%")
    print(f"  >>> Gain RMSE (V5 pure)   : {avg_gain_rmse_v5:+.1f}%")
    if with_pv_correction:
        avg_gain_mae_corr = (1 - np.mean(
            [r["metrics_with_correction"]["gain_mae_vs_baseline"]
             for r in fold_results])) * 100
        print(f"  >>> Gain MAE  (V5+corr)   : {avg_gain_mae_corr:+.1f}%")
    print(f"  (gain positif = on bat OIKEN)")

    # Diagnostic extreme agrege (signature V5)
    print(f"")
    print(f"  Diagnostic extreme (load < -1.5) agrege :")
    deficits = [r["extreme_metrics"].get("deficit_min", 0)
                for r in fold_results
                if "deficit_min" in r["extreme_metrics"]]
    if deficits:
        print(f"    deficit min : mean={np.mean(deficits):+.3f}, "
              f"max={max(deficits, key=abs):+.3f}")
        n_ok = sum(1 for d in deficits if abs(d) < 0.20)
        print(f"    folds avec saturation eliminee (|deficit|<0.20) : "
              f"{n_ok}/{len(deficits)}")

    # Diurne agrege
    print(f"")
    print(f"  Diagnostic diurne (10h-16h) agrege :")
    diurn_base = np.mean([
        r["diurnal_metrics_no_correction"].get("mae_diurnal_baseline",
                                                np.nan)
        for r in fold_results
    ])
    diurn_v5 = np.mean([
        r["diurnal_metrics_no_correction"].get("mae_diurnal_model", np.nan)
        for r in fold_results
    ])
    print(f"    MAE baseline : {diurn_base:.4f}")
    print(f"    MAE V5       : {diurn_v5:.4f}")
    if diurn_v5 < diurn_base:
        gain_diurnal = (diurn_base - diurn_v5) / diurn_base * 100
        print(f"    -> V5 bat la baseline en diurne : -{gain_diurnal:.1f}%")
    else:
        print(f"    -> V5 reste au-dessus de la baseline en diurne, "
              "verifier qualite proxy PV")

    return {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "pv_proxy_col": pv_proxy_col,
        "folds": fold_results,
        "predictions": all_predictions,
        "gammas_per_fold": gammas,
    }


# ============================================================
# Modele final (V5)
# ============================================================

def train_final_model(
    df: pl.DataFrame,
    feature_cols: list[str],
    pv_proxy_col: str | None,
    xgb_params: dict,
    with_pv_correction: bool,
):
    import xgboost as xgb

    print_section("MODELE FINAL V5 (entrainement sur tout le dataset)")

    load_full = df[LOAD_COL].to_numpy().astype(np.float32)
    X = split_X(df, feature_cols)
    print(f"  X : {X.shape}, load : {load_full.shape}")

    if pv_proxy_col is not None:
        pv_full = df[pv_proxy_col].to_numpy().astype(np.float64)
        gamma_final, gamma_info_final = estimate_gamma(load_full, pv_full)
        print(f"  Gamma final (proxy={pv_proxy_col}) : "
              f"{gamma_final:+.6f}, "
              f"R^2={gamma_info_final.get('r2', 0):.3f}")
        target_full = make_target_residual(load_full, pv_full,
                                            gamma_final).astype(np.float32)
    else:
        gamma_final = 0.0
        gamma_info_final = {"warning": "no pv_proxy column"}
        pv_full = np.zeros_like(load_full, dtype=np.float64)
        target_full = load_full.copy()

    print(f"  Entrainement XGBoost final (cible residu V5)...", flush=True)
    t0 = time.time()
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, target_full, verbose=False)
    fit_time = time.time() - t0
    print(f"    Fit time : {fit_time:.1f}s")

    # Calibration finale de la PV correction post-hoc (heritage V4)
    if with_pv_correction and pv_proxy_col is not None:
        target_pred_full = model.predict(X)
        load_pred_full = reconstruct_load(target_pred_full, pv_full,
                                          gamma_final)
        beta, alpha, threshold, corr_info = estimate_pv_correction(
            y_true_train=load_full,
            y_pred_train=load_pred_full,
            pv_proxy_train=pv_full,
        )
        print(f"  PV correction post-hoc finale :")
        print(f"    beta={beta:+.6f}, alpha={alpha:+.4f}, "
              f"threshold={threshold:.3f}, "
              f"R^2={corr_info.get('r2', 0):.3f}")
        pv_correction_final = {
            "beta": float(beta), "alpha": float(alpha),
            "threshold": float(threshold),
            "amplitude_clip": PV_CORR_AMPLITUDE_CLIP,
            "proxy_col": pv_proxy_col, "info": corr_info,
        }
    else:
        pv_correction_final = {
            "beta": 0.0, "alpha": 0.0, "threshold": 0.0,
            "amplitude_clip": PV_CORR_AMPLITUDE_CLIP,
            "proxy_col": pv_proxy_col,
            "info": {"info": "post-hoc disabled (V5 pure)"},
        }

    return model, gamma_final, gamma_info_final, pv_correction_final


# ============================================================
# Sauvegarde (schema aligne dashboard.py V3/V4/V5)
# ============================================================

def save_predictions(all_predictions: list[dict], path: Path) -> None:
    """
    Schema parquet attendu par dashboard.py :
      timestamp, load_true, forecast_baseline,
      load_pred_raw, load_pred_corrected,
      drift_correction_applied, fold

    En V5 pure : load_pred_raw == load_pred_corrected,
                 drift_correction_applied = 0 partout.
    En V5 + post-hoc : drift_correction_applied contient la correction
                       additive cumulee.
    """
    keys = ["timestamp", "load_true", "forecast_baseline",
            "load_pred_raw", "load_pred_corrected",
            "drift_correction_applied", "fold"]
    merged: dict[str, list] = {k: [] for k in keys}
    for p in all_predictions:
        for k in keys:
            merged[k].extend(p[k])

    df_pred = pl.DataFrame(merged)
    df_pred.write_parquet(path)


def feature_importance(model,
                        feature_cols: list[str]) -> list[tuple[str, float]]:
    importances = model.get_booster().get_score(importance_type="gain")
    by_idx = {}
    for k, v in importances.items():
        if k.startswith("f"):
            try:
                idx = int(k[1:])
                by_idx[feature_cols[idx]] = v
            except (ValueError, IndexError):
                by_idx[k] = v
        else:
            by_idx[k] = v
    return sorted(by_idx.items(), key=lambda x: -x[1])


def load_no_leak_check_from_features(dataset_name: str) -> dict | None:
    fpath = DATA_REPORTS / f"features_report_{dataset_name}.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, encoding="utf-8") as f:
            r = json.load(f)
        return r.get("no_leak_check")
    except Exception:
        return None


# ============================================================
# Main
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost J+1 OIKEN ML - V5 RESIDU PV (multi-dataset)"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=DEFAULT_DATASET,
        help=f"Nom du dataset (defaut : {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Entraine successivement sur tous les datasets configures.",
    )
    parser.add_argument(
        "--with-pv-correction",
        action="store_true",
        help="Active la PV correction post-hoc V4 EN PLUS du residu V5. "
             "Utile si V5 laisse un biais residuel sur certains folds.",
    )
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna, utiliser params par defaut")
    parser.add_argument("--quick", action="store_true",
                        help="Optuna rapide (10 trials)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help=f"Nombre de trials Optuna "
                             f"(defaut: {OPTUNA_N_TRIALS_DEFAULT})")
    return parser.parse_args(argv)


def run_one_dataset(args: argparse.Namespace, dataset_name: str) -> int:
    cfg = get_dataset_config(dataset_name)
    label: str = cfg["label"]
    in_path = get_features_path(dataset_name)
    model_path = get_model_path(dataset_name)
    pred_path = get_predictions_path(dataset_name)
    report_path = get_model_report_path(dataset_name)

    global_start = time.time()

    print("=" * 70)
    print(f"MODELE XGBOOST J+1 - OIKEN ML - dataset='{dataset_name}' "
          f"({label})")
    print("V5 RESIDU PV : cible = load - gamma * pv_predicted_kwh")
    print(f"PV correction post-hoc cumulative : "
          f"{'ACTIVE' if args.with_pv_correction else 'desactivee'}")
    print("=" * 70)

    if not in_path.exists():
        print(f"\nERREUR: features absent : {in_path}")
        print(f"Lance d'abord : python -m src.features --dataset "
              f"{dataset_name}")
        return 1

    print(f"\nChargement : {in_path}")
    df = pl.read_parquet(in_path)
    print(f"  {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    print(f"  Periode : {str(df['timestamp'].min())[:16]} -> "
          f"{str(df['timestamp'].max())[:16]}")

    if "forecast_load" not in df.columns:
        print("\nERREUR: forecast_load absent du DataFrame.")
        return 3

    try:
        import xgboost as xgb
        print(f"  XGBoost version : {xgb.__version__}")
    except ImportError:
        print("\nERREUR: xgboost non installe.")
        return 2

    pv_proxy_col = select_pv_proxy_column(df)
    if pv_proxy_col:
        print(f"  Proxy PV detecte    : {pv_proxy_col}")
    else:
        print(f"  Proxy PV detecte    : AUCUN (V5 degenerera en V3)")

    feature_cols = select_feature_cols(df, pv_proxy_col)
    print(f"  Garde-fou standalone : OK ({len(feature_cols)} features)")
    if pv_proxy_col:
        print(f"  ({pv_proxy_col} exclu des features, sert a la cible)")

    cv_params = get_cv_params(dataset_name)
    n_days_avail = df.height // cv_params["samples_per_day"]
    n_days_needed = (cv_params["min_train_days"]
                     + cv_params["n_folds"] * cv_params["test_size_days"])
    print(f"  CV : {cv_params['n_folds']} folds x "
          f"{cv_params['test_size_days']}j test, "
          f"{cv_params['min_train_days']}j train initial = "
          f"{n_days_needed}j requis (dispo {n_days_avail}j, marge "
          f"{n_days_avail - n_days_needed}j)")

    tune_info = None
    if args.no_tune:
        print("\n[--no-tune] Skip Optuna")
        xgb_params = dict(XGB_PARAMS_DEFAULT)
    else:
        if args.quick:
            n_trials = 10
        elif args.n_trials is not None:
            n_trials = args.n_trials
        else:
            n_trials = OPTUNA_N_TRIALS_DEFAULT

        try:
            xgb_params, tune_info = run_optuna_tuning(
                df, feature_cols, pv_proxy_col, n_trials=n_trials,
                cv_params=cv_params,
            )
        except ImportError:
            print("Fallback : params par defaut")
            xgb_params = dict(XGB_PARAMS_DEFAULT)

    cv_results = run_cv(
        df, xgb_params,
        with_pv_correction=args.with_pv_correction,
        cv_params=cv_params,
    )

    model, gamma_final, gamma_info_final, pv_correction_final = (
        train_final_model(
            df, cv_results["feature_cols"], cv_results["pv_proxy_col"],
            xgb_params, with_pv_correction=args.with_pv_correction,
        )
    )

    print_section("TOP 20 FEATURES (importance gain)")
    importance = feature_importance(model, cv_results["feature_cols"])
    for name, score in importance[:20]:
        print(f"  {name:40s}: {score:>10.1f}")

    DATA_MODELS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_REPORTS.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_path))
    save_predictions(cv_results["predictions"], pred_path)

    no_leak_check = load_no_leak_check_from_features(dataset_name)

    report = {
        "dataset": dataset_name,
        "model": "XGBoost",
        "version": "v5_residu_pv",
        "mode": ("standalone + cible residu PV "
                 "(target = load - gamma * pv_proxy)"),
        "target": LOAD_COL,
        "target_transformation": (
            f"target = {LOAD_COL} - gamma * {cv_results['pv_proxy_col']} "
            "(diurne, OLS sur train de chaque fold)"
        ),
        "xgb_params_used": xgb_params,
        "n_features": cv_results["n_features"],
        "feature_cols": cv_results["feature_cols"],
        "cv_strategy": {
            "n_folds": cv_params["n_folds"],
            "test_size_days": cv_params["test_size_days"],
            "min_train_days": cv_params["min_train_days"],
            "scheme": "expanding_window",
        },
        "tuning": tune_info,
        "folds": cv_results["folds"],
        # === V5 : bloc principal (declenche l'affichage V5 du dashboard) ===
        "residual_target": {
            "pv_proxy_col": cv_results["pv_proxy_col"],
            "diurnal_quantile": GAMMA_DIURNAL_QUANTILE,
            "trim_quantiles": list(GAMMA_TRIM_QUANTILES),
            "min_diurnal_samples": GAMMA_MIN_DIURNAL_SAMPLES,
            "gammas_per_fold": cv_results["gammas_per_fold"],
            "final_model_gamma": float(gamma_final),
            "final_model_gamma_info": gamma_info_final,
            "purpose": (
                "absorber l'effet PV lineairement pour eliminer la "
                "saturation des arbres XGBoost dans les creux profonds "
                "de load (ete, fort PV)"
            ),
        },
        # === V4 (heritage, retrocompat dashboard) ===
        "pv_correction": {
            "enabled_in_run": bool(args.with_pv_correction),
            "type": "additive_conditional",
            "form": (
                "load_pred + clip(beta * pv_proxy + alpha, +/- amp)"
                " applied where pv_proxy > threshold"
            ),
            "tail_days_for_calibration": PV_CORR_TAIL_DAYS,
            "amplitude_clip": PV_CORR_AMPLITUDE_CLIP,
            "threshold_quantile_diurnal": PV_CORR_THRESHOLD_QUANTILE,
            "trim_quantiles": list(PV_CORR_TRIM_QUANTILES),
            "min_diurnal_samples": PV_CORR_MIN_DIURNAL_SAMPLES,
            "proxy_col_used": (cv_results["pv_proxy_col"]
                               if args.with_pv_correction else None),
            "final_model_correction": pv_correction_final,
        },
        # === V3 (heritage strict, retrocompat dashboard) ===
        "drift_correction": {
            "type": "DEPRECATED_replaced_by_residual_target",
            "form": "see residual_target block (V5)",
            "final_model_drift": {
                "slope_a_per_day": 0.0,
                "intercept_b": 0.0,
                "t_train_end_days": 0.0,
                "amplitude_clip": PV_CORR_AMPLITUDE_CLIP,
                "info": {"note": "see residual_target for V5 mechanism"},
            },
        },
        "feature_importance_top30": importance[:30],
        "no_leak_check": no_leak_check or {
            "n_warnings": 0,
            "warnings": [],
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    total_time = time.time() - global_start
    print_section(f"MODELE XGBOOST V5 '{dataset_name}' TERMINE en "
                  f"{fmt_time(total_time)}")
    print(f"  Modele final  -> {model_path}")
    print(f"  Predictions   -> {pred_path}")
    print(f"  Rapport       -> {report_path}")
    print(f"  Gamma final   : {gamma_final:+.6f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.all:
        rc_total = 0
        results: dict[str, int] = {}
        for name in DATASETS.keys():
            print()
            try:
                rc = run_one_dataset(args, name)
            except Exception as e:
                print(f"\n[FATAL] Exception dans run_one_dataset('{name}'): "
                      f"{type(e).__name__}: {e}")
                rc = 99
            results[name] = rc
            if rc != 0:
                rc_total = rc
        print()
        print("=" * 70)
        print("RESUME --all")
        print("=" * 70)
        for name, rc in results.items():
            status = "OK" if rc == 0 else f"ECHEC (code {rc})"
            print(f"  {name:15s} : {status}")
        return rc_total

    return run_one_dataset(args, args.dataset)


if __name__ == "__main__":
    sys.exit(main())