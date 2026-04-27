"""
Modele XGBoost pour forecast J+1 OIKEN ML - V2.

Differences vs V1 (= conversation precedente):
  - Drift correction ADDITIVE sur le residu (V1 etait multiplicative
    et inerte sur une cible centree autour de 0)
  - Tuning hyperparametres via Optuna (TPE sampler + pruning median)
  - Comparaison automatique avec/sans correction dans le rapport
  - Defaut a 20 trials Optuna (test rapide, ~15-20 min)
  - Affichages temps reel (progression Optuna, MAE/RMSE OIKEN par fold,
    ETA global)

Strategie:
  - Cible : load_residual = load - forecast_load (creee par features.py)
  - Validation : 5 folds expanding window (time-series CV)
  - Train initial : 720 jours (= 24 mois, 2 cycles saisonniers complets)
  - Test : 60 jours consecutifs par fold
  - Tuning : Optuna sur les 3 derniers folds (= conditions proches du
    deploiement, plus difficiles que les 2 premiers)

Forme de la correction additive:
  pred_residual_corrected(t) = pred_residual(t)
                              + clip(slope_a * (t - t_train_end), -C, +C)

Pre-requis (installer dans le venv via uv ou pip):
  uv add xgboost scikit-learn optuna pvlib

Sorties:
  data/models/xgboost_j1_final.json
  data/processed/predictions_xgboost_cv.parquet
  data/reports/model_xgboost_report.json (inclut best_params Optuna)

CLI:
  python -m src.modelv2              # 20 trials Optuna (~15-20 min)
  python -m src.modelv2 --quick      # 10 trials Optuna (~10 min)
  python -m src.modelv2 --no-tune    # pas de tuning, params par defaut
  python -m src.modelv2 --n-trials 60  # custom
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
    PROJECT_ROOT,
)


# ============================================================
# Constantes
# ============================================================

TARGET_COL = "load_residual"

# Time-series CV
N_FOLDS = 5
TEST_SIZE_DAYS = 60
MIN_TRAIN_DAYS = 720
SAMPLES_PER_DAY = 96

# Drift correction (ADDITIVE sur le residu)
DRIFT_TRAIN_TAIL_DAYS = 180   # 6 mois
DRIFT_AMPLITUDE_CLIP = 0.3    # ~1 std de la cible (std observee = 0.296)
DRIFT_MIN_DAYS_FOR_FIT = 30

# Optuna
OPTUNA_N_TRIALS_DEFAULT = 20
OPTUNA_N_FOLDS_FOR_TUNING = 3
OPTUNA_RANDOM_SEED = 42

# Colonnes a exclure des features (cible + leak)
EXCLUDED_FROM_FEATURES = frozenset({
    "timestamp",
    "load",
    "load_residual",
    "pv_total", "pv_central_valais", "pv_sion", "pv_sierre", "pv_remote",
    "meteo_temperature_2m", "meteo_pressure", "meteo_humidity",
    "meteo_wind_speed", "meteo_global_radiation", "meteo_gust_peak",
    "meteo_precipitation", "meteo_sunshine_duration",
    "meteo_wind_direction",
})

# Hyperparams par defaut (utilises si --no-tune)
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
DATA_REPORTS = PROJECT_ROOT / "data" / "reports"


# ============================================================
# Utilitaires affichage
# ============================================================

def fmt_time(seconds: float) -> str:
    """Formatte une duree en mm:ss ou hh:mm:ss."""
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
# Selection des features
# ============================================================

def select_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDED_FROM_FEATURES]


def split_X_y(df: pl.DataFrame, feature_cols: list[str]):
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df[TARGET_COL].to_numpy().astype(np.float32)
    return X, y


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
# Drift correction ADDITIVE
# ============================================================

def estimate_drift_additive(
    residual_true_train: np.ndarray,
    residual_pred_train: np.ndarray,
    time_idx_train: np.ndarray,
    tail_days: int = DRIFT_TRAIN_TAIL_DAYS,
    samples_per_day: int = SAMPLES_PER_DAY,
) -> tuple[float, float, dict]:
    tail_samples = tail_days * samples_per_day
    n = len(residual_true_train)
    if n < tail_samples:
        tail_samples = n

    sl = slice(n - tail_samples, n)
    err = residual_true_train[sl] - residual_pred_train[sl]
    ti = time_idx_train[sl]

    n_days = tail_samples // samples_per_day
    err_d = err[: n_days * samples_per_day].reshape(
        n_days, samples_per_day).mean(axis=1)
    ti_d = ti[: n_days * samples_per_day].reshape(
        n_days, samples_per_day).mean(axis=1)

    err_q = np.quantile(err_d, [0.05, 0.95])
    keep = (err_d >= err_q[0]) & (err_d <= err_q[1])
    err_clean = err_d[keep]
    ti_clean = ti_d[keep]

    if len(err_clean) < DRIFT_MIN_DAYS_FOR_FIT:
        return 0.0, 0.0, {
            "n_clean_days": int(len(err_clean)),
            "warning": (f"Trop peu de jours (< {DRIFT_MIN_DAYS_FOR_FIT})"
                        f", pas de correction"),
        }

    a, b = np.polyfit(ti_clean, err_clean, deg=1)

    pred_err = a * ti_clean + b
    ss_res = float(np.sum((err_clean - pred_err) ** 2))
    ss_tot = float(np.sum((err_clean - err_clean.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    info = {
        "n_clean_days": int(len(err_clean)),
        "slope_a_per_day": float(a),
        "intercept_b": float(b),
        "r2": float(r2),
        "tail_days_used": int(tail_days),
        "err_mean": float(err_clean.mean()),
        "err_std": float(err_clean.std()),
    }
    return float(a), float(b), info


def apply_drift_correction_additive(
    pred_residual: np.ndarray,
    time_idx_test: np.ndarray,
    slope_a: float,
    time_idx_train_end: float,
    amplitude_clip: float = DRIFT_AMPLITUDE_CLIP,
) -> tuple[np.ndarray, np.ndarray]:
    delta_t = time_idx_test - time_idx_train_end
    correction = slope_a * delta_t
    correction_clipped = np.clip(correction, -amplitude_clip, amplitude_clip)
    return pred_residual + correction_clipped, correction_clipped


# ============================================================
# Metriques
# ============================================================

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray,
              threshold: float = 0.1) -> float:
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])
                         / np.abs(y_true[mask])) * 100)


def compute_metrics(
    residual_true: np.ndarray,
    residual_pred: np.ndarray,
    forecast_load: np.ndarray,
    load_true: np.ndarray,
    drift_correction: np.ndarray | None = None,
) -> dict:
    if drift_correction is not None:
        residual_pred_used = residual_pred + drift_correction
    else:
        residual_pred_used = residual_pred

    mae_res = float(np.mean(np.abs(residual_true - residual_pred_used)))
    rmse_res = float(np.sqrt(
        np.mean((residual_true - residual_pred_used) ** 2)))

    load_pred = forecast_load + residual_pred_used
    mae_load = float(np.mean(np.abs(load_true - load_pred)))
    rmse_load = float(np.sqrt(np.mean((load_true - load_pred) ** 2)))
    mape_load = safe_mape(load_true, load_pred)

    mae_baseline = float(np.mean(np.abs(load_true - forecast_load)))
    rmse_baseline = float(np.sqrt(np.mean((load_true - forecast_load) ** 2)))

    gain_mae = mae_load / mae_baseline if mae_baseline > 0 else float("nan")
    gain_rmse = (rmse_load / rmse_baseline if rmse_baseline > 0
                 else float("nan"))

    return {
        "mae_residual": mae_res,
        "rmse_residual": rmse_res,
        "mae_load": mae_load,
        "rmse_load": rmse_load,
        "mape_load_pct": mape_load,
        "mae_baseline": mae_baseline,
        "rmse_baseline": rmse_baseline,
        "gain_mae_vs_baseline": gain_mae,
        "gain_rmse_vs_baseline": gain_rmse,
    }


# ============================================================
# Optuna tuning (avec progression temps reel)
# ============================================================

def make_optuna_objective(
    df: pl.DataFrame,
    feature_cols: list[str],
    folds_for_tuning: list[tuple[np.ndarray, np.ndarray]],
):
    """Closure-based objective pour Optuna."""
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
            X_tr, y_tr = split_X_y(df_tr, feature_cols)
            X_te, y_te = split_X_y(df_te, feature_cols)

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, verbose=False)
            pred = model.predict(X_te)
            mae = float(np.mean(np.abs(y_te - pred)))
            maes.append(mae)

            trial.report(mae, fold_k)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        return float(np.mean(maes))

    return objective


def make_progress_callback(n_trials: int, start_time: float):
    """
    Callback Optuna affiche apres chaque trial:
      - numero du trial / total
      - etat (OK / PRUNED / FAIL)
      - MAE du trial
      - meilleur MAE courant
      - temps ecoule
      - ETA restant base sur le temps moyen par trial

    Affiche aussi un mini bar de progression.
    """
    import optuna

    def callback(study, trial):
        n_done = len(study.trials)
        n_completed = sum(1 for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE)
        n_pruned = sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.PRUNED)
        n_failed = sum(1 for t in study.trials
                       if t.state == optuna.trial.TrialState.FAIL)

        elapsed = time.time() - start_time
        avg_per_trial = elapsed / max(1, n_done)
        remaining_trials = max(0, n_trials - n_done)
        eta = remaining_trials * avg_per_trial

        # Etat du trial courant
        if trial.state == optuna.trial.TrialState.COMPLETE:
            state_str = "OK    "
            val_str = f"{trial.value:.4f}"
        elif trial.state == optuna.trial.TrialState.PRUNED:
            state_str = "PRUNED"
            val_str = "  -   "
        else:
            state_str = "FAIL  "
            val_str = "  -   "

        # Meilleur MAE courant
        try:
            best_str = f"{study.best_value:.4f}"
        except ValueError:
            best_str = "  -   "

        # Mini bar de progression (20 chars)
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
    n_trials: int,
    n_folds_for_tuning: int = OPTUNA_N_FOLDS_FOR_TUNING,
) -> tuple[dict, dict]:
    """Lance le tuning Optuna sur les `n_folds_for_tuning` DERNIERS folds."""
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
    print(f"  Objectif : minimiser MAE residual moyenne sur les folds")
    print(f"  Affichage par trial active (callback de progression)")
    print(f"")

    all_folds = make_ts_folds(df.height)
    folds_for_tuning = all_folds[-n_folds_for_tuning:]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=OPTUNA_RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_optuna_objective(df, feature_cols, folds_for_tuning)

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
    print(f"  Best MAE residual : {study.best_value:.4f}")
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
# Entrainement d'un fold (avec drift additive + affichage detaille)
# ============================================================

def train_fold(
    fold_idx: int,
    df: pl.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    time_idx_days: np.ndarray,
    xgb_params: dict,
    cv_start_time: float | None = None,
    fit_times_history: list[float] | None = None,
) -> dict:
    """
    Entraine, predit, applique drift, log SANS et AVEC correction.

    Affichages temps reel :
      - progression fold k/N + ETA si historique disponible
      - tableau aligne MAE/RMSE pour baseline / modele raw / modele corr
      - delta avec correction
    """
    import xgboost as xgb

    # ---- Header de fold avec progression globale ----
    if fit_times_history is not None and len(fit_times_history) > 0:
        avg_fit = sum(fit_times_history) / len(fit_times_history)
        # Estimation: temps restant = (folds restants) * avg_fit
        # On ajoute une marge de 10s par fold (drift + metriques)
        folds_remaining = N_FOLDS - fold_idx
        eta = folds_remaining * (avg_fit + 10)
        eta_str = f", ETA {fmt_time(eta)}"
    else:
        eta_str = ""

    n_train_days = len(train_idx) // SAMPLES_PER_DAY
    n_test_days = len(test_idx) // SAMPLES_PER_DAY
    print_section(
        f"FOLD {fold_idx + 1}/{N_FOLDS} "
        f"(train {n_train_days}j, test {n_test_days}j{eta_str})",
        char="-"
    )

    df_train = df[train_idx]
    df_test = df[test_idx]

    # Periode test pour orientation visuelle
    ts_test_start = str(df_test["timestamp"][0])[:16]
    ts_test_end = str(df_test["timestamp"][-1])[:16]
    print(f"  Periode test : {ts_test_start} -> {ts_test_end}")

    X_train, y_train = split_X_y(df_train, feature_cols)
    X_test, y_test = split_X_y(df_test, feature_cols)

    # ---- Entrainement ----
    print(f"  Entrainement XGBoost...", flush=True)
    t0 = time.time()
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, verbose=False)
    fit_time = time.time() - t0
    print(f"    Fit time : {fit_time:.1f}s")
    if fit_times_history is not None:
        fit_times_history.append(fit_time)

    # ---- Predictions ----
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    forecast_test = df_test["forecast_load"].to_numpy()
    load_test_true = df_test["load"].to_numpy()

    # ---- Drift estimation ----
    a, b, drift_info = estimate_drift_additive(
        residual_true_train=y_train,
        residual_pred_train=pred_train,
        time_idx_train=time_idx_days[train_idx],
    )
    t_train_end = float(time_idx_days[train_idx[-1]])
    print(f"  Drift residu : a={a:+.6f}/jour, b={b:+.4f}, "
          f"R^2={drift_info.get('r2', 0):.3f}")

    # ---- Correction sur test ----
    _, correction = apply_drift_correction_additive(
        pred_residual=pred_test,
        time_idx_test=time_idx_days[test_idx],
        slope_a=a,
        time_idx_train_end=t_train_end,
    )

    # ---- Metriques ----
    metrics_raw = compute_metrics(
        residual_true=y_test, residual_pred=pred_test,
        forecast_load=forecast_test, load_true=load_test_true,
        drift_correction=None,
    )
    metrics_corr = compute_metrics(
        residual_true=y_test, residual_pred=pred_test,
        forecast_load=forecast_test, load_true=load_test_true,
        drift_correction=correction,
    )

    # ---- Affichage tableau aligne ----
    gain_mae_raw = (1 - metrics_raw["gain_mae_vs_baseline"]) * 100
    gain_rmse_raw = (1 - metrics_raw["gain_rmse_vs_baseline"]) * 100
    gain_mae_corr = (1 - metrics_corr["gain_mae_vs_baseline"]) * 100
    gain_rmse_corr = (1 - metrics_corr["gain_rmse_vs_baseline"]) * 100

    print(f"")
    print(f"  {'':22s}  {'MAE':>8s}  {'RMSE':>8s}  {'MAPE':>7s}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*7}")
    print(f"  Baseline OIKEN        : "
          f"{metrics_raw['mae_baseline']:>8.4f}  "
          f"{metrics_raw['rmse_baseline']:>8.4f}  "
          f"{'':>7s}")
    print(f"  Modele V2 (sans corr) : "
          f"{metrics_raw['mae_load']:>8.4f}  "
          f"{metrics_raw['rmse_load']:>8.4f}  "
          f"{metrics_raw['mape_load_pct']:>6.1f}%")
    print(f"  Gain raw vs baseline  : "
          f"{gain_mae_raw:>+7.1f}%  "
          f"{gain_rmse_raw:>+7.1f}%")

    delta_mae = metrics_corr['mae_load'] - metrics_raw['mae_load']
    delta_rmse = metrics_corr['rmse_load'] - metrics_raw['rmse_load']

    if abs(delta_mae) > 1e-5:
        # Correction non triviale
        print(f"  Modele V2 (avec corr) : "
              f"{metrics_corr['mae_load']:>8.4f}  "
              f"{metrics_corr['rmse_load']:>8.4f}  "
              f"{metrics_corr['mape_load_pct']:>6.1f}%")
        print(f"  Gain corr vs baseline : "
              f"{gain_mae_corr:>+7.1f}%  "
              f"{gain_rmse_corr:>+7.1f}%")
        print(f"  Delta correction      : "
              f"{delta_mae:>+8.5f}  "
              f"{delta_rmse:>+8.5f}  "
              f"({'amelioration' if delta_mae < 0 else 'degradation'})")
    else:
        print(f"  Correction additive   : delta MAE {delta_mae:+.5f} "
              f"(quasi nulle, drift inerte)")

    load_pred_raw = forecast_test + pred_test
    load_pred_corrected = forecast_test + pred_test + correction

    return {
        "fold": fold_idx + 1,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "test_start": str(df_test["timestamp"][0]),
        "test_end": str(df_test["timestamp"][-1]),
        "fit_time_s": round(fit_time, 1),
        "drift_info": drift_info,
        "drift_slope_a_per_day": float(a),
        "drift_intercept_b": float(b),
        "metrics_no_correction": metrics_raw,
        "metrics_with_correction": metrics_corr,
        "_predictions": {
            "timestamp": df_test["timestamp"].to_list(),
            "load_true": load_test_true.tolist(),
            "forecast_baseline": forecast_test.tolist(),
            "residual_true": y_test.tolist(),
            "residual_pred": pred_test.tolist(),
            "load_pred_raw": load_pred_raw.tolist(),
            "load_pred_corrected": load_pred_corrected.tolist(),
            "drift_correction_applied": correction.tolist(),
            "fold": [fold_idx + 1] * len(test_idx),
        },
    }


# ============================================================
# CV pipeline
# ============================================================

def run_cv(df: pl.DataFrame, xgb_params: dict) -> dict:
    """Orchestrateur : cree les folds, entraine chacun, agrege."""
    print_section("CROSS-VALIDATION 5 folds (expanding window)")

    feature_cols = select_feature_cols(df)
    print(f"  {len(feature_cols)} features (sur {df.shape[1]} colonnes)")
    print(f"  Cible : {TARGET_COL}")

    ts_min = df["timestamp"].min()
    time_idx_days = (
        (df["timestamp"] - ts_min).dt.total_seconds().to_numpy() / 86400.0
    )

    n_before = df.height
    df = df.filter(pl.col(TARGET_COL).is_not_null())
    n_after = df.height
    if n_after < n_before:
        print(f"  Filtre NaN cible : {n_before - n_after} lignes retirees")
        time_idx_days = (
            (df["timestamp"] - ts_min).dt.total_seconds().to_numpy()
            / 86400.0
        )

    folds = make_ts_folds(df.height)
    print(f"\n  Plan de validation :")
    for k, (tr, te) in enumerate(folds):
        ts_test_start = str(df["timestamp"][int(te[0])])[:10]
        ts_test_end = str(df["timestamp"][int(te[-1])])[:10]
        print(f"    Fold {k+1}: train {len(tr)//SAMPLES_PER_DAY:>4}j | "
              f"test [{ts_test_start} -> {ts_test_end}] "
              f"({len(te)//SAMPLES_PER_DAY}j)")

    cv_start = time.time()
    fit_times_history = []
    fold_results = []
    all_predictions = []
    for k, (tr, te) in enumerate(folds):
        res = train_fold(
            k, df, tr, te, feature_cols, time_idx_days, xgb_params,
            cv_start_time=cv_start,
            fit_times_history=fit_times_history,
        )
        all_predictions.append(res.pop("_predictions"))
        fold_results.append(res)

    cv_time = time.time() - cv_start
    print_section(f"RESULTATS AGREGES 5 folds ({fmt_time(cv_time)})")

    # Tableau aligne final
    print(f"")
    print(f"  {'Metrique':30s}  {'Sans corr':>11s}  {'Avec corr':>11s}  "
          f"{'Std (raw)':>11s}")
    print(f"  {'-'*30}  {'-'*11}  {'-'*11}  {'-'*11}")

    keys_pretty = [
        ("mae_residual", "MAE residual"),
        ("rmse_residual", "RMSE residual"),
        ("mae_load", "MAE load"),
        ("rmse_load", "RMSE load"),
        ("mae_baseline", "MAE baseline OIKEN"),
        ("rmse_baseline", "RMSE baseline OIKEN"),
        ("mape_load_pct", "MAPE load (%)"),
        ("gain_mae_vs_baseline", "Ratio MAE/baseline"),
        ("gain_rmse_vs_baseline", "Ratio RMSE/baseline"),
    ]
    for key, label in keys_pretty:
        vals_raw = [r["metrics_no_correction"][key] for r in fold_results]
        vals_corr = [r["metrics_with_correction"][key] for r in fold_results]
        mean_raw = float(np.mean(vals_raw))
        mean_corr = float(np.mean(vals_corr))
        std_raw = float(np.std(vals_raw))

        if "pct" in key:
            print(f"  {label:30s}  {mean_raw:>10.2f}%  {mean_corr:>10.2f}%  "
                  f"{std_raw:>10.2f}%")
        elif "gain" in key or "ratio" in key.lower():
            print(f"  {label:30s}  {mean_raw:>11.4f}  {mean_corr:>11.4f}  "
                  f"{std_raw:>11.4f}")
        else:
            print(f"  {label:30s}  {mean_raw:>11.4f}  {mean_corr:>11.4f}  "
                  f"{std_raw:>11.4f}")

    # Synthese gain global
    avg_gain_mae_raw = (1 - np.mean(
        [r["metrics_no_correction"]["gain_mae_vs_baseline"]
         for r in fold_results])) * 100
    avg_gain_rmse_raw = (1 - np.mean(
        [r["metrics_no_correction"]["gain_rmse_vs_baseline"]
         for r in fold_results])) * 100
    print(f"")
    print(f"  >>> Gain moyen MAE  vs baseline OIKEN : {avg_gain_mae_raw:+.1f}%")
    print(f"  >>> Gain moyen RMSE vs baseline OIKEN : {avg_gain_rmse_raw:+.1f}%")

    return {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "folds": fold_results,
        "predictions": all_predictions,
    }


# ============================================================
# Modele final
# ============================================================

def train_final_model(df: pl.DataFrame, feature_cols: list[str],
                      xgb_params: dict):
    """Entraine le modele final sur TOUT le dataset."""
    import xgboost as xgb

    print_section("MODELE FINAL (entrainement sur tout le dataset)")

    X, y = split_X_y(df, feature_cols)
    print(f"  X : {X.shape}, y : {y.shape}")

    print(f"  Entrainement XGBoost final...", flush=True)
    t0 = time.time()
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y, verbose=False)
    fit_time = time.time() - t0
    print(f"    Fit time : {fit_time:.1f}s")

    pred_residual = model.predict(X)

    ts_min = df["timestamp"].min()
    time_idx_days = (
        (df["timestamp"] - ts_min).dt.total_seconds().to_numpy() / 86400.0
    )
    a, b, drift_info = estimate_drift_additive(
        residual_true_train=y,
        residual_pred_train=pred_residual,
        time_idx_train=time_idx_days,
    )
    t_train_end_final = float(time_idx_days[-1])
    print(f"  Drift final : a={a:+.6f}/jour, b={b:+.4f} "
          f"(R^2={drift_info.get('r2', 0):.3f})")
    print(f"  t_train_end (jours depuis 1ere ligne) : "
          f"{t_train_end_final:.2f}")
    print(f"  --> en inference : ajouter clip({a:+.6f} * (t-t_end), "
          f"+/-{DRIFT_AMPLITUDE_CLIP}) au pred_residual")

    return model, {
        "slope_a_per_day": float(a),
        "intercept_b": float(b),
        "t_train_end_days": t_train_end_final,
        "amplitude_clip": DRIFT_AMPLITUDE_CLIP,
        "info": drift_info,
    }


# ============================================================
# Sauvegarde
# ============================================================

def save_predictions(all_predictions: list[dict], path: Path) -> None:
    keys = ["timestamp", "load_true", "forecast_baseline",
            "residual_true", "residual_pred",
            "load_pred_raw", "load_pred_corrected",
            "drift_correction_applied", "fold"]
    merged = {k: [] for k in keys}
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


# ============================================================
# Main
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="XGBoost J+1 OIKEN ML - V2 (Optuna + drift additif)"
    )
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip Optuna, utiliser params par defaut")
    parser.add_argument("--quick", action="store_true",
                        help="Optuna rapide (10 trials)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help=f"Nombre de trials Optuna "
                             f"(defaut: {OPTUNA_N_TRIALS_DEFAULT})")
    args = parser.parse_args(argv)

    global_start = time.time()

    print("=" * 70)
    print("MODELE XGBOOST J+1 - OIKEN ML")
    print("V2 : drift additif + Optuna + comparaison MAE/RMSE OIKEN")
    print("=" * 70)

    in_path = DATA_PROCESSED / "dataset_features.parquet"
    if not in_path.exists():
        print(f"\nERREUR: features absent : {in_path}")
        print("Lance d'abord : python -m src.features")
        return 1

    print(f"\nChargement : {in_path}")
    df = pl.read_parquet(in_path)
    print(f"  {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    print(f"  Periode : {str(df['timestamp'].min())[:16]} -> "
          f"{str(df['timestamp'].max())[:16]}")

    try:
        import xgboost as xgb
        print(f"  XGBoost version : {xgb.__version__}")
    except ImportError:
        print("\nERREUR: xgboost non installe.")
        print("Lance: uv add xgboost  (ou: pip install xgboost)")
        return 2

    feature_cols = select_feature_cols(df)

    # ---- Tuning ----
    tune_info = None
    if args.no_tune:
        print("\n[--no-tune] Skip Optuna, utilisation des params par defaut")
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
                df, feature_cols, n_trials=n_trials,
            )
        except ImportError:
            print("Fallback : params par defaut")
            xgb_params = dict(XGB_PARAMS_DEFAULT)

    # ---- CV evaluation ----
    cv_results = run_cv(df, xgb_params)

    # ---- Modele final ----
    model, drift_final = train_final_model(
        df, cv_results["feature_cols"], xgb_params
    )

    # ---- Feature importance ----
    print_section("TOP 20 FEATURES (importance gain)")
    importance = feature_importance(model, cv_results["feature_cols"])
    for name, score in importance[:20]:
        print(f"  {name:40s}: {score:>10.1f}")

    # ---- Sauvegardes ----
    DATA_MODELS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_REPORTS.mkdir(parents=True, exist_ok=True)

    model_path = DATA_MODELS / "xgboost_j1_final.json"
    model.save_model(str(model_path))

    pred_path = DATA_PROCESSED / "predictions_xgboost_cv.parquet"
    save_predictions(cv_results["predictions"], pred_path)

    report = {
        "model": "XGBoost",
        "version": "v2_optuna_additive",
        "xgb_params_used": xgb_params,
        "n_features": cv_results["n_features"],
        "feature_cols": cv_results["feature_cols"],
        "cv_strategy": {
            "n_folds": N_FOLDS,
            "test_size_days": TEST_SIZE_DAYS,
            "min_train_days": MIN_TRAIN_DAYS,
            "scheme": "expanding_window",
        },
        "tuning": tune_info,
        "folds": cv_results["folds"],
        "drift_correction": {
            "type": "additive",
            "form": ("pred_residual + clip(slope_a*(t - t_train_end),"
                     " +/- amplitude_clip)"),
            "tail_days_for_calibration": DRIFT_TRAIN_TAIL_DAYS,
            "amplitude_clip": DRIFT_AMPLITUDE_CLIP,
            "final_model_drift": drift_final,
        },
        "feature_importance_top30": importance[:30],
    }
    report_path = DATA_REPORTS / "model_xgboost_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    total_time = time.time() - global_start
    print_section(f"MODELE XGBOOST V2 TERMINE en {fmt_time(total_time)}")
    print(f"  Modele final  -> {model_path}")
    print(f"  Predictions   -> {pred_path}")
    print(f"  Rapport       -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())