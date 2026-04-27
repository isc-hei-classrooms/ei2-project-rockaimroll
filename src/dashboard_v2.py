"""
Dashboard HTML statique pour le modele V2 (XGBoost + Optuna + drift additif).

Genere un fichier HTML autonome (Plotly.js depuis CDN) qui permet
d'explorer interactivement les predictions du modele :
  - Synthese : KPIs par fold, distribution des residus
  - Mois     : heatmap calendaire MAE par jour, profil moyen
  - Jour     : 4 courbes (reel, baseline, modele, modele corrige) + erreurs
  - Diagnostics : MAE par heure, scatter pred vs reel, residus par DoW
  - Drift    : pente / intercept / R2 par fold, impact correction
  - Features : top 30 importance (gain) groupees par categorie

Sources:
  data/processed/predictions_xgboost_cv.parquet
  data/reports/model_xgboost_report.json

Sortie:
  data/reports/dashboard_v2.html  (autonome, ouvre dans navigateur)

Utilisation:
  python -m src.dashboard_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

# Le module est utilisable hors-package : si import echoue, fallback paths
try:
    from src.config import PROJECT_ROOT, DATA_PROCESSED
    DATA_REPORTS = PROJECT_ROOT / "data" / "reports"
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_REPORTS = PROJECT_ROOT / "data" / "reports"


# ============================================================
# Couleurs et styles (CSS variables)
# ============================================================

PALETTE = {
    "real": "#0F172A",         # noir-bleu : verite terrain
    "baseline": "#1E40AF",     # bleu nuit : baseline OIKEN
    "model": "#047857",        # vert sapin : XGBoost
    "corrected": "#B45309",    # ambre : modele + correction
    "error": "#991B1B",        # rouge brique
    "muted": "#64748B",        # gris ardoise
    "accent": "#0891B2",       # cyan : selection
}


# ============================================================
# Chargement et agregations
# ============================================================

def load_data(predictions_path: Path, report_path: Path) -> dict:
    """Charge predictions parquet + rapport JSON."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions absentes : {predictions_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Rapport absent : {report_path}")

    df = pl.read_parquet(predictions_path)
    print(f"  Predictions  : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    print(f"  Rapport      : v{report.get('version', '?')}")

    # Convertir timestamp en datetime UTC si pas deja
    if df["timestamp"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="UTC")
        )
    elif df["timestamp"].dtype == pl.Datetime and df["timestamp"].dtype.time_zone is None:
        df = df.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC")
        )

    # Convertir en heure locale Zurich pour l'affichage
    df = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("Europe/Zurich").alias("ts_local")
    )

    return {"df": df, "report": report}


def compute_daily_aggregates(df: pl.DataFrame) -> pl.DataFrame:
    """MAE / RMSE / MAPE par jour pour le modele et la baseline."""
    daily = (
        df.with_columns([
            pl.col("ts_local").dt.date().alias("date"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
            (pl.col("load_true") - pl.col("load_pred_corrected")).abs()
                .alias("err_corr"),
        ])
        .group_by("date", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
            pl.col("err_corr").mean().alias("mae_corr"),
            pl.col("load_true").mean().alias("load_mean"),
            pl.col("load_true").max().alias("load_max"),
            pl.col("load_true").min().alias("load_min"),
            pl.len().alias("n_points"),
        ])
        .sort("date")
    )
    return daily


def compute_monthly_aggregates(df: pl.DataFrame) -> pl.DataFrame:
    """MAE par mois (annee-mois)."""
    monthly = (
        df.with_columns([
            pl.col("ts_local").dt.strftime("%Y-%m").alias("month"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
            (pl.col("load_true") - pl.col("load_pred_corrected")).abs()
                .alias("err_corr"),
        ])
        .group_by("month", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
            pl.col("err_corr").mean().alias("mae_corr"),
            ((pl.col("err_model") - pl.col("err_baseline")) /
             pl.col("err_baseline") * 100).mean().alias("gain_pct"),
            pl.len().alias("n_points"),
        ])
        .sort("month")
    )
    return monthly


def compute_hourly_diagnostics(df: pl.DataFrame) -> pl.DataFrame:
    """MAE par heure (0-23) toutes folds confondues."""
    hourly = (
        df.with_columns([
            pl.col("ts_local").dt.hour().alias("hour"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
        ])
        .group_by("hour", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
        ])
        .sort("hour", "fold")
    )
    return hourly


def compute_dow_diagnostics(df: pl.DataFrame) -> pl.DataFrame:
    """MAE par jour de la semaine (0=lundi)."""
    dow = (
        df.with_columns([
            pl.col("ts_local").dt.weekday().alias("dow"),
            (pl.col("load_true") - pl.col("forecast_baseline")).abs()
                .alias("err_baseline"),
            (pl.col("load_true") - pl.col("load_pred_raw")).abs()
                .alias("err_model"),
        ])
        .group_by("dow", "fold")
        .agg([
            pl.col("err_baseline").mean().alias("mae_baseline"),
            pl.col("err_model").mean().alias("mae_model"),
        ])
        .sort("dow", "fold")
    )
    return dow


def categorize_feature(name: str) -> str:
    """Assigne une categorie a une feature pour grouper le top 30."""
    if name.startswith(("cal_", "cyc_")):
        return "Calendaire"
    if name.startswith("sun_"):
        return "Solaire"
    if name.startswith("lag_"):
        return "Lags"
    if name.startswith("roll_"):
        return "Rolling stats"
    if name.startswith("temp_"):
        return "Temperature"
    if name.startswith("pv_"):
        return "PV / rendement"
    if name.startswith("iqr_"):
        return "Incertitude COSMO-E"
    if name.startswith("delta_"):
        return "Delta meteo"
    if name.startswith("inter_"):
        return "Interactions"
    if name.startswith("pred_"):
        return "Prevision meteo"
    if name == "forecast_load":
        return "Baseline OIKEN"
    return "Autre"


# ============================================================
# Serialisation JSON pour embedding dans le HTML
# ============================================================

def df_to_compact_json(df: pl.DataFrame) -> dict:
    """Serialise un Polars DataFrame en {col: [values]}."""
    out = {}
    for col in df.columns:
        s = df[col]
        dt = s.dtype
        if dt == pl.Date:
            out[col] = s.dt.strftime("%Y-%m-%d").to_list()
        elif dt == pl.Datetime:
            out[col] = s.dt.strftime("%Y-%m-%dT%H:%M").to_list()
        elif dt in (pl.Float32, pl.Float64):
            out[col] = [None if v is None else round(float(v), 5)
                        for v in s.to_list()]
        else:
            out[col] = s.to_list()
    return out


def build_dataset(data: dict) -> dict:
    """Construit l'objet JSON complet a embarquer dans le HTML."""
    df = data["df"]
    report = data["report"]

    daily = compute_daily_aggregates(df)
    monthly = compute_monthly_aggregates(df)
    hourly = compute_hourly_diagnostics(df)
    dow = compute_dow_diagnostics(df)

    # Series complete par jour (96 pts) : on indexe par date pour acces O(1)
    df_local = df.with_columns(
        pl.col("ts_local").dt.strftime("%Y-%m-%d").alias("date_key"),
        pl.col("ts_local").dt.strftime("%H:%M").alias("hhmm"),
    )
    by_day = {}
    for date_key, sub in df_local.group_by("date_key", maintain_order=True):
        sub_sorted = sub.sort("ts_local")
        date_str = date_key[0]
        by_day[date_str] = {
            "hhmm": sub_sorted["hhmm"].to_list(),
            "load_true": [round(float(v), 4)
                          for v in sub_sorted["load_true"].to_list()],
            "forecast_baseline": [round(float(v), 4)
                                  for v in sub_sorted["forecast_baseline"]
                                  .to_list()],
            "load_pred_raw": [round(float(v), 4)
                              for v in sub_sorted["load_pred_raw"].to_list()],
            "load_pred_corrected": [round(float(v), 4)
                                    for v in sub_sorted["load_pred_corrected"]
                                    .to_list()],
            "fold": int(sub_sorted["fold"][0]),
            "n_points": sub_sorted.height,
        }

    # Categoriser feature importance
    fi_top30 = report.get("feature_importance_top30", [])
    fi_with_cat = [
        {"name": name, "score": float(score),
         "category": categorize_feature(name)}
        for name, score in fi_top30
    ]

    payload = {
        "meta": {
            "model": report.get("model", "XGBoost"),
            "version": report.get("version", "v2"),
            "n_features": report.get("n_features", 0),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "n_days": len(by_day),
            "n_predictions": int(df.height),
            "ts_min": df["ts_local"].min().strftime("%Y-%m-%d %H:%M"),
            "ts_max": df["ts_local"].max().strftime("%Y-%m-%d %H:%M"),
        },
        "tuning": report.get("tuning"),
        "xgb_params": report.get("xgb_params_used", {}),
        "cv_strategy": report.get("cv_strategy", {}),
        "drift_correction": report.get("drift_correction", {}),
        "folds": report.get("folds", []),
        "daily": df_to_compact_json(daily),
        "monthly": df_to_compact_json(monthly),
        "hourly": df_to_compact_json(hourly),
        "dow": df_to_compact_json(dow),
        "by_day": by_day,
        "feature_importance": fi_with_cat,
    }
    return payload


# ============================================================
# Template HTML (CSS + JS embarques)
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OIKEN ML - Dashboard V2</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root {
  --bg: #FAFAF7;
  --bg-elev: #FFFFFF;
  --bg-sunken: #F1F1EC;
  --border: #E5E5DF;
  --border-strong: #C5C5BC;
  --text: #1A1A1A;
  --text-muted: #5C5C58;
  --text-faint: #8A8A85;
  --color-real: __COLOR_REAL__;
  --color-baseline: __COLOR_BASELINE__;
  --color-model: __COLOR_MODEL__;
  --color-corrected: __COLOR_CORRECTED__;
  --color-error: __COLOR_ERROR__;
  --color-accent: __COLOR_ACCENT__;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
  --radius: 6px;
}
[data-theme="dark"] {
  --bg: #0F1115;
  --bg-elev: #181B22;
  --bg-sunken: #0A0C10;
  --border: #2A2F3A;
  --border-strong: #3D4350;
  --text: #E8E8E5;
  --text-muted: #A0A4AB;
  --text-faint: #6B6F78;
  --color-real: #F5F5F2;
  --color-baseline: #60A5FA;
  --color-model: #34D399;
  --color-corrected: #FBBF24;
  --color-error: #F87171;
  --color-accent: #22D3EE;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Sans', -apple-system, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.num, .mono {
  font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
h1, h2, h3 {
  font-family: 'Fraunces', Georgia, serif;
  font-weight: 600;
  letter-spacing: -0.01em;
}

/* Header */
.header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg-elev);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 24px;
}
.brand {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.brand-title {
  font-family: 'Fraunces', serif;
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.brand-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-faint);
  font-variant-numeric: tabular-nums;
}
.header-stats {
  margin-left: auto;
  display: flex;
  gap: 18px;
  font-size: 12px;
  color: var(--text-muted);
}
.header-stat strong {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text);
  margin-right: 4px;
}
.theme-toggle {
  background: transparent;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius);
  color: var(--text);
  padding: 6px 12px;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.theme-toggle:hover { background: var(--bg-sunken); }

/* Tabs */
.tabs {
  display: flex;
  gap: 0;
  background: var(--bg-elev);
  border-bottom: 1px solid var(--border);
  padding: 0 24px;
  overflow-x: auto;
}
.tab {
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--text-muted);
  padding: 12px 18px;
  font-family: inherit;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: color 0.15s, border-color 0.15s;
}
.tab:hover { color: var(--text); }
.tab.active {
  color: var(--text);
  border-bottom-color: var(--color-accent);
}
.tab-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-faint);
  margin-right: 6px;
}

/* Content */
.content {
  padding: 24px;
  max-width: 1600px;
  margin: 0 auto;
}
.panel {
  display: none;
}
.panel.active { display: block; }
.panel-title {
  font-size: 22px;
  margin-bottom: 4px;
}
.panel-subtitle {
  color: var(--text-muted);
  font-size: 13px;
  margin-bottom: 20px;
}

/* Cards */
.cards {
  display: grid;
  gap: 16px;
}
.cards.cols-2 { grid-template-columns: repeat(2, 1fr); }
.cards.cols-3 { grid-template-columns: repeat(3, 1fr); }
.cards.cols-4 { grid-template-columns: repeat(4, 1fr); }
.cards.cols-5 { grid-template-columns: repeat(5, 1fr); }
@media (max-width: 1100px) {
  .cards.cols-3, .cards.cols-4, .cards.cols-5 {
    grid-template-columns: repeat(2, 1fr);
  }
}
.card {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}
.card-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-faint);
  margin-bottom: 8px;
}
.kpi-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 28px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  margin-bottom: 4px;
}
.kpi-sub {
  font-size: 12px;
  color: var(--text-muted);
}
.kpi-sub .delta-pos { color: var(--color-error); font-weight: 500; }
.kpi-sub .delta-neg { color: var(--color-model); font-weight: 500; }

/* Plot containers */
.plot {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  margin-bottom: 16px;
}
.plot-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin: 4px 4px 8px;
}
.plot-meta {
  font-size: 11px;
  color: var(--text-faint);
  font-family: 'JetBrains Mono', monospace;
  margin: 0 4px 8px;
}

/* Selectors */
.controls {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  margin-bottom: 16px;
}
.controls-label {
  font-size: 12px;
  color: var(--text-muted);
  font-weight: 500;
}
.control {
  background: var(--bg);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  padding: 6px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  cursor: pointer;
}
.control:focus { outline: 2px solid var(--color-accent); outline-offset: -1px; }
.control-group { display: flex; gap: 4px; align-items: center; }
.btn {
  background: var(--bg);
  border: 1px solid var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  padding: 5px 10px;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.btn:hover { background: var(--bg-sunken); }
.btn.active {
  background: var(--color-accent);
  color: white;
  border-color: var(--color-accent);
}

/* Tables */
table.tbl {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
table.tbl th, table.tbl td {
  padding: 8px 10px;
  text-align: right;
  border-bottom: 1px solid var(--border);
}
table.tbl th {
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.06em;
  text-align: right;
}
table.tbl th:first-child, table.tbl td:first-child { text-align: left; }
table.tbl td.num { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
table.tbl tr:hover td { background: var(--bg-sunken); }
.tbl-wrap {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 12px;
}

/* Calendar heatmap */
.calendar-wrap { display: grid; gap: 16px; }
.calendar-month {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px;
}
.cal-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}
.cal-month-name {
  font-family: 'Fraunces', serif;
  font-size: 16px;
  font-weight: 600;
}
.cal-month-mae {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text-muted);
}
.cal-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 3px;
}
.cal-dow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  color: var(--text-faint);
  text-align: center;
  padding: 3px 0;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.cal-day {
  aspect-ratio: 1;
  border-radius: 3px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: border-color 0.1s, transform 0.1s;
  position: relative;
}
.cal-day:hover {
  border-color: var(--color-accent);
  transform: scale(1.06);
  z-index: 2;
}
.cal-day.empty {
  background: transparent;
  cursor: default;
  pointer-events: none;
}
.cal-day-num {
  font-size: 11px;
  font-weight: 500;
}
.cal-day-mae {
  font-size: 9px;
  opacity: 0.7;
}

/* Day view layout */
.day-grid {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 16px;
}
@media (max-width: 1100px) {
  .day-grid { grid-template-columns: 1fr; }
}
.day-summary {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.day-summary-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.day-summary-row:last-child { border-bottom: none; padding-bottom: 0; }
.day-summary-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }
.day-summary-value { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 500; }

/* Legend */
.legend-line {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 11px;
  color: var(--text-muted);
  padding: 8px 12px;
  margin-bottom: 12px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.legend-swatch {
  width: 16px;
  height: 3px;
  border-radius: 2px;
}

/* Footer info */
.footer {
  padding: 24px;
  text-align: center;
  font-size: 11px;
  color: var(--text-faint);
  border-top: 1px solid var(--border);
  margin-top: 32px;
  font-family: 'JetBrains Mono', monospace;
}
</style>
</head>
<body data-theme="light">

<div class="header">
  <div class="brand">
    <div class="brand-title">OIKEN ML, dashboard V2</div>
    <div class="brand-meta" id="brand-meta">XGBoost + Optuna + drift additif</div>
  </div>
  <div class="header-stats">
    <div class="header-stat"><strong id="hs-days">-</strong> jours</div>
    <div class="header-stat"><strong id="hs-pts">-</strong> points</div>
    <div class="header-stat"><strong id="hs-features">-</strong> features</div>
  </div>
  <button class="theme-toggle" id="theme-toggle">Theme sombre</button>
</div>

<div class="tabs" id="tabs">
  <button class="tab active" data-panel="synthese"><span class="tab-num">01</span>Synthese</button>
  <button class="tab" data-panel="mois"><span class="tab-num">02</span>Mois</button>
  <button class="tab" data-panel="jour"><span class="tab-num">03</span>Jour</button>
  <button class="tab" data-panel="diagnostics"><span class="tab-num">04</span>Diagnostics</button>
  <button class="tab" data-panel="drift"><span class="tab-num">05</span>Drift</button>
  <button class="tab" data-panel="features"><span class="tab-num">06</span>Features</button>
</div>

<div class="content">

<!-- ============================================================ -->
<!-- PANEL SYNTHESE -->
<!-- ============================================================ -->
<div class="panel active" id="panel-synthese">
  <h1 class="panel-title">Synthese</h1>
  <p class="panel-subtitle" id="synthese-sub">Performance globale du modele V2 sur 5 folds (expanding window).</p>

  <div class="cards cols-4" id="kpi-cards"></div>

  <div style="height:16px;"></div>

  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">MAE par fold (load reconstruit)</div><div id="plot-mae-folds" style="height:340px;"></div></div>
    <div class="plot"><div class="plot-title">Gain MAE vs baseline OIKEN par fold</div><div id="plot-gain-folds" style="height:340px;"></div></div>
  </div>

  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Distribution des erreurs (modele vs baseline)</div><div id="plot-error-dist" style="height:380px;"></div></div>
    <div class="plot"><div class="plot-title">Tableau detaille par fold</div><div class="tbl-wrap"><table class="tbl" id="tbl-folds"></table></div></div>
  </div>
</div>

<!-- ============================================================ -->
<!-- PANEL MOIS -->
<!-- ============================================================ -->
<div class="panel" id="panel-mois">
  <h1 class="panel-title">Vue mensuelle</h1>
  <p class="panel-subtitle">Carte calendaire MAE par jour. Cliquer un jour pour basculer en vue jour.</p>

  <div class="controls">
    <span class="controls-label">Metrique affichee</span>
    <div class="control-group">
      <button class="btn active" data-metric="mae_model">Modele</button>
      <button class="btn" data-metric="mae_baseline">Baseline</button>
      <button class="btn" data-metric="gain">Gain (%)</button>
    </div>
  </div>

  <div class="calendar-wrap" id="calendar-wrap"></div>

  <div style="height:16px;"></div>

  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Profil moyen 24h par mois</div><div id="plot-monthly-profile" style="height:380px;"></div></div>
    <div class="plot"><div class="plot-title">MAE moyenne par mois (modele vs baseline)</div><div id="plot-monthly-mae" style="height:380px;"></div></div>
  </div>
</div>

<!-- ============================================================ -->
<!-- PANEL JOUR -->
<!-- ============================================================ -->
<div class="panel" id="panel-jour">
  <h1 class="panel-title">Vue journaliere</h1>
  <p class="panel-subtitle">96 pas de 15 min. Comparaison reel / baseline OIKEN / modele / modele corrige.</p>

  <div class="controls">
    <span class="controls-label">Date</span>
    <select class="control" id="day-select"></select>
    <button class="btn" id="day-prev">< prec</button>
    <button class="btn" id="day-next">suiv ></button>
    <span class="controls-label" style="margin-left:24px;">Traces</span>
    <div class="control-group">
      <button class="btn active" data-trace="real">Reel</button>
      <button class="btn active" data-trace="baseline">Baseline</button>
      <button class="btn active" data-trace="model">Modele</button>
      <button class="btn" data-trace="corrected">Corrige</button>
    </div>
  </div>

  <div class="day-grid">
    <div>
      <div class="plot"><div class="plot-title" id="day-plot-title">Charge predite vs reelle</div><div id="plot-day-main" style="height:420px;"></div></div>
      <div class="plot"><div class="plot-title">Erreurs absolues (residus)</div><div id="plot-day-errors" style="height:220px;"></div></div>
    </div>
    <div class="day-summary" id="day-summary"></div>
  </div>
</div>

<!-- ============================================================ -->
<!-- PANEL DIAGNOSTICS -->
<!-- ============================================================ -->
<div class="panel" id="panel-diagnostics">
  <h1 class="panel-title">Diagnostics</h1>
  <p class="panel-subtitle">Decomposition de l'erreur par heure et jour de semaine, scatter pred vs reel.</p>

  <div class="controls">
    <span class="controls-label">Fold</span>
    <select class="control" id="diag-fold-select">
      <option value="all">Tous (agrege)</option>
      <option value="1">Fold 1</option>
      <option value="2">Fold 2</option>
      <option value="3">Fold 3</option>
      <option value="4">Fold 4</option>
      <option value="5">Fold 5</option>
    </select>
  </div>

  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">MAE par heure de la journee</div><div id="plot-diag-hour" style="height:360px;"></div></div>
    <div class="plot"><div class="plot-title">MAE par jour de semaine</div><div id="plot-diag-dow" style="height:360px;"></div></div>
  </div>

  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Scatter : prediction vs verite (modele)</div><div id="plot-diag-scatter" style="height:420px;"></div></div>
    <div class="plot"><div class="plot-title">Distribution des residus signes (load_pred - load_true)</div><div id="plot-diag-residual-hist" style="height:420px;"></div></div>
  </div>
</div>

<!-- ============================================================ -->
<!-- PANEL DRIFT -->
<!-- ============================================================ -->
<div class="panel" id="panel-drift">
  <h1 class="panel-title">Correction additive (drift PV)</h1>
  <p class="panel-subtitle">Pente, intercept et impact effectif de la correction par fold.</p>

  <div class="cards cols-3">
    <div class="card">
      <div class="card-title">Forme</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size:12px;" id="drift-form">-</div>
    </div>
    <div class="card">
      <div class="card-title">Tail days calibration</div>
      <div class="kpi-value" id="drift-tail">-</div>
      <div class="kpi-sub">jours en fin de train</div>
    </div>
    <div class="card">
      <div class="card-title">Amplitude clip</div>
      <div class="kpi-value" id="drift-clip">-</div>
      <div class="kpi-sub">borne max d'extrapolation</div>
    </div>
  </div>

  <div style="height:16px;"></div>

  <div class="plot">
    <div class="plot-title">Tableau drift par fold</div>
    <div class="tbl-wrap"><table class="tbl" id="tbl-drift"></table></div>
  </div>

  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Pente estimee par fold</div><div id="plot-drift-slope" style="height:340px;"></div></div>
    <div class="plot"><div class="plot-title">R2 du fit lineaire par fold</div><div id="plot-drift-r2" style="height:340px;"></div></div>
  </div>
</div>

<!-- ============================================================ -->
<!-- PANEL FEATURES -->
<!-- ============================================================ -->
<div class="panel" id="panel-features">
  <h1 class="panel-title">Importance des features</h1>
  <p class="panel-subtitle">Top 30 du modele final (importance gain, XGBoost native).</p>

  <div class="cards cols-2">
    <div class="plot"><div class="plot-title">Top 30 features (gain)</div><div id="plot-fi-bar" style="height:680px;"></div></div>
    <div class="plot"><div class="plot-title">Importance agregee par categorie</div><div id="plot-fi-category" style="height:340px;"></div>
      <div class="tbl-wrap" style="margin-top:12px;"><table class="tbl" id="tbl-fi-category"></table></div>
    </div>
  </div>
</div>

</div><!-- /content -->

<div class="footer" id="footer">
  Generated <span id="footer-time"></span> | OIKEN ML V2 | XGBoost + Optuna + drift additif
</div>

<script id="data-payload" type="application/json">__DATA_JSON__</script>

<script>
"use strict";

const DATA = JSON.parse(document.getElementById("data-payload").textContent);
const FOLD_COLORS = ["#0F766E", "#1E40AF", "#7C3AED", "#B45309", "#BE123C"];
const PLOTLY_FONT = { family: "'IBM Plex Sans', sans-serif", size: 11 };

// ============ Theme ============
function getCssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function plotlyLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: Object.assign({}, PLOTLY_FONT, { color: getCssVar("--text") }),
    margin: { l: 50, r: 18, t: 28, b: 38 },
    xaxis: { gridcolor: getCssVar("--border"), zerolinecolor: getCssVar("--border-strong"), tickfont: { size: 10 } },
    yaxis: { gridcolor: getCssVar("--border"), zerolinecolor: getCssVar("--border-strong"), tickfont: { size: 10 } },
    legend: { orientation: "h", y: -0.15, x: 0, xanchor: "left", font: { size: 11 } },
    hovermode: "x unified",
  }, extra);
}
const PLOTLY_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };

const themeToggle = document.getElementById("theme-toggle");
themeToggle.addEventListener("click", () => {
  const cur = document.body.getAttribute("data-theme");
  const next = cur === "light" ? "dark" : "light";
  document.body.setAttribute("data-theme", next);
  themeToggle.textContent = next === "light" ? "Theme sombre" : "Theme clair";
  // Re-render tous les plots actifs
  setTimeout(renderAll, 50);
});

// ============ Tabs ============
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.dataset.panel).classList.add("active");
    setTimeout(renderAll, 30);
  });
});

// ============ Header info ============
document.getElementById("hs-days").textContent = DATA.meta.n_days.toLocaleString("fr-CH");
document.getElementById("hs-pts").textContent = DATA.meta.n_predictions.toLocaleString("fr-CH");
document.getElementById("hs-features").textContent = DATA.meta.n_features;
document.getElementById("brand-meta").textContent = `${DATA.meta.model} ${DATA.meta.version} | ${DATA.meta.ts_min} -> ${DATA.meta.ts_max}`;
document.getElementById("footer-time").textContent = DATA.meta.generated_at;

// ============ Helpers ============
function fmt(n, dec = 4) {
  if (n === null || n === undefined || isNaN(n)) return "-";
  return Number(n).toFixed(dec);
}
function fmtPct(n, dec = 1) {
  if (n === null || n === undefined || isNaN(n)) return "-";
  return Number(n).toFixed(dec) + "%";
}
function meanArr(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
function stdArr(arr) {
  const m = meanArr(arr);
  return Math.sqrt(meanArr(arr.map(v => (v - m) ** 2)));
}

// ============ SYNTHESE ============
function renderSynthese() {
  // KPI cards
  const folds = DATA.folds;
  const allMaeModel = folds.map(f => f.metrics_no_correction.mae_load);
  const allMaeBase = folds.map(f => f.metrics_no_correction.mae_baseline);
  const allRmseModel = folds.map(f => f.metrics_no_correction.rmse_load);
  const allRmseBase = folds.map(f => f.metrics_no_correction.rmse_baseline);
  const allMape = folds.map(f => f.metrics_no_correction.mape_load_pct);
  const meanGain = meanArr(folds.map(f => f.metrics_no_correction.gain_mae_vs_baseline));
  const improve = (1 - meanGain) * 100;

  const cards = [
    { title: "MAE moyenne (modele)", value: fmt(meanArr(allMaeModel)),
      sub: `vs baseline ${fmt(meanArr(allMaeBase))}` },
    { title: "RMSE moyenne (modele)", value: fmt(meanArr(allRmseModel)),
      sub: `vs baseline ${fmt(meanArr(allRmseBase))}` },
    { title: "MAPE moyenne", value: fmt(meanArr(allMape), 1) + "%",
      sub: "tous folds confondus" },
    { title: "Amelioration vs baseline", value: improve.toFixed(1) + "%",
      sub: "MAE relative gagnee", improve },
  ];
  document.getElementById("kpi-cards").innerHTML = cards.map(c => {
    const cls = c.improve !== undefined && c.improve > 0 ? "delta-neg" : "";
    return `<div class="card">
      <div class="card-title">${c.title}</div>
      <div class="kpi-value">${c.value}</div>
      <div class="kpi-sub ${cls}">${c.sub}</div>
    </div>`;
  }).join("");

  // Plot MAE par fold
  Plotly.newPlot("plot-mae-folds", [
    { x: folds.map(f => "Fold " + f.fold), y: allMaeBase, type: "bar",
      name: "Baseline OIKEN", marker: { color: getCssVar("--color-baseline") } },
    { x: folds.map(f => "Fold " + f.fold), y: allMaeModel, type: "bar",
      name: "Modele V2", marker: { color: getCssVar("--color-model") } },
  ], plotlyLayout({
    barmode: "group",
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "MAE (load standardise)", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  // Plot gain par fold
  const gains = folds.map(f => (1 - f.metrics_no_correction.gain_mae_vs_baseline) * 100);
  Plotly.newPlot("plot-gain-folds", [
    { x: folds.map(f => "Fold " + f.fold), y: gains, type: "bar",
      marker: { color: gains.map(g => g > 20 ? getCssVar("--color-model") : g > 0 ? getCssVar("--color-corrected") : getCssVar("--color-error")) },
      text: gains.map(g => g.toFixed(1) + "%"), textposition: "outside" },
  ], plotlyLayout({
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Gain MAE relatif (%)", font: { size: 11 } }, zeroline: true }),
    showlegend: false,
  }), PLOTLY_CONFIG);

  // Distribution erreurs
  const allErrModel = [], allErrBase = [];
  Object.values(DATA.by_day).forEach(day => {
    for (let i = 0; i < day.load_true.length; i++) {
      allErrModel.push(day.load_pred_raw[i] - day.load_true[i]);
      allErrBase.push(day.forecast_baseline[i] - day.load_true[i]);
    }
  });
  Plotly.newPlot("plot-error-dist", [
    { x: allErrBase, type: "histogram", name: "Baseline OIKEN",
      marker: { color: getCssVar("--color-baseline") }, opacity: 0.55,
      xbins: { size: 0.025 } },
    { x: allErrModel, type: "histogram", name: "Modele V2",
      marker: { color: getCssVar("--color-model") }, opacity: 0.7,
      xbins: { size: 0.025 } },
  ], plotlyLayout({
    barmode: "overlay",
    xaxis: Object.assign(plotlyLayout().xaxis, { title: { text: "Erreur signee (pred - true)", font: { size: 11 } }, range: [-1.5, 1.5] }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Frequence", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  // Tableau folds
  const rows = folds.map(f => {
    const m = f.metrics_no_correction;
    const gain = (1 - m.gain_mae_vs_baseline) * 100;
    return `<tr>
      <td>Fold ${f.fold}</td>
      <td class="num">${f.test_start.slice(0,10)}</td>
      <td class="num">${f.test_end.slice(0,10)}</td>
      <td class="num">${fmt(m.mae_baseline)}</td>
      <td class="num">${fmt(m.mae_load)}</td>
      <td class="num">${fmt(m.rmse_load)}</td>
      <td class="num">${fmt(m.mape_load_pct, 1)}%</td>
      <td class="num" style="color:${gain > 20 ? 'var(--color-model)' : gain > 0 ? 'var(--color-corrected)' : 'var(--color-error)'}">${gain.toFixed(1)}%</td>
    </tr>`;
  }).join("");
  document.getElementById("tbl-folds").innerHTML = `
    <thead><tr>
      <th>Fold</th><th>Debut</th><th>Fin</th>
      <th>MAE base</th><th>MAE modele</th><th>RMSE</th><th>MAPE</th><th>Gain</th>
    </tr></thead><tbody>${rows}</tbody>`;
}

// ============ MOIS ============
let currentMonthMetric = "mae_model";
function getDailyMap() {
  // Construit Map<dateStr, {mae_model, mae_baseline, mae_corr, gain}>
  const m = new Map();
  const d = DATA.daily;
  for (let i = 0; i < d.date.length; i++) {
    m.set(d.date[i], {
      mae_model: d.mae_model[i],
      mae_baseline: d.mae_baseline[i],
      mae_corr: d.mae_corr[i],
      gain: ((d.mae_baseline[i] - d.mae_model[i]) / d.mae_baseline[i]) * 100,
      fold: d.fold[i],
      load_mean: d.load_mean[i],
    });
  }
  return m;
}
function maeColor(value, min, max, theme) {
  // Green at min (low MAE = good), red at max (high MAE = bad)
  if (value === null || value === undefined || isNaN(value)) return "transparent";
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  // Interpolation simple : vert -> jaune -> rouge
  const r = Math.round(255 * t);
  const g = Math.round(180 * (1 - Math.abs(t - 0.5) * 2 + 0.5));
  const b = Math.round(80 * (1 - t));
  const alpha = theme === "dark" ? 0.65 : 0.55;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
function gainColor(value, theme) {
  if (value === null || value === undefined || isNaN(value)) return "transparent";
  const alpha = theme === "dark" ? 0.7 : 0.55;
  if (value > 30) return `rgba(4, 120, 87, ${alpha})`;
  if (value > 15) return `rgba(132, 204, 22, ${alpha})`;
  if (value > 0) return `rgba(245, 158, 11, ${alpha})`;
  return `rgba(220, 38, 38, ${alpha})`;
}
function renderCalendar() {
  const dailyMap = getDailyMap();
  const dates = Array.from(dailyMap.keys()).sort();
  if (!dates.length) return;

  // Determiner range pour echelle de couleurs
  const allMae = Array.from(dailyMap.values()).map(v => v[currentMonthMetric === "gain" ? "gain" : currentMonthMetric]).filter(v => v !== null && v !== undefined && !isNaN(v));
  const minVal = Math.min(...allMae);
  const maxVal = Math.max(...allMae);
  const theme = document.body.getAttribute("data-theme");

  // Grouper par mois
  const byMonth = new Map();
  dates.forEach(d => {
    const k = d.slice(0, 7);
    if (!byMonth.has(k)) byMonth.set(k, []);
    byMonth.get(k).push(d);
  });

  const monthsFr = ["", "Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin",
                    "Juillet", "Aout", "Septembre", "Octobre", "Novembre", "Decembre"];
  const wrap = document.getElementById("calendar-wrap");
  wrap.style.gridTemplateColumns = `repeat(auto-fit, minmax(280px, 1fr))`;
  let html = "";
  byMonth.forEach((days, ymKey) => {
    const [year, month] = ymKey.split("-").map(Number);
    const monthName = `${monthsFr[month]} ${year}`;
    const monthMaes = days.map(d => dailyMap.get(d)[currentMonthMetric]).filter(v => !isNaN(v));
    const monthMean = meanArr(monthMaes);

    // Calculer offset du 1er du mois (lundi=0)
    const first = new Date(year, month - 1, 1);
    const firstDow = (first.getDay() + 6) % 7; // 0=lundi
    const daysInMonth = new Date(year, month, 0).getDate();

    let cells = "";
    for (let i = 0; i < firstDow; i++) {
      cells += `<div class="cal-day empty"></div>`;
    }
    for (let day = 1; day <= daysInMonth; day++) {
      const dStr = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
      const data = dailyMap.get(dStr);
      if (!data) {
        cells += `<div class="cal-day empty"><span class="cal-day-num" style="opacity:0.3">${day}</span></div>`;
      } else {
        const val = currentMonthMetric === "gain" ? data.gain : data[currentMonthMetric];
        const bg = currentMonthMetric === "gain" ? gainColor(val, theme) : maeColor(val, minVal, maxVal, theme);
        const display = currentMonthMetric === "gain" ? val.toFixed(0) + "%" : val.toFixed(3);
        const tooltip = `${dStr}\nMAE modele: ${data.mae_model.toFixed(4)}\nMAE baseline: ${data.mae_baseline.toFixed(4)}\nGain: ${data.gain.toFixed(1)}%\nFold ${data.fold}`;
        cells += `<div class="cal-day" style="background:${bg};" data-date="${dStr}" title="${tooltip}">
          <span class="cal-day-num">${day}</span>
          <span class="cal-day-mae">${display}</span>
        </div>`;
      }
    }

    html += `<div class="calendar-month">
      <div class="cal-header">
        <span class="cal-month-name">${monthName}</span>
        <span class="cal-month-mae">moy ${currentMonthMetric === "gain" ? monthMean.toFixed(1) + "%" : monthMean.toFixed(4)}</span>
      </div>
      <div class="cal-grid">
        <div class="cal-dow">L</div><div class="cal-dow">M</div><div class="cal-dow">M</div>
        <div class="cal-dow">J</div><div class="cal-dow">V</div><div class="cal-dow">S</div><div class="cal-dow">D</div>
        ${cells}
      </div>
    </div>`;
  });
  wrap.innerHTML = html;

  // Click handlers : navigation vers vue jour
  wrap.querySelectorAll(".cal-day[data-date]").forEach(el => {
    el.addEventListener("click", () => {
      const d = el.dataset.date;
      document.querySelector('.tab[data-panel="jour"]').click();
      setTimeout(() => {
        const sel = document.getElementById("day-select");
        sel.value = d;
        sel.dispatchEvent(new Event("change"));
      }, 100);
    });
  });
}
function renderMonthlyPlots() {
  const m = DATA.monthly;
  // Profil moyen 24h par mois : on agrege depuis by_day
  const byMonth = {};
  Object.entries(DATA.by_day).forEach(([date, day]) => {
    const ym = date.slice(0, 7);
    if (!byMonth[ym]) byMonth[ym] = { hours: Array(96).fill(null).map(() => []),
                                       hoursBase: Array(96).fill(null).map(() => []) };
    for (let i = 0; i < day.load_true.length && i < 96; i++) {
      byMonth[ym].hours[i].push(day.load_true[i]);
      byMonth[ym].hoursBase[i].push(day.forecast_baseline[i]);
    }
  });
  const traces = [];
  const months = Object.keys(byMonth).sort();
  months.forEach((ym, idx) => {
    const mean = byMonth[ym].hours.map(arr => meanArr(arr));
    const x = Array.from({ length: 96 }, (_, i) => `${String(Math.floor(i/4)).padStart(2,'0')}:${String((i%4)*15).padStart(2,'0')}`);
    traces.push({
      x, y: mean, mode: "lines", name: ym, type: "scatter",
      line: { width: 1.8 },
    });
  });
  Plotly.newPlot("plot-monthly-profile", traces, plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, { title: { text: "Heure locale", font: { size: 11 } }, dtick: 16 }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Charge moyenne", font: { size: 11 } } }),
    legend: { orientation: "h", y: -0.2, font: { size: 10 } },
  }), PLOTLY_CONFIG);

  // MAE par mois bar groupe
  Plotly.newPlot("plot-monthly-mae", [
    { x: m.month, y: m.mae_baseline, type: "bar", name: "Baseline",
      marker: { color: getCssVar("--color-baseline") } },
    { x: m.month, y: m.mae_model, type: "bar", name: "Modele",
      marker: { color: getCssVar("--color-model") } },
  ], plotlyLayout({
    barmode: "group",
    xaxis: Object.assign(plotlyLayout().xaxis, { title: { text: "Mois", font: { size: 11 } } }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "MAE", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);
}

document.querySelectorAll('#panel-mois .btn[data-metric]').forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll('#panel-mois .btn[data-metric]').forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    currentMonthMetric = b.dataset.metric;
    renderCalendar();
  });
});

// ============ JOUR ============
let traceVisibility = { real: true, baseline: true, model: true, corrected: false };

function populateDaySelect() {
  const sel = document.getElementById("day-select");
  const dates = Object.keys(DATA.by_day).sort();
  sel.innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join("");
}
function renderDay(date) {
  const day = DATA.by_day[date];
  if (!day) return;

  const x = day.hhmm;
  const traces = [];
  if (traceVisibility.real) {
    traces.push({ x, y: day.load_true, mode: "lines", name: "Reel",
      line: { color: getCssVar("--color-real"), width: 2.2 }, type: "scatter" });
  }
  if (traceVisibility.baseline) {
    traces.push({ x, y: day.forecast_baseline, mode: "lines", name: "Baseline OIKEN",
      line: { color: getCssVar("--color-baseline"), width: 1.6, dash: "dot" }, type: "scatter" });
  }
  if (traceVisibility.model) {
    traces.push({ x, y: day.load_pred_raw, mode: "lines", name: "Modele V2",
      line: { color: getCssVar("--color-model"), width: 1.8 }, type: "scatter" });
  }
  if (traceVisibility.corrected) {
    traces.push({ x, y: day.load_pred_corrected, mode: "lines", name: "Modele + drift",
      line: { color: getCssVar("--color-corrected"), width: 1.6, dash: "dash" }, type: "scatter" });
  }

  Plotly.newPlot("plot-day-main", traces, plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, { dtick: 8, tickangle: 0 }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Charge standardisee", font: { size: 11 } } }),
    margin: { l: 50, r: 18, t: 12, b: 38 },
  }), PLOTLY_CONFIG);

  // Erreurs
  const errModel = day.load_pred_raw.map((v, i) => v - day.load_true[i]);
  const errBase = day.forecast_baseline.map((v, i) => v - day.load_true[i]);
  Plotly.newPlot("plot-day-errors", [
    { x, y: errBase, mode: "lines", name: "Erreur baseline",
      line: { color: getCssVar("--color-baseline"), width: 1.4 }, type: "scatter" },
    { x, y: errModel, mode: "lines", name: "Erreur modele",
      line: { color: getCssVar("--color-model"), width: 1.6 }, type: "scatter" },
    { x, y: x.map(() => 0), mode: "lines", showlegend: false,
      line: { color: getCssVar("--text-faint"), width: 0.8, dash: "solid" }, type: "scatter" },
  ], plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, { dtick: 8 }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Erreur signee", font: { size: 11 } }, zeroline: true }),
    margin: { l: 50, r: 18, t: 12, b: 38 },
  }), PLOTLY_CONFIG);

  // Summary
  const maeModel = meanArr(errModel.map(Math.abs));
  const maeBase = meanArr(errBase.map(Math.abs));
  const rmseModel = Math.sqrt(meanArr(errModel.map(v => v * v)));
  const gain = ((maeBase - maeModel) / maeBase) * 100;
  const loadMean = meanArr(day.load_true);
  const loadMax = Math.max(...day.load_true);
  const loadMin = Math.min(...day.load_true);
  const dt = new Date(date);
  const dowFr = ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"][dt.getDay()];
  const gainCls = gain > 20 ? "delta-neg" : gain < 0 ? "delta-pos" : "";
  const summary = `
    <div class="day-summary-row">
      <div>
        <div class="day-summary-label">Date</div>
        <div class="day-summary-value">${date}</div>
        <div style="font-size:11px; color: var(--text-muted);">${dowFr} | Fold ${day.fold} | ${day.n_points} points</div>
      </div>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">MAE modele</span>
      <span class="day-summary-value">${fmt(maeModel)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">MAE baseline</span>
      <span class="day-summary-value">${fmt(maeBase)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">RMSE modele</span>
      <span class="day-summary-value">${fmt(rmseModel)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Gain MAE</span>
      <span class="day-summary-value ${gainCls}">${gain.toFixed(1)}%</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Charge moyenne</span>
      <span class="day-summary-value">${fmt(loadMean, 3)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Charge max</span>
      <span class="day-summary-value">${fmt(loadMax, 3)}</span>
    </div>
    <div class="day-summary-row">
      <span class="day-summary-label">Charge min</span>
      <span class="day-summary-value">${fmt(loadMin, 3)}</span>
    </div>
  `;
  document.getElementById("day-summary").innerHTML = summary;
  document.getElementById("day-plot-title").textContent = `Charge predite vs reelle | ${date} (${dowFr}, fold ${day.fold})`;
}

document.getElementById("day-select").addEventListener("change", e => {
  renderDay(e.target.value);
});
document.getElementById("day-prev").addEventListener("click", () => {
  const sel = document.getElementById("day-select");
  if (sel.selectedIndex > 0) { sel.selectedIndex--; sel.dispatchEvent(new Event("change")); }
});
document.getElementById("day-next").addEventListener("click", () => {
  const sel = document.getElementById("day-select");
  if (sel.selectedIndex < sel.options.length - 1) { sel.selectedIndex++; sel.dispatchEvent(new Event("change")); }
});
document.querySelectorAll('#panel-jour .btn[data-trace]').forEach(b => {
  b.addEventListener("click", () => {
    b.classList.toggle("active");
    traceVisibility[b.dataset.trace] = b.classList.contains("active");
    renderDay(document.getElementById("day-select").value);
  });
});

// ============ DIAGNOSTICS ============
let diagFold = "all";
function renderDiagnostics() {
  const h = DATA.hourly;
  const d = DATA.dow;

  // Filter by fold
  const filterFold = (data) => {
    if (diagFold === "all") return data;
    const idx = data.fold.map((f, i) => f == diagFold ? i : -1).filter(x => x >= 0);
    const out = {};
    Object.keys(data).forEach(k => out[k] = idx.map(i => data[k][i]));
    return out;
  };
  // Aggregate by hour (mean over folds if all)
  const hf = filterFold(h);
  const hourMap = {};
  hf.hour.forEach((hr, i) => {
    if (!hourMap[hr]) hourMap[hr] = { mae_model: [], mae_baseline: [] };
    hourMap[hr].mae_model.push(hf.mae_model[i]);
    hourMap[hr].mae_baseline.push(hf.mae_baseline[i]);
  });
  const hours = Object.keys(hourMap).sort((a, b) => +a - +b);
  Plotly.newPlot("plot-diag-hour", [
    { x: hours, y: hours.map(h => meanArr(hourMap[h].mae_baseline)), type: "bar", name: "Baseline",
      marker: { color: getCssVar("--color-baseline") }, opacity: 0.85 },
    { x: hours, y: hours.map(h => meanArr(hourMap[h].mae_model)), type: "bar", name: "Modele",
      marker: { color: getCssVar("--color-model") }, opacity: 0.95 },
  ], plotlyLayout({
    barmode: "group",
    xaxis: Object.assign(plotlyLayout().xaxis, { title: { text: "Heure", font: { size: 11 } }, dtick: 1 }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "MAE", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  // DoW
  const df = filterFold(d);
  const dowMap = {};
  df.dow.forEach((dw, i) => {
    if (!dowMap[dw]) dowMap[dw] = { mae_model: [], mae_baseline: [] };
    dowMap[dw].mae_model.push(df.mae_model[i]);
    dowMap[dw].mae_baseline.push(df.mae_baseline[i]);
  });
  const dows = Object.keys(dowMap).sort((a, b) => +a - +b);
  const dowLabels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"];
  Plotly.newPlot("plot-diag-dow", [
    { x: dows.map(d => dowLabels[(+d) - 1] || d), y: dows.map(d => meanArr(dowMap[d].mae_baseline)), type: "bar", name: "Baseline",
      marker: { color: getCssVar("--color-baseline") }, opacity: 0.85 },
    { x: dows.map(d => dowLabels[(+d) - 1] || d), y: dows.map(d => meanArr(dowMap[d].mae_model)), type: "bar", name: "Modele",
      marker: { color: getCssVar("--color-model") }, opacity: 0.95 },
  ], plotlyLayout({
    barmode: "group",
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "MAE", font: { size: 11 } } }),
  }), PLOTLY_CONFIG);

  // Scatter pred vs real (sample)
  const allTrue = [], allPred = [];
  Object.entries(DATA.by_day).forEach(([date, day]) => {
    if (diagFold !== "all" && day.fold != diagFold) return;
    for (let i = 0; i < day.load_true.length; i++) {
      allTrue.push(day.load_true[i]);
      allPred.push(day.load_pred_raw[i]);
    }
  });
  // Subsample if too many
  let sampleTrue = allTrue, samplePred = allPred;
  if (allTrue.length > 6000) {
    const stride = Math.ceil(allTrue.length / 6000);
    sampleTrue = allTrue.filter((_, i) => i % stride === 0);
    samplePred = allPred.filter((_, i) => i % stride === 0);
  }
  const minV = Math.min(...sampleTrue, ...samplePred);
  const maxV = Math.max(...sampleTrue, ...samplePred);
  Plotly.newPlot("plot-diag-scatter", [
    { x: sampleTrue, y: samplePred, mode: "markers", type: "scatter",
      marker: { color: getCssVar("--color-model"), size: 3, opacity: 0.4 },
      name: "Predictions" },
    { x: [minV, maxV], y: [minV, maxV], mode: "lines", type: "scatter",
      line: { color: getCssVar("--text-faint"), dash: "dash", width: 1 },
      name: "y = x", showlegend: true },
  ], plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, { title: { text: "Reel", font: { size: 11 } } }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Predit (modele)", font: { size: 11 } } }),
    hovermode: "closest",
  }), PLOTLY_CONFIG);

  // Hist residus signes
  const allRes = allPred.map((p, i) => p - allTrue[i]);
  Plotly.newPlot("plot-diag-residual-hist", [
    { x: allRes, type: "histogram", marker: { color: getCssVar("--color-model") },
      opacity: 0.85, xbins: { size: 0.02 }, name: "Residus" },
  ], plotlyLayout({
    xaxis: Object.assign(plotlyLayout().xaxis, { title: { text: "Residu signe (pred - true)", font: { size: 11 } }, range: [-1.2, 1.2] }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Frequence", font: { size: 11 } } }),
    showlegend: false,
  }), PLOTLY_CONFIG);
}
document.getElementById("diag-fold-select").addEventListener("change", e => {
  diagFold = e.target.value;
  renderDiagnostics();
});

// ============ DRIFT ============
function renderDrift() {
  const dc = DATA.drift_correction || {};
  document.getElementById("drift-form").textContent = dc.form || "-";
  document.getElementById("drift-tail").textContent = dc.tail_days_for_calibration || "-";
  document.getElementById("drift-clip").textContent = "+/- " + (dc.amplitude_clip || "-");

  // Tableau drift par fold
  const rows = DATA.folds.map(f => {
    const di = f.drift_info || {};
    const mNo = f.metrics_no_correction;
    const mCo = f.metrics_with_correction;
    const delta = ((mCo.mae_load - mNo.mae_load) * 1000); // en milli-units
    return `<tr>
      <td>Fold ${f.fold}</td>
      <td class="num">${(f.drift_slope_a_per_day || 0).toExponential(2)}</td>
      <td class="num">${fmt(f.drift_intercept_b)}</td>
      <td class="num">${fmt(di.r2 || 0, 3)}</td>
      <td class="num">${di.n_clean_days || "-"}</td>
      <td class="num">${fmt(mNo.mae_load)}</td>
      <td class="num">${fmt(mCo.mae_load)}</td>
      <td class="num" style="color:${delta < 0 ? 'var(--color-model)' : delta > 0 ? 'var(--color-error)' : 'var(--text-muted)'}">${delta > 0 ? "+" : ""}${delta.toFixed(2)} e-3</td>
    </tr>`;
  }).join("");
  document.getElementById("tbl-drift").innerHTML = `
    <thead><tr>
      <th>Fold</th><th>Pente a (/jour)</th><th>Intercept b</th>
      <th>R2 fit</th><th>n jours</th>
      <th>MAE sans corr</th><th>MAE avec corr</th><th>Delta MAE</th>
    </tr></thead><tbody>${rows}</tbody>`;

  // Plot pentes
  const slopes = DATA.folds.map(f => f.drift_slope_a_per_day);
  Plotly.newPlot("plot-drift-slope", [
    { x: DATA.folds.map(f => "Fold " + f.fold), y: slopes, type: "bar",
      marker: { color: slopes.map(s => s > 0 ? getCssVar("--color-model") : getCssVar("--color-baseline")) },
      text: slopes.map(s => s.toExponential(2)), textposition: "outside" },
  ], plotlyLayout({
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Pente / jour", font: { size: 11 } }, zeroline: true }),
    showlegend: false,
  }), PLOTLY_CONFIG);

  // Plot R2
  const r2s = DATA.folds.map(f => (f.drift_info || {}).r2 || 0);
  Plotly.newPlot("plot-drift-r2", [
    { x: DATA.folds.map(f => "Fold " + f.fold), y: r2s, type: "bar",
      marker: { color: r2s.map(r => r > 0.1 ? getCssVar("--color-model") : getCssVar("--text-muted")) },
      text: r2s.map(r => r.toFixed(3)), textposition: "outside" },
  ], plotlyLayout({
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "R2 du fit lineaire", font: { size: 11 } }, range: [0, 0.3] }),
    showlegend: false,
  }), PLOTLY_CONFIG);
}

// ============ FEATURES ============
function renderFeatures() {
  const fi = DATA.feature_importance || [];
  if (!fi.length) return;
  const sorted = [...fi].sort((a, b) => b.score - a.score);

  // Categorie -> couleur stable
  const cats = [...new Set(sorted.map(f => f.category))];
  const palette = ["#0F766E", "#1E40AF", "#7C3AED", "#B45309", "#BE123C", "#0891B2", "#65A30D", "#DC2626", "#9333EA", "#0D9488"];
  const catColor = {};
  cats.forEach((c, i) => catColor[c] = palette[i % palette.length]);

  // Bar horizontal top 30
  Plotly.newPlot("plot-fi-bar", [{
    x: sorted.map(f => f.score).reverse(),
    y: sorted.map(f => f.name).reverse(),
    type: "bar",
    orientation: "h",
    marker: { color: sorted.map(f => catColor[f.category]).reverse() },
    text: sorted.map(f => f.category).reverse(),
    textposition: "outside",
    hovertemplate: "<b>%{y}</b><br>Importance: %{x:.2f}<br>Categorie: %{text}<extra></extra>",
  }], plotlyLayout({
    margin: { l: 230, r: 110, t: 12, b: 40 },
    xaxis: Object.assign(plotlyLayout().xaxis, { title: { text: "Importance (gain)", font: { size: 11 } } }),
    yaxis: Object.assign(plotlyLayout().yaxis, { tickfont: { size: 10, family: "'JetBrains Mono', monospace" } }),
    showlegend: false,
  }), PLOTLY_CONFIG);

  // Aggregation par categorie
  const agg = {};
  sorted.forEach(f => {
    if (!agg[f.category]) agg[f.category] = { sum: 0, count: 0 };
    agg[f.category].sum += f.score;
    agg[f.category].count++;
  });
  const aggArr = Object.entries(agg).map(([k, v]) => ({
    category: k, sum: v.sum, count: v.count, mean: v.sum / v.count,
  })).sort((a, b) => b.sum - a.sum);

  Plotly.newPlot("plot-fi-category", [{
    x: aggArr.map(a => a.category),
    y: aggArr.map(a => a.sum),
    type: "bar",
    marker: { color: aggArr.map(a => catColor[a.category]) },
    text: aggArr.map(a => a.count + " feat"),
    textposition: "outside",
    hovertemplate: "<b>%{x}</b><br>Importance totale: %{y:.2f}<br>%{text}<extra></extra>",
  }], plotlyLayout({
    margin: { l: 50, r: 18, t: 12, b: 90 },
    xaxis: Object.assign(plotlyLayout().xaxis, { tickangle: -25, tickfont: { size: 10 } }),
    yaxis: Object.assign(plotlyLayout().yaxis, { title: { text: "Importance totale (gain)", font: { size: 11 } } }),
    showlegend: false,
  }), PLOTLY_CONFIG);

  document.getElementById("tbl-fi-category").innerHTML = `
    <thead><tr>
      <th>Categorie</th><th>n features (top 30)</th><th>Importance totale</th><th>Importance moyenne</th>
    </tr></thead><tbody>
    ${aggArr.map(a => `<tr>
      <td>${a.category}</td>
      <td class="num">${a.count}</td>
      <td class="num">${a.sum.toFixed(2)}</td>
      <td class="num">${a.mean.toFixed(2)}</td>
    </tr>`).join("")}
    </tbody>`;
}

// ============ Render orchestrator ============
function renderAll() {
  const active = document.querySelector(".panel.active").id;
  if (active === "panel-synthese") renderSynthese();
  else if (active === "panel-mois") { renderCalendar(); renderMonthlyPlots(); }
  else if (active === "panel-jour") {
    if (!document.getElementById("day-select").value) {
      populateDaySelect();
    }
    renderDay(document.getElementById("day-select").value);
  }
  else if (active === "panel-diagnostics") renderDiagnostics();
  else if (active === "panel-drift") renderDrift();
  else if (active === "panel-features") renderFeatures();
}

// Initialization
populateDaySelect();
renderSynthese();

// Resize handler
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderAll, 200);
});

</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    """Injecte les couleurs et donnees dans le template."""
    html = HTML_TEMPLATE
    html = html.replace("__COLOR_REAL__", PALETTE["real"])
    html = html.replace("__COLOR_BASELINE__", PALETTE["baseline"])
    html = html.replace("__COLOR_MODEL__", PALETTE["model"])
    html = html.replace("__COLOR_CORRECTED__", PALETTE["corrected"])
    html = html.replace("__COLOR_ERROR__", PALETTE["error"])
    html = html.replace("__COLOR_ACCENT__", PALETTE["accent"])
    # JSON safe : echapper </script> et </ pour eviter sortie de la balise
    payload_json = (json.dumps(payload, ensure_ascii=False, default=str)
                    .replace("</", "<\\/"))
    html = html.replace("__DATA_JSON__", payload_json)
    return html


# ============================================================
# Main
# ============================================================

def main(argv: list[str] | None = None) -> int:
    print("=" * 60)
    print("DASHBOARD V2 - generation HTML")
    print("=" * 60)

    pred_path = DATA_PROCESSED / "predictions_xgboost_cv.parquet"
    report_path = DATA_REPORTS / "model_xgboost_report.json"

    print(f"\nChargement :")
    try:
        data = load_data(pred_path, report_path)
    except FileNotFoundError as e:
        print(f"\nERREUR : {e}")
        print("Lance d'abord : python -m src.model")
        return 1

    print(f"\nAgregations...")
    payload = build_dataset(data)
    print(f"  {payload['meta']['n_days']} jours indexes pour vue jour")
    print(f"  {len(payload['daily']['date'])} aggregats journaliers")
    print(f"  {len(payload['monthly']['month'])} aggregats mensuels")

    print(f"\nGeneration HTML...")
    html = render_html(payload)
    size_mb = len(html.encode("utf-8")) / 1024 / 1024

    out_path = DATA_REPORTS / "dashboard_v2.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  -> {out_path}")
    print(f"  Taille : {size_mb:.2f} MB")
    print("\nDashboard genere. Ouvre le fichier HTML dans ton navigateur.")
    return 0


if __name__ == "__main__":
    sys.exit(main())