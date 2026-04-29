"""
Visualisations pour la justification des choix de features.

Génère 6 graphiques qui expliquent :
1. Top corrélations avec la charge (justification features gardées)
2. Data leakage des lags courts (justification exclusion en J+1)
3. Patterns temporels (justification features calendaires)
4. Température vs charge non-linéaire (justification LightGBM)
5. Impact du PV sur la charge nette (justification features PV)
6. Effet vacances/fériés (justification is_weekend, is_holiday, is_school_holiday)

Les figures sont sauvegardées dans data/processed/figures/

Utilisation :
    python -m src.visualisation
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from archives.config import DATA_PROCESSED, TIMEZONE


FIG_DIR = DATA_PROCESSED / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

BLUE = "#2b6cb0"
RED = "#e53e3e"
GREEN = "#38a169"
ORANGE = "#dd6b20"


# ============================================================
# 1. Top corrélations avec la charge
# ============================================================

def plot_top_correlations(df: pl.DataFrame):
    """Bar plot des features utilisées en forecast J+1, triées par corrélation."""

    # Features réellement disponibles et utilisées en J+1
    j1_features = [
        "load_lag_96", "load_lag_672",
        "pv_total_lag_96", "pv_total_lag_672",
        "pred_t_2m_ctrl", "pred_glob_ctrl",
        "pred_t_2m_stde", "pred_glob_stde",
        "doy_cos", "month_cos",
    ]

    # Garder uniquement celles présentes dans le dataset
    available = [f for f in j1_features if f in df.columns]

    correlations = {}
    for col in available:
        corr = df.select(pl.corr("load", col)).item()
        if corr is not None and not np.isnan(corr):
            correlations[col] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    names = [x[0] for x in sorted_corr][::-1]
    values = [x[1] for x in sorted_corr][::-1]
    colors = [BLUE if v > 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Corrélation avec la charge (load)")
    ax.set_title("Features utilisées en forecast J+1\n"
                 "Corrélation avec la charge",
                 fontweight="bold", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "01_top_correlations.png"), dpi=150)
    plt.close()
    print("  -> 01_top_correlations.png")


# ============================================================
# 2. Data leakage : load_lag_1 vs load
# ============================================================

def plot_data_leakage(df: pl.DataFrame):
    """Scatter plot load_lag_1 vs load + série temporelle sur 2 jours."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "load_lag_1" in df.columns:
        sample = df.sample(n=min(5000, df.shape[0]), seed=42)
        ax = axes[0]
        ax.scatter(
            sample["load_lag_1"].to_numpy(),
            sample["load"].to_numpy(),
            alpha=0.1, s=5, color=RED
        )
        ax.plot([0, 2], [0, 2], "k--", linewidth=1, label="y = x")
        ax.set_xlabel("load_lag_1 (charge il y a 15 min)")
        ax.set_ylabel("load (charge actuelle)")
        ax.set_title("load_lag_1 vs load : quasi identiques\n-> DATA LEAKAGE en J+1",
                     fontweight="bold", color=RED)
        corr = df.select(pl.corr("load", "load_lag_1")).item()
        ax.text(0.05, 0.95, f"r = {corr:.4f}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", color=RED)
        ax.legend()

    ax = axes[1]
    n = df.shape[0]
    slice_df = df.slice(n // 2, 192)
    ts = slice_df["timestamp"].to_numpy()
    ax.plot(ts, slice_df["load"].to_numpy(), label="load (réel)", color=BLUE, linewidth=1.5)
    if "load_lag_1" in slice_df.columns:
        ax.plot(ts, slice_df["load_lag_1"].to_numpy(), label="load_lag_1",
                color=RED, linewidth=1, linestyle="--", alpha=0.7)
    if "load_lag_96" in slice_df.columns:
        ax.plot(ts, slice_df["load_lag_96"].to_numpy(), label="load_lag_96 (J-1)",
                color=GREEN, linewidth=1, linestyle=":", alpha=0.7)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Charge normalisée")
    ax.set_title("Comparaison des lags sur 2 jours", fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "02_data_leakage_lags.png"), dpi=150)
    plt.close()
    print("  -> 02_data_leakage_lags.png")


# ============================================================
# 3. Patterns temporels de la charge
# ============================================================

def plot_charge_patterns(df: pl.DataFrame):
    """Profils moyens de charge par heure, jour de semaine et mois."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    local_ts = df["timestamp"].dt.convert_time_zone(TIMEZONE)

    ax = axes[0]
    hour_data = df.with_columns(local_ts.dt.hour().alias("h"))
    hourly = hour_data.group_by("h").agg(
        pl.col("load").mean().alias("mean"),
        pl.col("load").std().alias("std"),
    ).sort("h")
    h = hourly["h"].to_numpy()
    m = hourly["mean"].to_numpy()
    s = hourly["std"].to_numpy()
    ax.plot(h, m, color=BLUE, linewidth=2)
    ax.fill_between(h, m - s, m + s, alpha=0.2, color=BLUE)
    ax.set_xlabel("Heure")
    ax.set_ylabel("Charge moyenne")
    ax.set_title("Profil journalier\n-> justifie hour_sin / hour_cos", fontweight="bold")
    ax.set_xticks(range(0, 24, 3))

    ax = axes[1]
    wd_data = df.with_columns(local_ts.dt.weekday().alias("wd"))
    weekly = wd_data.group_by("wd").agg(pl.col("load").mean()).sort("wd")
    days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    ax.bar(range(7), weekly["load"].to_numpy(), color=BLUE, alpha=0.8)
    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.set_ylabel("Charge moyenne")
    ax.set_title("Profil hebdomadaire\n-> justifie weekday_sin + is_weekend", fontweight="bold")

    ax = axes[2]
    mo_data = df.with_columns(local_ts.dt.month().alias("mo"))
    monthly = mo_data.group_by("mo").agg(pl.col("load").mean()).sort("mo")
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax.bar(range(12), monthly["load"].to_numpy(), color=BLUE, alpha=0.8)
    ax.set_xticks(range(12))
    ax.set_xticklabels(months)
    ax.set_ylabel("Charge moyenne")
    ax.set_title("Profil saisonnier\n-> justifie month_sin + doy_sin", fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "03_charge_patterns.png"), dpi=150)
    plt.close()
    print("  -> 03_charge_patterns.png")


# ============================================================
# 4. Température vs charge
# ============================================================

def plot_temperature_vs_load(df: pl.DataFrame):
    """Scatter plot température prédite vs charge."""

    if "pred_t_2m_ctrl" not in df.columns:
        print("  (pred_t_2m_ctrl absent, graphique ignoré)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    sample = df.sample(n=min(10000, df.shape[0]), seed=42)
    ax.scatter(
        sample["pred_t_2m_ctrl"].to_numpy(),
        sample["load"].to_numpy(),
        alpha=0.05, s=3, color=BLUE
    )

    temp = df["pred_t_2m_ctrl"].to_numpy()
    load = df["load"].to_numpy()
    bins = np.linspace(np.nanpercentile(temp, 1), np.nanpercentile(temp, 99), 30)
    bin_centers = []
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (temp >= bins[i]) & (temp < bins[i+1])
        if mask.sum() > 50:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(load[mask]))

    ax.plot(bin_centers, bin_means, color=RED, linewidth=2.5, label="Moyenne par bin")
    ax.set_xlabel("Température prédite (°C)")
    ax.set_ylabel("Charge normalisée")
    ax.set_title("Température vs Charge\n"
                 "Corrélation forte : le chauffage domine la consommation",
                 fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "04_temperature_vs_charge.png"), dpi=150)
    plt.close()
    print("  -> 04_temperature_vs_charge.png")



# ============================================================
# 6. Vacances et jours fériés
# ============================================================

def plot_holiday_effect(df: pl.DataFrame):
    """Box plots de la charge selon le type de jour."""

    fig, ax = plt.subplots(figsize=(8, 5))

    normal = df.filter(
        (pl.col("is_weekend") == 0) &
        (pl.col("is_holiday") == 0) &
        (pl.col("is_school_holiday") == 0)
    )["load"].to_numpy()

    weekend = df.filter(pl.col("is_weekend") == 1)["load"].to_numpy()

    vacances = df.filter(
        (pl.col("is_school_holiday") == 1) &
        (pl.col("is_weekend") == 0)
    )["load"].to_numpy()

    feries = df.filter(
        (pl.col("is_holiday") == 1) &
        (pl.col("is_weekend") == 0)
    )["load"].to_numpy()

    data_by_cat = [normal.tolist(), weekend.tolist(), vacances.tolist(), feries.tolist()]
    cat_names = ["Jour ouvré", "Week-end", "Vacances\nscolaires", "Jour férié"]

    bp = ax.boxplot(data_by_cat, labels=cat_names, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=2))

    colors_box = [BLUE, ORANGE, GREEN, RED]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Charge normalisée")
    ax.set_title("Impact du type de jour sur la charge\n"
                 "-> justifie is_weekend, is_holiday, is_school_holiday",
                 fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "06_holiday_effect.png"), dpi=150)
    plt.close()
    print("  -> 06_holiday_effect.png")


# ============================================================
# Point d'entrée
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALISATIONS — Energy Informatics 2")
    print("=" * 60)

    print("\nChargement du dataset...")
    feat_path = DATA_PROCESSED / "dataset_features.parquet"
    merged_path = DATA_PROCESSED / "dataset_merged.parquet"

    if feat_path.exists():
        df = pl.read_parquet(feat_path)
        print(f"  dataset_features : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    elif merged_path.exists():
        df = pl.read_parquet(merged_path)
        print(f"  dataset_merged : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    else:
        print("ERREUR : aucun dataset trouvé. Lance d'abord :")
        print("  python -m src.normalization")
        print("  python -m src.features")
        exit(1)

    print(f"\nGénération des figures dans {FIG_DIR}/\n")

    plot_top_correlations(df)
    plot_data_leakage(df)
    plot_charge_patterns(df)
    plot_temperature_vs_load(df)
    plot_pv_impact(df)
    plot_holiday_effect(df)

    print(f"\n-> {len(list(FIG_DIR.glob('*.png')))} figures générées dans {FIG_DIR}/")
    print("\n" + "=" * 60)
    print("VISUALISATIONS TERMINÉES")
    print("=" * 60)