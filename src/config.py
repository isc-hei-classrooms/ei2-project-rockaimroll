"""
Configuration centralisee du projet OIKEN ML.

Toute valeur partagee entre plusieurs modules vit ici. Aucune logique,
uniquement des constantes. Les modules font `from src.config import ...`.

Sections:
  - Chemins et arborescence
  - Connexion InfluxDB
  - Sites meteo (perimetre OIKEN)
  - Mesures InfluxDB (reelles + previsionnelles)
  - Bornes physiques pour la detection d'outliers
  - Periode et fuseau OIKEN
"""

from pathlib import Path

# ============================================================
# 1. Chemins
# ============================================================
# PROJECT_ROOT pointe sur la racine du repo, peu importe d'ou le
# script est appele. .parent.parent suppose que ce fichier est en
# src/config.py. Adapter si reorganisation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"            # sorties d'acquisition
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # sorties normalisees
DATA_REPORTS = PROJECT_ROOT / "data" / "reports"      # rapports qualite

# Le CSV source OIKEN doit etre depose ici avant l'execution.
OIKEN_CSV_PATH = DATA_RAW / "oikendata.csv"


# ============================================================
# 2. InfluxDB (HES-SO)
# ============================================================
# Token partage du cours. Si rotation de cle, mettre a jour ici uniquement.
INFLUXDB_URL = "https://timeseries.hevs.ch"
INFLUXDB_TOKEN = (
    "ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0Y"
    "mVrk7hZGPpvb_5aaA-ZxhIw=="
)
INFLUXDB_ORG = "HESSOVS"
INFLUXDB_BUCKET = "MeteoSuisse"        # bucket principal (real + pred)
# Note: le bucket "MeteoSuissePrevision" existe aussi mais n'est pas
# utilise ici. A verifier ulterieurement si son contenu est pertinent.

# Timeout de requete Flux (s). Les requetes lourdes (3 ans, 32 fields)
# peuvent depasser le defaut de 60s.
INFLUX_TIMEOUT_MS = 600_000  # 10 minutes


# ============================================================
# 3. Sites meteo
# ============================================================
# Choix: couvrir le perimetre OIKEN (Valais central) avec une variete
# d'altitudes et de microclimats. Verifier qu'un site retenu a bien
# des donnees sur toute la periode (verification automatique faite
# dans acquisition_meteo_real.py).
#
# Categorisation:
#   - PLAINE        : Sion (482m), Visp (640m), Bouveret (375m)
#   - COTEAU MOYEN  : Montana (1428m), Evolene/Villa (1825m),
#                     Grachen (1605m), Montagnier Bagnes (820m)
#   - HAUTE MONTAGNE: Mottec (1564m), Zermatt (1638m), Evionnaz (482m)
#
# Note typographique: "Evolene / Villa" contient un ASCII slash + espaces.
# C'est exactement la chaine attendue par InfluxDB (verifie via
# explore_influx.py). Toute modification casse la requete.
SITES_VALAIS_CENTRAL = [
    "Sion",
    "Visp",
    "Montana",
    "Evolène / Villa",
]

# Sites etendus (a activer apres verification de couverture temporelle).
SITES_EXTENDED = SITES_VALAIS_CENTRAL + [
    "Mottec",
    "Grächen",
    "Zermatt",
    "Evionnaz",
    "Bouveret",
    "Montagnier, Bagnes",
]

# SITES = liste effectivement utilisee. On commence avec EXTENDED ;
# l'acquisition retire automatiquement les sites incomplets.
SITES = SITES_EXTENDED


# ============================================================
# 4. Mesures InfluxDB
# ============================================================
# 4.a) Reelles : 9 capteurs MeteoSuisse, pas natif 10 minutes.
MEASUREMENTS_REAL = [
    "Air temperature 2m above ground (current value)",
    "Atmospheric pressure at barometric altitude",
    "Global radiation (ten minutes mean)",
    "Gust peak (one second) (maximum)",
    "Precipitation (ten minutes total)",
    "Relative air humidity 2m above ground (current value)",
    "Sunshine duration (ten minutes total)",
    "Wind Direction (ten minutes mean)",
    "Wind speed scalar (ten minutes mean)",
]

# 4.b) Previsionnelles : COSMO-E, 8 variables * 4 statistiques = 32.
# Les statistiques representent la dispersion de l'ensemble (21 membres):
#   ctrl = run de controle (membre deterministe)
#   q10  = quantile 10% (borne basse de l'ensemble)
#   q90  = quantile 90% (borne haute de l'ensemble)
#   stde = ecart-type (dispersion)
PRED_VARIABLES = [
    "PRED_T_2M",      # Temperature 2m
    "PRED_GLOB",      # Rayonnement global
    "PRED_DURSUN",    # Duree d'ensoleillement
    "PRED_TOT_PREC",  # Precipitations totales
    "PRED_RELHUM_2M", # Humidite relative 2m
    "PRED_FF_10M",    # Vitesse vent 10m
    "PRED_DD_10M",    # Direction vent 10m
    "PRED_PS",        # Pression au sol
]
PRED_VARIANTS = ["ctrl", "q10", "q90", "stde"]

MEASUREMENTS_PRED = [
    f"{var}_{variant}"
    for var in PRED_VARIABLES
    for variant in PRED_VARIANTS
]

# 4.c) Quel run de prevision utiliser ?
# Les runs vont de "00" a "45" (= heure de lancement modulo 6h en realite,
# COSMO-E tournant 4 fois par jour). Pour un forecast J+1 publie le matin
# avant 06:00, on utilise le run "00" (lance a 00 UTC, disponible vers 03h).
PRED_RUN = "00"


# ============================================================
# 5. Renommage des colonnes
# ============================================================
# Les noms InfluxDB sont longs et fragiles. On les mappe vers des noms
# courts, snake_case, utilisables comme noms de colonnes Polars.
MEASUREMENT_SHORT_NAMES = {
    "Air temperature 2m above ground (current value)": "temperature_2m",
    "Atmospheric pressure at barometric altitude": "pressure",
    "Global radiation (ten minutes mean)": "global_radiation",
    "Gust peak (one second) (maximum)": "gust_peak",
    "Precipitation (ten minutes total)": "precipitation",
    "Relative air humidity 2m above ground (current value)": "humidity",
    "Sunshine duration (ten minutes total)": "sunshine_duration",
    "Wind Direction (ten minutes mean)": "wind_direction",
    "Wind speed scalar (ten minutes mean)": "wind_speed",
}

# Pour les previsions, le nom InfluxDB en minuscules suffit (deja court).


# ============================================================
# 6. Bornes physiques (validation des donnees)
# ============================================================
# Plages plausibles pour une station valaisanne (altitude 300-1900m).
# Toute valeur en dehors est marquee comme outlier en normalisation,
# puis remplacee selon une strategie (NULL + interpolation, voir
# normalization.py). Sources: ranges typiques MeteoSuisse + bon sens.
PHYSICAL_BOUNDS = {
    "temperature_2m":     (-40.0,  45.0),  # °C
    "pressure":           ( 700.0, 1050.0),  # hPa (corrige niveau mer)
    "global_radiation":   (   0.0, 1400.0),  # W/m² (max theorique ~1100)
    "gust_peak":          (   0.0,  60.0),  # m/s (au-dela = tempete)
    "precipitation":      (   0.0,  50.0),  # mm/10min (extreme)
    "humidity":           (   0.0, 100.0),  # %
    "sunshine_duration":  (   0.0,  10.0),  # min/10min
    "wind_direction":     (   0.0, 360.0),  # degres
    "wind_speed":         (   0.0,  40.0),  # m/s moyen
}

# Charge OIKEN: standardisee, donc theoriquement centree autour de 0.
# Pas de borne physique stricte, mais un seuil sigma pour outliers.
LOAD_OUTLIER_SIGMA = 6.0
PV_MIN_KWH = 0.0          # PV ne peut pas etre negatif
PV_MAX_KWH = 50_000.0     # plafond defensif (reseau OIKEN)


# ============================================================
# 7. Periode et fuseau
# ============================================================
# Le CSV OIKEN couvre du 01.10.2022 00:15 au 29.09.2025 23:45 inclus,
# soit 105'120 lignes en pas 15 min (= 3 ans en grille locale).
OIKEN_START = "2022-10-01"  # bornes pour les requetes InfluxDB
OIKEN_END   = "2025-09-30"
OIKEN_EXPECTED_ROWS = 105_120

# Fuseau dans lequel les timestamps OIKEN sont ETIQUETES.
# Hypothese forte (a verifier en normalisation): le CSV est en heure
# locale suisse, avec changements DST. Les timestamps sont des
# "wall-clock" locaux, pas des UTC.
TIMEZONE_OIKEN = "Europe/Zurich"

# Pas de temps cible pour la fusion finale.
TARGET_RESOLUTION = "15m"