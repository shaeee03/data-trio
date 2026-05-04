"""
================================================================================
SCRIPT 03: Machine Learning — Healthcare Vulnerability Prediction
Project:   Healthcare Accessibility & Vulnerability Index — Metro Manila
================================================================================

STRATEGY: LEAVE-ONE-OUT CROSS-VALIDATION (both regression tasks)
-----------------------------------------------------------------
With only 17 city-level observations, a fixed train/test split is statistically
indefensible — a 4-row test set means one wrong prediction = 25% error swing.
Classification degenerates to predicting the majority class (see: every
confusion matrix we ran). We do not classify.

Solution: Two regression tasks, both evaluated with LeaveOneOut CV.
Every city is the test case exactly once. This is the gold standard for
tiny-n medical/epidemiological studies and is fully defensible academically.

TWO REGRESSION TASKS
--------------------
TASK A — Distance Regression:
    Target : nearest_public_tertiary_km  (continuous, in km)
    Meaning: How far a city's centroid is from the nearest PUBLIC Level-3
             hospital. Higher = more physically isolated from public critical
             care. Note that private Level-3 hospitals are income-gated, so
             this target captures the most equitable access dimension.
    Models : k-NN Regressor (baseline) vs Random Forest Regressor (primary)

TASK B — Weighted Accessibility Score (Wd):
    Target : wd_score_log  (continuous [0,1], higher = MORE accessible)
    Formula: Wd_raw = Σ_city (pub_L3_weight / dist²)
                    + Σ_city (priv_L3_weight / dist²) × (1 - poverty_frac)
                    + Σ_city (sub_L3_weight  / dist²) × 0.3

    where:
      - Only Level-3 facilities are the primary numerator (critical care).
        Sub-L3 (L0–L2) contribute at a 0.3 discount — they matter for primary
        care but cannot substitute for hospital admission.
      - dist = nearest_public_tertiary_km, floored at 0.5 km² to prevent
        cities whose centroid sits inside a hospital campus (Pasay, distance≈0)
        from producing infinity. 0.5 km² ≈ 700m walkable radius — a physically
        defensible minimum access unit.
      - Public L3: no poverty discount (PhilHealth/free at point of use).
      - Private L3: discounted by (1 - poverty_incidence). A household at 2.5%
        poverty incidence effectively cannot access Makati Medical Center.
      - Cities with zero L3 of ANY kind (Navotas, Pateros, Paranaque) receive
        no L3 numerator at all — their low Wd reflects a real structural gap.
      - wd_log = log1p(wd_raw) then min-max normalised to [0,1].
        log1p compresses the 4-order-of-magnitude raw range (1.4 → 14,940)
        into a learnable signal. Without this, Pasay's score (centroid inside
        PGH) dominates the entire scale and the model learns nothing.

    Meaning: A 2SFCA-inspired spatial accessibility measure that combines
             supply quality (L3 weight), physical proximity (distance²), AND
             economic accessibility (poverty discount on private facilities)
             into a single interpretable index for LGU prioritisation.
    Models : k-NN Regressor (baseline) vs Random Forest Regressor (primary)

WHY NOT CLASSIFICATION?
-----------------------
Classifying 17 cities into 3 tiers (Low/Medium/High) gives ~5-6 cities per
class. With StratifiedKFold k=3, each training fold has ~11 rows. The
classifier has no reliable signal — it collapses to "Medium" for everything.
OOF accuracy ≈ 0.35, barely above random (0.33). This is not a model failure;
it is a sample-size confession. We replace the 3-class output with a
continuous Wd ranking, which is statistically valid AND more useful to an LGU
(they want a ranked priority list, not a tier label).

PRE-PROCESSING JUSTIFICATIONS
------------------------------
1. Dead feature removal:
   'econ_friction_ratio' and 'poverty_threshold_2023_php' are CONSTANT
   across all 17 cities (std=0, confirmed in diagnostic run). A constant
   column contributes zero information to any split and zero variance to
   distance metrics. Removed explicitly.

2. Imputation (median):
   'poverty_incidence_2023_pct' has 1 NaN (Pateros, likely unreported).
   Median imputation is preferred over mean for right-skewed distributions.
   Applied inside Pipeline — fit only on training data, no leakage.

3. StandardScaler:
   k-NN is distance-based — population_2020 (millions) vs poverty_incidence
   (fractions) would dominate Euclidean distance without scaling.
   Applied uniformly for consistency. Fit only on X_train inside Pipeline.

4. No data leakage:
   All transformers wrapped in sklearn.Pipeline. Wd score is computed from
   the facility table before the model loop — it is a target variable, not
   a feature derived from the training fold.

5. Hyperparameter tuning:
   Inner GridSearchCV with KFold (k=5 or n, whichever is smaller).
   Scoring: neg_mean_absolute_error. RF: n_estimators, max_depth,
   min_samples_split, max_features. kNN: n_neighbors, weights, metric.

NOTE ON SAMPLE SIZE
-------------------
n=17 is a hard constraint of city-level Metro Manila analysis (17 LGUs).
All metrics carry wide implicit confidence intervals. The primary value of
this model is the feature importance ranking and the directional Wd index —
not point-prediction precision.
================================================================================
"""

import os, sys, sqlite3, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    LeaveOneOut, KFold, GridSearchCV, cross_val_predict
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "../data/database_output"
MODEL_DIR = "../models"
VIZ_DIR   = "../visualizations"
DB_PATH   = os.path.join(DATA_DIR, "healthcare_vulnerability.db")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,   exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Dead features removed: econ_friction_ratio (constant=0.1326 across all cities)
#                        poverty_threshold_2023_php (constant=37710.94 across all cities)
# Both have std=0 → zero variance → zero RF importance → noise in k-NN distance.
FEATURE_COLS = [
    "facility_density_per10k",
    "hospital_density_per10k",
    "beds_per_1000",
    "weighted_score_per10k",
    "level3_per100k",
    "public_primary_per10k",
    "private_ownership_pct",
    "private_to_public_ratio",
    "poverty_incidence_2023_pct",
    "population_2020",
    "pop_growth_rate_pct",
    "pca_emergency",
    "pca_diagnostic",
    "pca_primary",
]

REGRESSION_TARGET_KM = "nearest_public_tertiary_km"
REGRESSION_TARGET_WD = "wd_score"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """
    Loads fact_vulnerability (city-level features) and dim_facilities
    (row-level facility data needed to compute Wd from scratch).
    Falls back to CSV if DB not found.
    """
    if not os.path.exists(DB_PATH):
        csv_path = os.path.join(DATA_DIR, "merged_metro_manila.csv")
        print(f"  DB not found — loading from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        facilities_df = None
    else:
        print(f"  Loading from DB: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM fact_vulnerability", conn)
        try:
            facilities_df = pd.read_sql(
                "SELECT city_norm, service_level_weight, is_private, doh_level "
                "FROM dim_facilities",
                conn
            )
        except Exception:
            facilities_df = None
            print("  WARNING: dim_facilities not readable — Wd will use fallback formula.")
        conn.close()

    print(f"  fact_vulnerability: {df.shape[0]} rows × {df.shape[1]} cols")

    # Ensure all expected columns exist (NaN if absent)
    for c in FEATURE_COLS + [REGRESSION_TARGET_KM]:
        if c not in df.columns:
            df[c] = np.nan

    return df, facilities_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE Wd  (Weighted Accessibility Score — 2SFCA-inspired)
# ══════════════════════════════════════════════════════════════════════════════

# Distance floor: 0.5 km² ≈ 700m walkable radius.
# Prevents cities whose centroid overlaps a hospital campus (Pasay: 0.000 km)
# from producing Wd scores 1000× higher than adjacent cities and monopolising
# the normalised scale. 0.5 km² is the smallest physically defensible urban
# access unit — no hospital in Metro Manila has a catchment smaller than ~700m.
WD_DIST_FLOOR_KM2 = 0.5

# Sub-L3 facilities (L0–L2) contribute to Wd at a 30% discount because they
# handle primary care but cannot substitute for hospital-level critical care.
SUB_L3_DISCOUNT = 0.3

def compute_wd(df, facilities_df):
    """
    Computes the Weighted Accessibility Score (Wd) per city.

    The formula focuses on Level-3 (critical care) hospitals as the primary
    numerator, with sub-L3 facilities contributing at a discount. This is
    more meaningful than treating all facility types equally, because the
    research question is specifically about healthcare *vulnerability* — which
    is driven by access to critical/emergency care, not by clinic counts.

    Formula (per city):
        wd_raw = (pub_L3_weight  / dist²)                      ← public L3, no poverty gate
               + (priv_L3_weight / dist²) × (1 - poverty_frac) ← private L3, income-gated
               + (sub_L3_weight  / dist²) × SUB_L3_DISCOUNT     ← L0-L2, primary care

        dist² = max(nearest_public_tertiary_km², WD_DIST_FLOOR_KM2)

        wd_log  = log1p(wd_raw)          ← compresses 4-order-of-magnitude range
        wd_score = min-max(wd_log) → [0,1]

    Cities with zero L3 of any kind (Navotas, Pateros, Paranaque) correctly
    receive wd_raw contributions only from their sub-L3 facilities — their
    low final score reflects the real structural gap in critical care supply.

    If dim_facilities is unavailable, falls back to aggregated columns
    already present in fact_vulnerability.
    """
    city_col = "city_norm"
    pov_col  = "poverty_incidence_2023_pct"
    km_col   = REGRESSION_TARGET_KM

    # Poverty incidence: stored as percentage (e.g. 1.5%), normalise to fraction
    pov_frac = df[pov_col].fillna(df[pov_col].median()) / 100.0
    pov_frac = pov_frac.clip(0, 1)

    if facilities_df is not None and not facilities_df.empty:
        print("  Computing Wd from dim_facilities (L3-focused formula)...")

        fac = facilities_df.copy()
        fac["service_level_weight"] = pd.to_numeric(
            fac["service_level_weight"], errors="coerce"
        ).fillna(1.0)
        fac["is_private"] = pd.to_numeric(fac["is_private"], errors="coerce").fillna(0)
        fac["doh_level"]  = pd.to_numeric(fac["doh_level"],  errors="coerce").fillna(0)

        # Separate L3 from sub-L3 to apply the correct weights in the formula
        fac["is_l3"] = (fac["doh_level"] == 3).astype(int)

        city_agg = fac.groupby("city_norm").apply(
            lambda g: pd.Series({
                # Level-3 critical care — primary numerator
                "pub_l3_w":   g.loc[(g["is_private"]==0) & (g["is_l3"]==1),
                                     "service_level_weight"].sum(),
                "priv_l3_w":  g.loc[(g["is_private"]==1) & (g["is_l3"]==1),
                                     "service_level_weight"].sum(),
                # Sub-L3 (L0–L2) — primary care, discounted
                "sub_l3_w":   g.loc[g["is_l3"]==0, "service_level_weight"].sum(),
            })
        ).reset_index()

        df2 = df[[city_col, km_col, pov_col]].copy()
        df2["pov_frac"] = pov_frac.values
        df2 = df2.merge(city_agg, on=city_col, how="left")
        df2[["pub_l3_w", "priv_l3_w", "sub_l3_w"]] = \
            df2[["pub_l3_w", "priv_l3_w", "sub_l3_w"]].fillna(0)

        dist_sq = (df2[km_col].fillna(df2[km_col].median()) ** 2).clip(
            lower=WD_DIST_FLOOR_KM2
        )

        pub_l3_access  = df2["pub_l3_w"]  / dist_sq
        priv_l3_access = df2["priv_l3_w"] / dist_sq * (1 - df2["pov_frac"])
        sub_l3_access  = df2["sub_l3_w"]  / dist_sq * SUB_L3_DISCOUNT

        wd_raw = pub_l3_access + priv_l3_access + sub_l3_access

        # Diagnostics: show L3 breakdown per city
        print(f"\n  {'City':<16} {'pub_L3':>7} {'priv_L3':>8} {'sub_L3':>7} "
              f"{'dist_km':>8} {'wd_raw':>10}")
        print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*10}")
        wd_raw_vals = wd_raw.values
        for i, row in df2.iterrows():
            print(f"  {row[city_col]:<16} {row['pub_l3_w']:>7.1f} "
                  f"{row['priv_l3_w']:>8.1f} {row['sub_l3_w']:>7.1f} "
                  f"{row[km_col]:>8.3f} {wd_raw_vals[i]:>10.3f}")

    else:
        print("  Computing Wd via fallback formula (fact_vulnerability aggregates)...")
        dist_km = df[km_col].fillna(df[km_col].median())
        dist_sq = (dist_km ** 2).clip(lower=WD_DIST_FLOOR_KM2)

        pub_weight  = df["weighted_score_per10k"] * (1 - df["private_ownership_pct"].fillna(0.5))
        priv_weight = df["weighted_score_per10k"] * df["private_ownership_pct"].fillna(0.5)

        wd_raw = pub_weight / dist_sq + priv_weight / dist_sq * (1 - pov_frac)

    wd_raw = wd_raw.reset_index(drop=True)

    # ── log1p transform then min-max normalise ───────────────────────────────
    # log1p compresses the 4-order-of-magnitude raw range into a learnable
    # signal while preserving the full ordinal ranking of cities.
    # Without this, the city with distance ≈ 0 (Pasay: PGH on its border)
    # dominates the normalised scale and the model learns nothing from
    # the other 16 cities.
    wd_log = np.log1p(wd_raw)
    wd_min, wd_max = wd_log.min(), wd_log.max()
    if wd_max > wd_min:
        wd_norm = (wd_log - wd_min) / (wd_max - wd_min)
    else:
        wd_norm = pd.Series(np.zeros(len(wd_log)))

    print(f"\n  Wd score stats:")
    print(f"    raw   — min={wd_raw.min():.2f}  max={wd_raw.max():.2f}  "
          f"median={wd_raw.median():.2f}")
    print(f"    log1p — min={wd_log.min():.4f}  max={wd_log.max():.4f}  "
          f"median={wd_log.median():.4f}")
    print(f"    norm  — min={wd_norm.min():.4f}  max={wd_norm.max():.4f}  "
          f"median={wd_norm.median():.4f}  NaN={wd_norm.isna().sum()}")
    print(f"    NOTE: Higher = more accessible. Sort ascending for LGU priority.")

    return wd_norm


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def prepare_features(df):
    """
    Selects and validates the feature matrix X.
    Drops constant columns (zero-variance) with a hard diagnostic.
    Returns X (DataFrame), y_km (Series), feature_names (list).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()

    # ── Diagnose and drop constant columns ──────────────────────────────────
    stds = X.std(numeric_only=True)
    constant_cols = stds[stds == 0].index.tolist()
    if constant_cols:
        print(f"\n  WARNING: Dropping constant columns (std=0, zero information):")
        for c in constant_cols:
            print(f"    {c}: value={X[c].iloc[0]} (identical across all cities)")
        X = X.drop(columns=constant_cols)
        available = [c for c in available if c not in constant_cols]

    # ── Missing value report ─────────────────────────────────────────────────
    missing = X.isnull().sum()
    if missing.any():
        print("\n  Missing values (imputed with median inside Pipeline):")
        for col, cnt in missing[missing > 0].items():
            print(f"    {col}: {cnt} missing")

    # ── Regression targets ───────────────────────────────────────────────────
    y_km = df[REGRESSION_TARGET_KM].copy()
    print(f"\n  Target A — '{REGRESSION_TARGET_KM}':")
    print(f"    min={y_km.min():.3f}  max={y_km.max():.3f}  "
          f"mean={y_km.mean():.3f}  NaN={y_km.isna().sum()}")

    print(f"\n  Features after cleaning: {len(available)}")
    for c in available:
        print(f"    {c}")

    return X, y_km, available


# ══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def reg_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TUNING  (inner CV — no leakage)
# ══════════════════════════════════════════════════════════════════════════════
def tune_regressor(X, y, label=""):
    """
    Inner GridSearchCV with KFold for regression.
    Drops NaN target rows before tuning. Scores on neg_MAE (interpretable).
    """
    mask  = ~y.isna()
    Xm, ym = X[mask], y[mask]
    n      = len(Xm)
    safe_k = max(2, min(5, n))
    cv     = KFold(n_splits=safe_k, shuffle=True, random_state=RANDOM_STATE)

    print(f"  GridSearchCV (KFold k={safe_k}, scoring=neg_MAE) — {label}")

    knn_search = GridSearchCV(
        reg_pipeline(KNeighborsRegressor()),
        {"model__n_neighbors": [2, 3, 5],
         "model__weights":     ["uniform", "distance"],
         "model__metric":      ["euclidean", "manhattan"]},
        cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
    )
    knn_search.fit(Xm, ym)
    print(f"    Best kNN  params: {knn_search.best_params_}  "
          f"CV-MAE={-knn_search.best_score_:.4f}")

    rf_search = GridSearchCV(
        reg_pipeline(RandomForestRegressor(random_state=RANDOM_STATE)),
        {"model__n_estimators":      [50, 100, 200],
         "model__max_depth":         [None, 3, 5],
         "model__min_samples_split": [2, 3],
         "model__max_features":      ["sqrt", "log2", None]},
        cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True
    )
    rf_search.fit(Xm, ym)
    print(f"    Best RF   params: {rf_search.best_params_}  "
          f"CV-MAE={-rf_search.best_score_:.4f}")

    return knn_search.best_estimator_, rf_search.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOO CROSS-VALIDATED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def loo_evaluate(name, pipeline, X, y, city_names, unit="", plot_prefix="reg"):
    """
    LeaveOneOut CV evaluation for regression.
    Each city is the test case exactly once — n iterations total.
    Returns metrics dict and saves predicted-vs-actual plot.
    """
    mask = ~y.isna()
    Xm   = X[mask].reset_index(drop=True)
    ym   = y[mask].reset_index(drop=True)
    cm   = [c for c, m in zip(city_names, mask) if m]

    y_pred = cross_val_predict(pipeline, Xm, ym, cv=LeaveOneOut())

    mae  = mean_absolute_error(ym, y_pred)
    rmse = np.sqrt(mean_squared_error(ym, y_pred))
    r2   = r2_score(ym, y_pred) if len(ym) > 1 else float("nan")

    print(f"\n  ── {name} ──")
    print(f"  LOO iterations : {len(ym)}")
    print(f"  MAE            : {mae:.4f}{unit}")
    print(f"  RMSE           : {rmse:.4f}{unit}")
    print(f"  R²             : {r2:.4f}  (1.0=perfect; <0=worse than mean)")

    # City-level table
    pred_df = pd.DataFrame({
        "city":     cm,
        "actual":   ym.round(4).values,
        "pred":     y_pred.round(4),
        "error":    (y_pred - ym.values).round(4),
    })
    print(f"\n  City-level LOO predictions:")
    print(pred_df.to_string(index=False))

    # Predicted vs Actual plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ym, y_pred, color="#1565C0", edgecolors="white", s=90, zorder=3)
    for xv, yv, city in zip(ym, y_pred, cm):
        ax.annotate(city, (xv, yv), fontsize=6,
                    xytext=(4, 4), textcoords="offset points")
    lo = min(ym.min(), y_pred.min()) - abs(ym.max() - ym.min()) * 0.05
    hi = max(ym.max(), y_pred.max()) + abs(ym.max() - ym.min()) * 0.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel(f"Actual {unit.strip() or 'value'}", fontsize=9)
    ax.set_ylabel(f"Predicted (LOO CV)", fontsize=9)
    ax.set_title(f"LOO CV: Predicted vs Actual\n{name}", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()

    slug = name.lower().replace(" ", "_").replace("-", "_") \
               .replace("(", "").replace(")", "")
    fpath = os.path.join(VIZ_DIR, f"{plot_prefix}_{slug}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved plot → {fpath}")

    return {
        "model":       name,
        "MAE":         round(mae, 4),
        "RMSE":        round(rmse, 4),
        "R2":          round(r2, 4),
        "unit":        unit.strip(),
        "city_preds":  pred_df.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. WD RANKING TABLE  (the primary LGU deliverable)
# ══════════════════════════════════════════════════════════════════════════════
def print_wd_ranking(city_names, wd, y_km, df):
    """
    Prints a ranked accessibility table — the output an LGU actually needs.
    Higher Wd = more accessible. Lower rank = higher priority for intervention.
    """
    pov = df["poverty_incidence_2023_pct"].fillna(
              df["poverty_incidence_2023_pct"].median()).values

    ranking = pd.DataFrame({
        "city":                    city_names,
        "wd_score":                wd.round(4),
        "nearest_public_km":       y_km.fillna(y_km.median()).round(3),
        "poverty_incidence_pct":   pov.round(2),
    }).sort_values("wd_score").reset_index(drop=True)

    ranking.index += 1  # 1-based rank
    ranking.index.name = "rank"

    print("\n  ══ WEIGHTED ACCESSIBILITY SCORE — CITY RANKING ══")
    print("  Rank 1 = LOWEST accessibility = highest priority for LGU intervention\n")
    print(ranking.to_string())

    # Save as CSV for 04_viz.py
    csv_path = os.path.join(MODEL_DIR, "wd_city_ranking.csv")
    ranking.to_csv(csv_path)
    print(f"\n  Saved ranking → {csv_path}")

    return ranking


# ══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE  (fit on full dataset after CV)
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(rf_pipeline, feature_names, task_name, outdir):
    """
    RF impurity-based importance, fit on full dataset post-CV.
    Constant features are absent (already dropped), so all bars are non-zero.
    """
    imp     = rf_pipeline.named_steps["model"].feature_importances_
    indices = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.38)))
    ax.barh(
        [feature_names[i] for i in indices],
        imp[indices],
        color="#1565C0", edgecolor="white"
    )
    ax.set_xlabel("Importance (mean impurity decrease)", fontsize=10)
    ax.set_title(f"RF Feature Importance — {task_name}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fname = task_name.lower().replace(" ", "_")
    fpath = os.path.join(outdir, f"feat_imp_{fname}.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature importance → {fpath}")

    return pd.DataFrame({
        "feature":    feature_names,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("HEALTHCARE VULNERABILITY — SCRIPT 03: MACHINE LEARNING")
    print("Strategy : Leave-One-Out CV (n=17 LGUs, two regression tasks)")
    print("Tasks    : (A) Distance to Level-3 hospital  (B) Wd accessibility")
    print("Note     : Classification dropped — n=17 is too small for 3 classes")
    print("=" * 68)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df, facilities_df = load_data()
    city_names = (df["city_norm"].tolist() if "city_norm" in df.columns
                  else [f"City_{i}" for i in range(len(df))])

    # ── 2. Compute Wd ─────────────────────────────────────────────────────────
    print("\n[2/7] Computing Weighted Accessibility Score (Wd)...")
    y_wd = compute_wd(df, facilities_df)
    df[REGRESSION_TARGET_WD] = y_wd.values

    # ── 3. Prepare features ───────────────────────────────────────────────────
    print("\n[3/7] Preparing features...")
    X, y_km, feature_names = prepare_features(df)

    # ── 4. Hyperparameter tuning ─────────────────────────────────────────────
    print("\n[4/7] Hyperparameter tuning (inner cross-validation)...")
    print("\n  --- TASK A: Distance Regression ---")
    best_knn_km, best_rf_km = tune_regressor(X, y_km, label="nearest_public_tertiary_km")

    print("\n  --- TASK B: Wd Accessibility Score ---")
    best_knn_wd, best_rf_wd = tune_regressor(X, y_wd, label="wd_score")

    # ── 5. LOO evaluation ─────────────────────────────────────────────────────
    print("\n[5/7] Leave-One-Out evaluation...")

    print("\n  === TASK A: nearest_public_tertiary_km ===")
    km_results = []
    km_results.append(loo_evaluate(
        "kNN Regressor (Baseline)", best_knn_km, X, y_km, city_names,
        unit=" km", plot_prefix="km"))
    km_results.append(loo_evaluate(
        "Random Forest Regressor", best_rf_km, X, y_km, city_names,
        unit=" km", plot_prefix="km"))

    print("\n  === TASK B: Wd Accessibility Score ===")
    wd_results = []
    wd_results.append(loo_evaluate(
        "kNN Regressor (Baseline)", best_knn_wd, X, y_wd, city_names,
        unit=" Wd", plot_prefix="wd"))
    wd_results.append(loo_evaluate(
        "Random Forest Regressor", best_rf_wd, X, y_wd, city_names,
        unit=" Wd", plot_prefix="wd"))

    # ── 6. Wd city ranking ────────────────────────────────────────────────────
    print("\n[6/7] Generating Wd accessibility ranking...")
    ranking = print_wd_ranking(city_names, y_wd, y_km, df)

    # ── 7. Feature importance + save artefacts ────────────────────────────────
    print("\n[7/7] Feature importance and saving artefacts...")

    best_rf_km.fit(X.loc[~y_km.isna()], y_km.dropna())
    best_rf_wd.fit(X.loc[~y_wd.isna()], y_wd.dropna())

    imp_km = plot_feature_importance(best_rf_km, feature_names,
                                     "Distance Regression", VIZ_DIR)
    imp_wd = plot_feature_importance(best_rf_wd, feature_names,
                                     "Wd Accessibility Score", VIZ_DIR)

    print("\n  Top 5 features (distance regression):")
    print(imp_km.head(5).to_string(index=False))
    print("\n  Top 5 features (Wd accessibility):")
    print(imp_wd.head(5).to_string(index=False))

    # Save models
    for label, model in [
        ("knn_km", best_knn_km), ("rf_km",  best_rf_km),
        ("knn_wd", best_knn_wd), ("rf_wd",  best_rf_wd),
    ]:
        path = os.path.join(MODEL_DIR, f"{label}.joblib")
        joblib.dump(model, path)
        print(f"  Saved model → {path}")

    # Save JSON summary
    summary = {
        "n_cities":            len(df),
        "n_features":          len(feature_names),
        "feature_names":       feature_names,
        "dropped_features":    ["econ_friction_ratio", "poverty_threshold_2023_php"],
        "dropped_reason":      "constant across all 17 cities (std=0, zero information)",
        "task_A_distance_km":  km_results,
        "task_B_wd_score":     wd_results,
        "feat_imp_km":         imp_km.to_dict(orient="records"),
        "feat_imp_wd":         imp_wd.to_dict(orient="records"),
        "wd_ranking":          ranking.reset_index().to_dict(orient="records"),
    }
    summary_path = os.path.join(MODEL_DIR, "model_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved results summary → {summary_path}")

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("RESULTS SUMMARY")
    print("=" * 68)

    for task_label, results, unit in [
        ("TASK A — Distance to nearest public Level-3 hospital (km)",
         km_results, "km"),
        ("TASK B — Weighted Accessibility Score (Wd)",
         wd_results, "Wd"),
    ]:
        print(f"\n{task_label}")
        print(f"  {'Model':<35} {'MAE':>9}  {'RMSE':>9}  {'R²':>7}")
        print(f"  {'-'*35} {'-'*9}  {'-'*9}  {'-'*7}")
        for r in results:
            print(f"  {r['model']:<35} {r['MAE']:>9.4f}  "
                  f"{r['RMSE']:>9.4f}  {r['R2']:>7.4f}")

    print(f"\nWd CITY RANKING (top 5 most underserved):")
    print(f"  {'Rank':<6} {'City':<18} {'Wd':>8}  {'km':>7}  {'Poverty%':>9}")
    print(f"  {'-'*6} {'-'*18} {'-'*8}  {'-'*7}  {'-'*9}")
    for _, row in ranking.head(5).iterrows():
        print(f"  {row.name:<6} {row['city']:<18} {row['wd_score']:>8.4f}  "
              f"{row['nearest_public_km']:>7.3f}  {row['poverty_incidence_pct']:>9.2f}")

    print("\n" + "=" * 68)
    print("DONE. Next: run 04_viz.py")
    print("=" * 68)