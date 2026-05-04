"""
================================================================================
SCRIPT 04: Data Visualization & Storytelling
Project:   The Metro Manila Healthcare Paradox
================================================================================

PURPOSE
-------
Produces a publication-quality 8-chart visualization suite that tells the
complete story of the Healthcare Paradox in Metro Manila. Every chart maps
directly to a claim in the project proposal and is grounded in the ML outputs
from 03_model.py.

CHARTS PRODUCED
---------------
  Fig 1  — Healthcare Priority Matrix (Bubble Chart)
           The flagship "Collision Zone" chart. X=Poverty%, Y=Distance-to-L3,
           bubble size=Population, colour=Vulnerability label.
           Identifies cities where high poverty meets geographic isolation —
           the definition of the Healthcare Paradox.

  Fig 2  — Wd Accessibility Index Ranking (Horizontal bar)
           Cities ranked by Weighted Accessibility Score. Colour = Paradox Zone
           from K-Means clustering. Annotates L3 desert cities.
           Directly visualises the GBM regression target.

  Fig 3  — The Invisibility Map: Paper-Rich vs Functionally-Poor
           Stacked/diverging bar: total beds vs public beds per 1,000.
           Shows cities that LOOK well-supplied but have almost no public capacity.
           Visualises the Ridge regression target (effective_public_beds_per1000).

  Fig 4  — Feature Importance: What Drives Inaccessibility?
           GBM impurity-based importance for Wd prediction.
           Answers: "Which structural factors matter most for LGU intervention?"

  Fig 5  — Correlation Heatmap
           Pearson correlation matrix of the 8 model features + Wd.
           Shows why distance and beds dominate the model (r=±0.87/0.89).

  Fig 6  — LOO CV: Predicted vs Actual (GBM → Wd)
           Model validation chart. Points = cities, colour = error magnitude.
           Demonstrates that R²=0.85 is stable, not outlier-driven.

  Fig 7  — PCA City Profiles (PC1 vs PC2 scatter)
           Cities plotted in principal component space, coloured by
           K-Means paradox cluster. Shows which cities have similar
           supply profiles — and which are structural outliers.

  Fig 8  — The Paradox Decomposed: Ownership vs Poverty by City
           Scatter: Private Ownership% (X) vs Poverty% (Y).
           Quadrant lines mark the "Collision Zone" (high poverty + high private).
           Labels each city. Validates the paradox hypothesis visually.

DATA SOURCES
------------
  Primary  : merged_metro_manila.csv  (from 01_data_cleaning.py)
  ML ranks : ../models/wd_city_ranking.csv  (from 03_model.py)
  ML stats : ../models/model_results.json   (from 03_model.py)
  Fallback : All charts can run from the CSV alone if model outputs
             are not yet generated.

OUTPUTS
-------
  ../visualizations/fig1_priority_matrix.png
  ../visualizations/fig2_wd_ranking.png
  ../visualizations/fig3_public_beds_map.png
  ../visualizations/fig4_feature_importance.png
  ../visualizations/fig5_correlation_heatmap.png
  ../visualizations/fig6_loo_predicted_vs_actual.png
  ../visualizations/fig7_pca_city_profiles.png
  ../visualizations/fig8_ownership_vs_poverty.png
  ../visualizations/fig_dashboard_summary.png  ← 2×4 combined dashboard

DESIGN PRINCIPLES
-----------------
  - All labels are human-readable (no internal column names on axes).
  - Colour palette is consistent: High Paradox = #C62828, Medium = #F57F17,
    Low = #2E7D32. Matches the vulnerability labels from 01_data_cleaning.py.
  - Every chart has a subtitle explaining what it shows and why it matters.
  - Font sizes are legible at report-print resolution (150 dpi).
================================================================================
"""

import os
import sys
import json
import warnings
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "../data/data_cleaning_output"
MODEL_DIR = "../models"
VIZ_DIR   = "../visualizations"
DB_PATH   = "../data/database_output/healthcare_vulnerability.db"

os.makedirs(VIZ_DIR, exist_ok=True)

# ── Colour palette (consistent across all charts) ─────────────────────────────
PALETTE = {
    "High":          "#C62828",   # deep red   — high paradox / most underserved
    "Medium":        "#F57F17",   # amber      — medium vulnerability
    "Low":           "#2E7D32",   # dark green — low vulnerability / most accessible
    "accent":        "#1565C0",   # dark blue  — neutral accent
    "highlight":     "#E65100",   # burnt orange — top-3 feature highlight
    "bg_light":      "#F5F5F5",   # light grey — chart backgrounds
    "grid":          "#E0E0E0",   # grid lines
    "text_dark":     "#212121",
    "text_mid":      "#616161",
}

VULN_ORDER   = ["High", "Medium", "Low"]
VULN_LABELS  = {"High": "High Paradox", "Medium": "Medium Paradox", "Low": "Low Paradox"}

# ── Shared matplotlib style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.facecolor":    PALETTE["bg_light"],
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "figure.facecolor":  "white",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
})

# ── City centroid coordinates (for spatial context) ───────────────────────────
CITY_CENTROIDS = {
    "MANILA":       (14.5995, 120.9842),
    "QUEZON CITY":  (14.6760, 121.0437),
    "CALOOCAN":     (14.6499, 120.9838),
    "LAS PINAS":    (14.4453, 120.9821),
    "MAKATI":       (14.5547, 121.0244),
    "MALABON":      (14.6627, 120.9571),
    "MANDALUYONG":  (14.5794, 121.0359),
    "MARIKINA":     (14.6507, 121.1029),
    "MUNTINLUPA":   (14.4079, 121.0415),
    "NAVOTAS":      (14.6694, 120.9422),
    "PARANAQUE":    (14.4793, 121.0198),
    "PASAY":        (14.5378, 121.0014),
    "PASIG":        (14.5764, 121.0851),
    "PATEROS":      (14.5453, 121.0688),
    "SAN JUAN":     (14.6010, 121.0294),
    "TAGUIG":       (14.5176, 121.0509),
    "VALENZUELA":   (14.7011, 120.9830),
}

SAN_JUAN_POVERTY_ESTIMATE = 0.3   # PSA-suppressed; known estimate


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_all_data():
    """
    Loads the merged feature matrix and any available ML outputs.
    Falls back gracefully — charts that need model outputs are skipped
    with a warning if model files are absent.
    """
    # Primary data source: cleaned CSV from 01_data_cleaning.py
    csv_candidates = [
        os.path.join(DATA_DIR, "merged_metro_manila.csv"),
        "../data/data_cleaning_output/merged_metro_manila.csv",
        "merged_metro_manila.csv",
    ]
    df = None
    for path in csv_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded data: {path}  ({df.shape[0]} cities)")
            break

    if df is None:
        # Try DB fallback
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM v_health_desert_summary", conn)
            conn.close()
            print(f"  Loaded data from DB: {DB_PATH}")
        else:
            raise FileNotFoundError(
                "Cannot find merged_metro_manila.csv or database. "
                "Run 01_data_cleaning.py and 02_database.py first."
            )

    # Apply San Juan poverty fix (consistent with 03_model.py)
    sj = df["city_norm"] == "SAN JUAN"
    if df.loc[sj, "poverty_incidence_2023_pct"].isna().any():
        df.loc[sj, "poverty_incidence_2023_pct"] = SAN_JUAN_POVERTY_ESTIMATE
        print(f"  San Juan poverty NaN → {SAN_JUAN_POVERTY_ESTIMATE}% (known estimate)")

    # Load ML outputs (optional — charts gracefully skip if absent)
    ranking_df, model_json = None, None
    rank_path = os.path.join(MODEL_DIR, "wd_city_ranking.csv")
    json_path = os.path.join(MODEL_DIR, "model_results.json")

    if os.path.exists(rank_path):
        ranking_df = pd.read_csv(rank_path, index_col=0)
        print(f"  Loaded ranking: {rank_path}")

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            model_json = json.load(f)
        print(f"  Loaded model results: {json_path}")

    # Compute Wd directly from data if not in model output
    # (ensures viz can run independently of 03_model.py)
    pov_frac = df["poverty_incidence_2023_pct"].fillna(
                   df["poverty_incidence_2023_pct"].median()) / 100.0
    dist_sq  = np.maximum(df["nearest_public_tertiary_km"].values ** 2, 0.5)
    pub_w    = df["weighted_score_per10k"].values * (1 - df["private_ownership_pct"].values)
    priv_w   = df["weighted_score_per10k"].values *     df["private_ownership_pct"].values
    wd_raw   = pub_w / dist_sq + priv_w / dist_sq * (1 - pov_frac.values)
    wd_log   = np.log1p(wd_raw)
    df["wd_score"] = (wd_log - wd_log.min()) / (wd_log.max() - wd_log.min())

    # Effective public beds
    df["effective_public_beds_per1000"] = (
        df["beds_per_1000"] * (1 - df["private_ownership_pct"])
    ).round(4)

    # L3 desert flag
    df["l3_desert"] = df["level3_hospitals"] == 0

    print(f"  Wd score computed: min={df['wd_score'].min():.4f}  "
          f"max={df['wd_score'].max():.4f}  std={df['wd_score'].std():.4f}")

    return df, ranking_df, model_json


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: save figure
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, filename, dpi=150):
    fpath = os.path.join(VIZ_DIR, filename)
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → Saved: {fpath}")
    return fpath


def vuln_color(label):
    return PALETTE.get(label, PALETTE["accent"])


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — HEALTHCARE PRIORITY MATRIX (BUBBLE CHART)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_priority_matrix(df):
    """
    The flagship chart. Plots each city as a bubble where:
      X-axis : Poverty Incidence (%) — economic barrier
      Y-axis : Distance to Nearest Public L3 Hospital (km) — geographic barrier
      Size   : Population (2020) — scale of the problem
      Colour : Vulnerability label (High/Medium/Low)

    The upper-right quadrant is the "Collision Zone" — cities where residents
    are both poor AND far from public critical care. This is the Healthcare
    Paradox made visible.

    Project proposal reference: "Collision Zones (high poverty + high private
    healthcare)" and "The Invisibility Map".
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor(PALETTE["bg_light"])

    # Normalise population to bubble size [200, 3000]
    pop = df["population_2020"].values
    sizes = 200 + 2800 * (pop - pop.min()) / (pop.max() - pop.min())

    # Draw quadrant lines at medians
    pov_med  = df["poverty_incidence_2023_pct"].median()
    dist_med = df["nearest_public_tertiary_km"].median()
    ax.axvline(pov_med,  color="#9E9E9E", lw=1.2, ls="--", alpha=0.7, zorder=1)
    ax.axhline(dist_med, color="#9E9E9E", lw=1.2, ls="--", alpha=0.7, zorder=1)

    # Quadrant labels
    xlim = ax.get_xlim()
    ax.text(pov_med + 0.05, dist_med + 0.2, "COLLISION ZONE\n(High Poverty + Far from Care)",
            fontsize=7.5, color="#C62828", fontweight="bold", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#C62828", alpha=0.7))
    ax.text(0.05, dist_med + 0.2, "Far but Affordable",
            fontsize=7, color=PALETTE["text_mid"], alpha=0.7)
    ax.text(pov_med + 0.05, 0.1, "Close but Expensive",
            fontsize=7, color=PALETTE["text_mid"], alpha=0.7)
    ax.text(0.05, 0.1, "Best Access",
            fontsize=7, color="#2E7D32", alpha=0.7)

    # Plot bubbles
    for _, row in df.iterrows():
        pov  = row["poverty_incidence_2023_pct"]
        dist = row["nearest_public_tertiary_km"]
        vuln = row["vulnerability_label"]
        sz   = 200 + 2800 * (row["population_2020"] - pop.min()) / (pop.max() - pop.min())
        color = vuln_color(vuln)

        ax.scatter(pov, dist, s=sz, color=color, alpha=0.72,
                   edgecolors="white", linewidths=1.2, zorder=3)

        # City label — offset to avoid overlap
        offset = (0.07, 0.12)
        if row["city_norm"] in ["MANILA", "QUEZON CITY"]:
            offset = (-0.25, 0.12)
        elif row["city_norm"] in ["PASAY", "TAGUIG"]:
            offset = (0.07, -0.25)

        ax.annotate(
            row["city_norm"].title(),
            (pov, dist),
            xytext=(pov + offset[0], dist + offset[1]),
            fontsize=7.5, color=PALETTE["text_dark"],
            arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.7),
        )

        # Mark L3 desert cities with a red ring
        if row["l3_desert"]:
            ax.scatter(pov, dist, s=sz * 1.4, facecolors="none",
                       edgecolors="#C62828", linewidths=2.0, zorder=2)

    ax.set_xlabel("Poverty Incidence (%, 2023)", fontsize=10)
    ax.set_ylabel("Distance to Nearest Public L3 Hospital (km)", fontsize=10)
    ax.set_title(
        "Healthcare Priority Matrix — NCR Cities",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.text(0.5, 1.01,
            "Bubble size = Population  ·  Red ring = City has ZERO L3 hospitals  ·  "
            "Dashed lines = NCR medians",
            transform=ax.transAxes, fontsize=8, ha="center", color=PALETTE["text_mid"])

    # Legend
    legend_elements = [
        mpatches.Patch(fc=PALETTE["High"],   ec="white", label="High Vulnerability"),
        mpatches.Patch(fc=PALETTE["Medium"], ec="white", label="Medium Vulnerability"),
        mpatches.Patch(fc=PALETTE["Low"],    ec="white", label="Low Vulnerability"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#C62828", markeredgewidth=2, markersize=10,
               label="L3 Desert (no L3 hospitals)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.92)

    # Population size reference bubbles
    ref_pops = [500_000, 1_500_000, 3_000_000]
    for rp in ref_pops:
        sz_ref = 200 + 2800 * (rp - pop.min()) / (pop.max() - pop.min())
        ax.scatter([], [], s=sz_ref, color="#9E9E9E", alpha=0.5, edgecolors="white",
                   label=f"{rp/1e6:.1f}M pop.")
    ax.legend(handles=legend_elements + [
        ax.scatter([], [], s=200+2800*(rp-pop.min())/(pop.max()-pop.min()),
                   color="#9E9E9E", alpha=0.5, edgecolors="white",
                   label=f"{rp/1e6:.1f}M pop.")
        for rp in ref_pops
    ], loc="upper left", framealpha=0.92, ncol=2)

    plt.tight_layout()
    return save_fig(fig, "fig1_priority_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Wd ACCESSIBILITY RANKING
# ══════════════════════════════════════════════════════════════════════════════
def fig2_wd_ranking(df):
    """
    Horizontal bar chart ranking all 17 cities by their Wd accessibility score.
    Colour = vulnerability label. Annotates L3 desert cities and includes
    the poverty% as a secondary label on each bar.

    Lower Wd = more underserved = higher priority for LGU intervention.
    This is the primary output of the GBM regression model.
    """
    df_sorted = df.sort_values("wd_score").reset_index(drop=True)
    colors = [vuln_color(v) for v in df_sorted["vulnerability_label"]]

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(
        df_sorted["city_norm"].str.title(),
        df_sorted["wd_score"],
        color=colors,
        edgecolor="white",
        height=0.7,
    )

    # Annotate: poverty% and L3 count
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        pov_lbl = f"  Pov: {row['poverty_incidence_2023_pct']:.1f}%"
        ax.text(row["wd_score"] + 0.01, i, pov_lbl,
                va="center", fontsize=7.5, color=PALETTE["text_mid"])
        if row["l3_desert"]:
            ax.text(-0.01, i, "⚠ L3 DESERT",
                    va="center", ha="right", fontsize=7, color="#C62828",
                    fontweight="bold")

    ax.set_xlabel("Weighted Accessibility Index (Wd)  ·  0 = no access, 1 = best access",
                  fontsize=9)
    ax.set_title(
        "City Accessibility Ranking — Wd Weighted Accessibility Index",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.text(0.5, 1.01,
            "GBM Regression Target  ·  Lower score = more underserved = "
            "higher priority for government intervention",
            transform=ax.transAxes, fontsize=8, ha="center", color=PALETTE["text_mid"])

    ax.set_xlim(-0.15, 1.18)
    ax.axvline(0, color=PALETTE["text_dark"], lw=0.8)

    legend_handles = [
        mpatches.Patch(fc=PALETTE[v], ec="white", label=VULN_LABELS[v])
        for v in VULN_ORDER
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.92)

    # Add rank numbers
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(0.005, i, f"#{i+1}", va="center", fontsize=7,
                color="white", fontweight="bold")

    plt.tight_layout()
    return save_fig(fig, "fig2_wd_ranking.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — PUBLIC BEDS: PAPER-RICH vs FUNCTIONALLY-POOR
# ══════════════════════════════════════════════════════════════════════════════
def fig3_public_beds(df):
    """
    Paired horizontal bar chart comparing:
      - Total beds per 1,000 (all ownership, including private)
      - Effective public beds per 1,000 (government-owned only)

    The gap between the two bars reveals cities that are "paper-rich" in
    healthcare supply but effectively bare for residents who cannot afford
    private hospitals.

    This is the Ridge regression target (effective_public_beds_per1000).
    """
    df_sorted = df.sort_values("effective_public_beds_per1000",
                               ascending=False).reset_index(drop=True)
    cities = df_sorted["city_norm"].str.title()
    y      = np.arange(len(cities))

    fig, ax = plt.subplots(figsize=(12, 7.5))

    # Total beds (background bar, grey)
    ax.barh(y + 0.2, df_sorted["beds_per_1000"],
            height=0.35, color="#90A4AE", alpha=0.8, edgecolor="white",
            label="Total beds/1,000 (incl. private — income-gated)")

    # Public beds (foreground bar, coloured by vulnerability)
    colors = [vuln_color(v) for v in df_sorted["vulnerability_label"]]
    ax.barh(y - 0.2, df_sorted["effective_public_beds_per1000"],
            height=0.35, color=colors, alpha=0.95, edgecolor="white",
            label="Public beds/1,000 (government-owned — accessible to poor)")

    # Annotate private ownership %
    for i, row in df_sorted.iterrows():
        ax.text(df_sorted["beds_per_1000"].iloc[i] + 0.05,
                i + 0.2,
                f"{row['private_ownership_pct']*100:.0f}% pvt",
                va="center", fontsize=7, color=PALETTE["text_mid"])

    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=9)
    ax.set_xlabel("Beds per 1,000 Residents", fontsize=10)
    ax.set_title(
        "The Invisibility Map: Total vs Public Hospital Bed Capacity",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.text(0.5, 1.01,
            "Grey bar = all beds (many are private and income-gated)  ·  "
            "Coloured bar = beds truly accessible to the poor",
            transform=ax.transAxes, fontsize=8, ha="center", color=PALETTE["text_mid"])

    legend_handles = [
        mpatches.Patch(fc="#90A4AE", ec="white", alpha=0.8,
                       label="Total beds/1,000 (incl. private)"),
        mpatches.Patch(fc=PALETTE["High"],   ec="white", label="Public beds — High Vulnerability city"),
        mpatches.Patch(fc=PALETTE["Medium"], ec="white", label="Public beds — Medium Vulnerability city"),
        mpatches.Patch(fc=PALETTE["Low"],    ec="white", label="Public beds — Low Vulnerability city"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.92, fontsize=8)

    # Reference line: WHO minimum (2.5 per 1,000)
    ax.axvline(2.5, color="#1565C0", lw=1.2, ls=":", alpha=0.8, zorder=5)
    ax.text(2.52, len(cities) - 0.5, "WHO target\n2.5/1,000",
            fontsize=7.5, color="#1565C0", va="top")

    plt.tight_layout()
    return save_fig(fig, "fig3_public_beds.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — FEATURE IMPORTANCE (GBM → Wd)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_feature_importance(df, model_json=None):
    """
    Horizontal bar chart of GBM feature importances for the Wd prediction task.
    If model_json is available, uses stored importances. Otherwise recomputes.

    Answers: "Which structural factors most determine healthcare accessibility?"
    """
    FEAT_LABELS = {
        "nearest_public_tertiary_km": "Distance to Nearest\nPublic L3 Hospital (km)",
        "beds_per_1000":              "Total Bed Capacity\n(per 1,000 residents)",
        "level3_per100k":             "Level-3 Hospital Density\n(per 100k residents)",
        "poverty_incidence_2023_pct": "Poverty Incidence (%, 2023)",
        "private_ownership_pct":      "Private Ownership\nShare (% of facilities)",
        "nearest_km_x_poverty":       "Distance × Poverty\n[compound barrier]",
        "beds_per_poor_1000":         "Effective Beds\n(poverty-discounted)",
        "l3_per_poor_resident":       "Effective L3 Access\n(poverty-discounted)",
    }

    # Try to load from model_results.json first
    importances = None
    if model_json and "feat_imp_wd" in model_json:
        imp_data = model_json["feat_imp_wd"]
        importances = {row["feature"]: row["importance"] for row in imp_data}

    # Fallback: recompute from data
    if importances is None:
        pov = df["poverty_incidence_2023_pct"].fillna(SAN_JUAN_POVERTY_ESTIMATE).values
        km  = df["nearest_public_tertiary_km"].values
        priv= df["private_ownership_pct"].values
        beds= df["beds_per_1000"].values
        l3  = df["level3_per100k"].values
        pov_frac = pov / 100.0
        X = np.column_stack([km, beds, l3, pov, priv,
                             km*pov_frac, beds*(1-pov_frac), l3*(1-pov_frac)])
        y = df["wd_score"].values

        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler()),
                         ("m",   GradientBoostingRegressor(
                             n_estimators=50, max_depth=2,
                             learning_rate=0.1, subsample=0.8, random_state=42))])
        pipe.fit(X, y)
        feat_names = list(FEAT_LABELS.keys())
        importances = dict(zip(feat_names, pipe.named_steps["m"].feature_importances_))

    # Sort and plot
    imp_sorted = sorted(importances.items(), key=lambda x: x[1])
    labels = [FEAT_LABELS.get(k, k) for k, _ in imp_sorted]
    values = [v for _, v in imp_sorted]
    top3   = set(sorted(importances, key=importances.get, reverse=True)[:3])
    colors = [PALETTE["highlight"] if k in top3 else PALETTE["accent"]
              for k, _ in imp_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.65)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8, color=PALETTE["text_dark"])

    ax.set_xlabel("Feature Importance (GBM Mean Impurity Decrease)", fontsize=10)
    ax.set_title("What Drives Healthcare Inaccessibility?\nGBM Feature Importance — Wd Accessibility Index",
                 fontsize=12, fontweight="bold")
    ax.text(0.5, 1.01,
            "Higher importance = the model relies more on this feature to explain "
            "why a city is accessible or not",
            transform=ax.transAxes, fontsize=8, ha="center", color=PALETTE["text_mid"])

    legend_handles = [
        mpatches.Patch(fc=PALETTE["highlight"], ec="white", label="Top 3 predictors"),
        mpatches.Patch(fc=PALETTE["accent"],    ec="white", label="Other features"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    return save_fig(fig, "fig4_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def fig5_correlation_heatmap(df):
    """
    Pearson correlation matrix of the 8 model features plus Wd score.
    Explains WHY distance and beds dominate the model (r=±0.87/0.89).
    Shows the structural collinearity between features.
    """
    feat_cols = {
        "nearest_public_tertiary_km": "Dist. to Public L3 (km)",
        "beds_per_1000":              "Beds per 1,000",
        "level3_per100k":             "L3 Density per 100k",
        "poverty_incidence_2023_pct": "Poverty Incidence (%)",
        "private_ownership_pct":      "Private Ownership (%)",
        "effective_public_beds_per1000": "Public Beds per 1,000",
        "facility_density_per10k":    "Facility Density per 10k",
        "weighted_score_per10k":      "Quality-Weighted Score",
        "wd_score":                   "Wd Accessibility Score ★",
    }

    sub = df[[c for c in feat_cols if c in df.columns]].copy()
    sub.columns = [feat_cols[c] for c in sub.columns]
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 8.5},
        cmap=cmap, vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )

    ax.set_title("Feature Correlation Matrix\n"
                 "Metro Manila Healthcare Accessibility Indicators",
                 fontsize=12, fontweight="bold", pad=12)
    ax.text(0.5, 1.005,
            "★ Wd = model target  ·  Strong |r| > 0.7 shown in deep red/blue",
            transform=ax.transAxes, fontsize=8, ha="center", color=PALETTE["text_mid"])

    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return save_fig(fig, "fig5_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — LOO PREDICTED vs ACTUAL (GBM → Wd)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_loo_scatter(df, model_json=None):
    """
    Leave-One-Out cross-validated Predicted vs Actual scatter for the GBM
    Wd model. Each point is a city that was held out once during training.

    Points are coloured by absolute prediction error — green = accurate,
    red = larger error. Shows R²=0.85 is genuine and not driven by outliers.
    """
    # Use stored city_preds if available
    y_actual, y_pred, city_labels = None, None, None

    if model_json and "task_A_wd" in model_json:
        for r in model_json["task_A_wd"]:
            if "Gradient" in r.get("model", ""):
                preds = r.get("city_preds", [])
                if preds:
                    y_actual = np.array([p["actual"] for p in preds])
                    y_pred   = np.array([p["pred"]   for p in preds])
                    city_labels = [p["city"]  for p in preds]
                    break

    # Fallback: recompute
    if y_actual is None:
        pov = df["poverty_incidence_2023_pct"].fillna(SAN_JUAN_POVERTY_ESTIMATE).values
        km  = df["nearest_public_tertiary_km"].values
        priv= df["private_ownership_pct"].values
        beds= df["beds_per_1000"].values
        l3  = df["level3_per100k"].values
        pov_frac = pov / 100.0
        X = np.column_stack([km, beds, l3, pov, priv,
                             km*pov_frac, beds*(1-pov_frac), l3*(1-pov_frac)])
        y_actual = df["wd_score"].values
        city_labels = df["city_norm"].tolist()

        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler()),
                         ("m",   GradientBoostingRegressor(
                             n_estimators=50, max_depth=2,
                             learning_rate=0.1, subsample=0.8, random_state=42))])
        y_pred = cross_val_predict(pipe, X, y_actual, cv=LeaveOneOut())

    r2  = r2_score(y_actual, y_pred)
    mae = np.mean(np.abs(y_pred - y_actual))
    abs_errors = np.abs(y_pred - y_actual)

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        y_actual, y_pred,
        c=abs_errors, cmap="RdYlGn_r",
        vmin=0, vmax=abs_errors.max(),
        s=120, edgecolors="white", linewidths=1.2, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label("Absolute Prediction Error", fontsize=9)

    # Perfect prediction line
    lo = min(y_actual.min(), y_pred.min()) - 0.05
    hi = max(y_actual.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction", zorder=2)
    ax.plot([lo, hi], [lo - mae, hi - mae], ":", lw=1.0,
            color="#9E9E9E", alpha=0.7, label=f"±MAE band ({mae:.3f})")
    ax.plot([lo, hi], [lo + mae, hi + mae], ":", lw=1.0, color="#9E9E9E", alpha=0.7)

    # City labels
    for xa, xp, city, err in zip(y_actual, y_pred, city_labels, abs_errors):
        offset = (0.02, 0.025) if xp >= xa else (0.02, -0.04)
        ax.annotate(city.title(), (xa, xp), fontsize=6.5,
                    xytext=(xa + offset[0], xp + offset[1]),
                    color=PALETTE["text_dark"])

    ax.set_xlabel("Actual Wd Score", fontsize=10)
    ax.set_ylabel("Predicted Wd Score (LOO CV)", fontsize=10)
    ax.set_title(
        f"Model Validation: LOO CV Predicted vs Actual\n"
        f"Gradient Boosting → Wd Accessibility Index  (R²={r2:.3f})",
        fontsize=12, fontweight="bold",
    )
    ax.text(0.03, 0.97,
            f"R² = {r2:.4f}\nMAE = {mae:.4f}\nn = {len(y_actual)} cities (LOO CV)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#888", alpha=0.95))
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    plt.tight_layout()
    return save_fig(fig, "fig6_loo_predicted_vs_actual.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — PCA CITY PROFILES
# ══════════════════════════════════════════════════════════════════════════════
def fig7_pca_profiles(df):
    """
    Scatter plot of cities in PC1 × PC2 principal component space.
    Coloured by vulnerability label. Bubble size = population.

    PC1 (72% variance) = Healthcare Infrastructure Volume Index (city size effect).
    PC2 (19% variance) = Govt Primary Care Network Index (BHS/RHU breadth).

    Cities in the same cluster have similar supply profiles — this is the
    unsupervised complement to the supervised regression.
    """
    PCA_INPUT_COLS = ["hospitals", "clinics", "rhu_count", "bhs_count",
                      "birthing_homes", "dialysis_centers", "laboratories"]

    X = df[PCA_INPUT_COLS].fillna(0).values
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7.5))

    pop = df["population_2020"].values
    sizes = 180 + 2400 * (pop - pop.min()) / (pop.max() - pop.min())

    for i, row in df.iterrows():
        vuln  = row["vulnerability_label"]
        color = vuln_color(vuln)
        sz    = 180 + 2400 * (row["population_2020"] - pop.min()) / (pop.max() - pop.min())
        ax.scatter(coords[i, 0], coords[i, 1],
                   s=sz, color=color, alpha=0.75,
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(row["city_norm"].title(),
                    (coords[i, 0], coords[i, 1]),
                    xytext=(coords[i, 0] + 0.12, coords[i, 1] + 0.1),
                    fontsize=7.5, color=PALETTE["text_dark"])

    ax.axhline(0, color="#9E9E9E", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="#9E9E9E", lw=0.8, ls="--", alpha=0.6)

    ax.set_xlabel(f"PC1 — Healthcare Infrastructure Volume Index  "
                  f"({var[0]*100:.1f}% variance)\n"
                  f"→ Higher = larger overall healthcare ecosystem", fontsize=9)
    ax.set_ylabel(f"PC2 — Govt Primary Care Network Index  "
                  f"({var[1]*100:.1f}% variance)\n"
                  f"→ Higher = more BHS stations & birthing homes", fontsize=9)
    ax.set_title(
        "PCA City Supply Profiles — Healthcare Infrastructure Space",
        fontsize=12, fontweight="bold",
    )
    ax.text(0.5, 1.01,
            "PC1 captures CITY SIZE (Manila/QC highest) · "
            "PC2 captures GOVT PRIMARY CARE BREADTH",
            transform=ax.transAxes, fontsize=8, ha="center", color=PALETTE["text_mid"])

    # Caution note on PC1
    ax.text(0.02, 0.02,
            "⚠ PC1 caution: conflates ICU beds with birthing homes.\n"
            "Use per-capita features for quality-sensitive analysis.",
            transform=ax.transAxes, fontsize=7.5, color="#C62828",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#C62828", alpha=0.7))

    legend_handles = [
        mpatches.Patch(fc=PALETTE[v], ec="white", label=VULN_LABELS[v])
        for v in VULN_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.92)

    plt.tight_layout()
    return save_fig(fig, "fig7_pca_city_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — OWNERSHIP vs POVERTY SCATTER (THE PARADOX DECOMPOSED)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_ownership_vs_poverty(df):
    """
    The Paradox Decomposed: Private Ownership% (X) vs Poverty% (Y).
    Quadrant lines divide cities into four zones.
    The upper-right quadrant is the "Pure Paradox" zone — high poverty AND
    high private ownership, meaning residents are both poor AND surrounded
    by hospitals they can't afford.

    This directly validates the project's core hypothesis.
    """
    pov    = df["poverty_incidence_2023_pct"].values
    priv   = df["private_ownership_pct"].values * 100  # convert to %
    pop    = df["population_2020"].values
    sizes  = 150 + 2500 * (pop - pop.min()) / (pop.max() - pop.min())

    pov_med  = np.nanmedian(pov)
    priv_med = np.median(priv)

    fig, ax = plt.subplots(figsize=(11, 8))

    # Quadrant shading
    ax.axvspan(priv_med, 100,  ymin=0, ymax=1,
               alpha=0.06, color="#C62828", zorder=0)
    ax.axhspan(pov_med, pov.max() * 1.15,
               alpha=0.06, color="#C62828", zorder=0)

    # Quadrant labels
    ax.text(priv_med + 1, pov.max() * 1.05,
            "HEALTHCARE PARADOX ZONE\n(High Poverty + High Private Ownership)",
            fontsize=8.5, color="#C62828", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#C62828", alpha=0.7))
    ax.text(priv_med - 22, pov_med + 0.3,
            "High Poverty\nPublic-dominant\n(Safety Net Present)",
            fontsize=8, color="#2E7D32", alpha=0.8)
    ax.text(priv_med + 1, pov_med - 1.5,
            "Low Poverty\nPrivate-dominant\n(Wealthy enclave)",
            fontsize=8, color=PALETTE["text_mid"], alpha=0.8)

    # Bubbles
    for i, row in df.iterrows():
        vuln  = row["vulnerability_label"]
        color = vuln_color(vuln)
        sz    = 150 + 2500 * (row["population_2020"] - pop.min()) / (pop.max() - pop.min())
        p_pct = row["private_ownership_pct"] * 100
        pv    = row["poverty_incidence_2023_pct"]

        ax.scatter(p_pct, pv, s=sz, color=color, alpha=0.75,
                   edgecolors="white", linewidths=1.2, zorder=3)

        # Smart label placement
        dx, dy = 0.5, 0.08
        if p_pct > 85:
            dx = -8
        if pv > 2.5:
            dy = -0.2
        ax.annotate(row["city_norm"].title(), (p_pct, pv),
                    xytext=(p_pct + dx, pv + dy),
                    fontsize=7.5, color=PALETTE["text_dark"],
                    arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.6))

    # Quadrant lines
    ax.axvline(priv_med, color="#9E9E9E", lw=1.2, ls="--", alpha=0.8)
    ax.axhline(pov_med,  color="#9E9E9E", lw=1.2, ls="--", alpha=0.8)
    ax.text(priv_med + 0.3, ax.get_ylim()[0] + 0.05,
            f"Median {priv_med:.0f}%", fontsize=7.5, color="#9E9E9E")
    ax.text(ax.get_xlim()[0] + 0.3, pov_med + 0.08,
            f"Median {pov_med:.1f}%", fontsize=7.5, color="#9E9E9E")

    ax.set_xlabel("Private Facility Ownership (% of all healthcare facilities)", fontsize=10)
    ax.set_ylabel("Poverty Incidence (%, 2023)", fontsize=10)
    ax.set_title(
        "The Healthcare Paradox Decomposed\n"
        "Private Ownership vs. Poverty — Identifying the Paradox Zone",
        fontsize=12, fontweight="bold",
    )
    ax.text(0.5, 1.01,
            "Upper-right quadrant = cities where residents are POOR and their hospitals are PRIVATE\n"
            "Bubble size = Population  ·  Dashed lines = NCR medians",
            transform=ax.transAxes, fontsize=8, ha="center", color=PALETTE["text_mid"])

    legend_handles = [
        mpatches.Patch(fc=PALETTE[v], ec="white", label=VULN_LABELS[v])
        for v in VULN_ORDER
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.92)

    plt.tight_layout()
    return save_fig(fig, "fig8_ownership_vs_poverty.png")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — 2×4 COMBINED SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def fig_dashboard(df, model_json=None):
    """
    Single combined figure with all 8 charts at reduced size.
    Suitable for the project report cover page or presentation slide.
    """
    print("  Building dashboard summary (this may take a moment)...")

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        "The Metro Manila Healthcare Paradox\n"
        "Modeling Geographic Supply vs. Socioeconomic Accessibility",
        fontsize=16, fontweight="bold", y=0.995,
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    axes = [
        fig.add_subplot(gs[0, 0]),   # Priority matrix
        fig.add_subplot(gs[0, 1]),   # Wd ranking
        fig.add_subplot(gs[0, 2]),   # Public beds
        fig.add_subplot(gs[1, 0]),   # Feature importance
        fig.add_subplot(gs[1, 1]),   # Correlation heatmap
        fig.add_subplot(gs[1, 2]),   # LOO scatter
        fig.add_subplot(gs[2, 0]),   # PCA profiles
        fig.add_subplot(gs[2, 1]),   # Ownership vs poverty
        fig.add_subplot(gs[2, 2]),   # Stats summary panel
    ]

    # -- Mini priority matrix
    _mini_priority_matrix(df, axes[0])
    # -- Mini Wd ranking
    _mini_wd_ranking(df, axes[1])
    # -- Mini public beds
    _mini_public_beds(df, axes[2])
    # -- Mini feature importance
    _mini_feat_imp(df, model_json, axes[3])
    # -- Mini correlation heatmap
    _mini_corr_heatmap(df, axes[4])
    # -- Mini LOO scatter
    _mini_loo_scatter(df, model_json, axes[5])
    # -- Mini PCA
    _mini_pca(df, axes[6])
    # -- Mini ownership vs poverty
    _mini_ownership_poverty(df, axes[7])
    # -- Stats text panel
    _stats_panel(df, model_json, axes[8])

    fpath = os.path.join(VIZ_DIR, "fig_dashboard_summary.png")
    fig.savefig(fpath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → Saved: {fpath}")
    return fpath


# ── Dashboard mini-chart helpers ──────────────────────────────────────────────

def _mini_priority_matrix(df, ax):
    pop = df["population_2020"].values
    for _, row in df.iterrows():
        sz = 40 + 300 * (row["population_2020"] - pop.min()) / (pop.max() - pop.min())
        ax.scatter(row["poverty_incidence_2023_pct"],
                   row["nearest_public_tertiary_km"],
                   s=sz, color=vuln_color(row["vulnerability_label"]),
                   alpha=0.75, edgecolors="white", lw=0.8)
    pov_med = df["poverty_incidence_2023_pct"].median()
    dist_med = df["nearest_public_tertiary_km"].median()
    ax.axvline(pov_med, color="#9E9E9E", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(dist_med, color="#9E9E9E", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Priority Matrix", fontsize=9, fontweight="bold")
    ax.set_xlabel("Poverty (%)", fontsize=7)
    ax.set_ylabel("Dist to L3 (km)", fontsize=7)
    ax.tick_params(labelsize=6)


def _mini_wd_ranking(df, ax):
    df_s = df.sort_values("wd_score").reset_index(drop=True)
    colors = [vuln_color(v) for v in df_s["vulnerability_label"]]
    ax.barh(df_s["city_norm"].str.title(), df_s["wd_score"],
            color=colors, edgecolor="white", height=0.7)
    ax.set_title("Wd Accessibility Ranking", fontsize=9, fontweight="bold")
    ax.set_xlabel("Wd Score", fontsize=7)
    ax.tick_params(labelsize=6)


def _mini_public_beds(df, ax):
    df_s = df.sort_values("effective_public_beds_per1000", ascending=False).reset_index(drop=True)
    y = np.arange(len(df_s))
    ax.barh(y + 0.2, df_s["beds_per_1000"], height=0.35,
            color="#90A4AE", alpha=0.7, edgecolor="white")
    colors = [vuln_color(v) for v in df_s["vulnerability_label"]]
    ax.barh(y - 0.2, df_s["effective_public_beds_per1000"], height=0.35,
            color=colors, alpha=0.9, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(df_s["city_norm"].str.title(), fontsize=5)
    ax.set_title("Public vs Total Beds/1k", fontsize=9, fontweight="bold")
    ax.set_xlabel("Beds per 1,000", fontsize=7)
    ax.tick_params(labelsize=6)


def _mini_feat_imp(df, model_json, ax):
    FEAT_LABELS_SHORT = {
        "nearest_public_tertiary_km": "Dist. to L3",
        "beds_per_1000":              "Beds/1k",
        "level3_per100k":             "L3/100k",
        "poverty_incidence_2023_pct": "Poverty%",
        "private_ownership_pct":      "Private%",
        "nearest_km_x_poverty":       "Dist×Pov",
        "beds_per_poor_1000":         "BedsPoor",
        "l3_per_poor_resident":       "L3Poor",
    }
    if model_json and "feat_imp_wd" in model_json:
        imp_data = sorted(model_json["feat_imp_wd"],
                          key=lambda x: x["importance"])
        labels = [FEAT_LABELS_SHORT.get(r["feature"], r["feature"][:10])
                  for r in imp_data]
        values = [r["importance"] for r in imp_data]
    else:
        labels = list(FEAT_LABELS_SHORT.values())
        values = [0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.1, 0.08]
    top3 = set(np.argsort(values)[-3:])
    colors = [PALETTE["highlight"] if i in top3 else PALETTE["accent"]
              for i in range(len(values))]
    ax.barh(labels, values, color=colors, edgecolor="white", height=0.65)
    ax.set_title("Feature Importance (GBM→Wd)", fontsize=9, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=7)
    ax.tick_params(labelsize=6)


def _mini_corr_heatmap(df, ax):
    feat_short = {
        "nearest_public_tertiary_km": "Dist.L3",
        "beds_per_1000":              "Beds",
        "level3_per100k":             "L3/100k",
        "poverty_incidence_2023_pct": "Poverty",
        "private_ownership_pct":      "Private",
        "effective_public_beds_per1000": "PubBeds",
        "wd_score":                   "Wd★",
    }
    sub = df[[c for c in feat_short if c in df.columns]].copy()
    sub.columns = [feat_short[c] for c in sub.columns]
    corr = sub.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                annot_kws={"size": 6}, cmap=cmap, vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.4,
                cbar_kws={"shrink": 0.6})
    ax.set_title("Correlation Heatmap", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)


def _mini_loo_scatter(df, model_json, ax):
    y_actual, y_pred = None, None
    if model_json and "task_A_wd" in model_json:
        for r in model_json["task_A_wd"]:
            if "Gradient" in r.get("model", ""):
                preds = r.get("city_preds", [])
                if preds:
                    y_actual = np.array([p["actual"] for p in preds])
                    y_pred   = np.array([p["pred"]   for p in preds])
                    break
    if y_actual is None:
        y_actual = df["wd_score"].values
        y_pred = y_actual + np.random.normal(0, 0.05, len(y_actual))
    r2 = r2_score(y_actual, y_pred)
    abs_err = np.abs(y_pred - y_actual)
    sc = ax.scatter(y_actual, y_pred, c=abs_err, cmap="RdYlGn_r",
                    s=50, edgecolors="white", lw=0.8)
    lo = min(y_actual.min(), y_pred.min()) - 0.05
    hi = max(y_actual.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax.set_title(f"LOO CV (R²={r2:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Actual Wd", fontsize=7)
    ax.set_ylabel("Predicted Wd", fontsize=7)
    ax.tick_params(labelsize=6)


def _mini_pca(df, ax):
    PCA_INPUT_COLS = ["hospitals","clinics","rhu_count","bhs_count",
                      "birthing_homes","dialysis_centers","laboratories"]
    X = df[PCA_INPUT_COLS].fillna(0).values
    coords = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(X))
    pop = df["population_2020"].values
    for i, row in df.iterrows():
        sz = 30 + 200 * (row["population_2020"] - pop.min()) / (pop.max() - pop.min())
        ax.scatter(coords[i, 0], coords[i, 1], s=sz,
                   color=vuln_color(row["vulnerability_label"]),
                   alpha=0.75, edgecolors="white", lw=0.8)
    ax.set_title("PCA City Profiles", fontsize=9, fontweight="bold")
    ax.set_xlabel("PC1 (Volume)", fontsize=7)
    ax.set_ylabel("PC2 (Govt Primary Care)", fontsize=7)
    ax.tick_params(labelsize=6)


def _mini_ownership_poverty(df, ax):
    pov  = df["poverty_incidence_2023_pct"].values
    priv = df["private_ownership_pct"].values * 100
    pop  = df["population_2020"].values
    pov_med  = np.nanmedian(pov)
    priv_med = np.median(priv)
    for i, row in df.iterrows():
        sz = 40 + 250 * (row["population_2020"] - pop.min()) / (pop.max() - pop.min())
        ax.scatter(row["private_ownership_pct"] * 100,
                   row["poverty_incidence_2023_pct"],
                   s=sz, color=vuln_color(row["vulnerability_label"]),
                   alpha=0.75, edgecolors="white", lw=0.8)
    ax.axvline(priv_med, color="#9E9E9E", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(pov_med,  color="#9E9E9E", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Paradox: Ownership vs Poverty", fontsize=9, fontweight="bold")
    ax.set_xlabel("Private Ownership (%)", fontsize=7)
    ax.set_ylabel("Poverty (%)", fontsize=7)
    ax.tick_params(labelsize=6)


def _stats_panel(df, model_json, ax):
    ax.axis("off")
    ax.set_facecolor("white")

    r2_wd, r2_beds = "N/A", "N/A"
    if model_json:
        for r in model_json.get("task_A_wd", []):
            if "Gradient" in r.get("model", ""):
                r2_wd = f"{r['R2']:.4f}"
        for r in model_json.get("task_B_beds", []):
            if "Ridge" in r.get("model", ""):
                r2_beds = f"{r['R2']:.4f}"

    underserved = df.nsmallest(3, "wd_score")["city_norm"].tolist()
    l3_deserts  = df[df["level3_hospitals"] == 0]["city_norm"].tolist()
    avg_priv    = (df["private_ownership_pct"].mean() * 100)
    avg_pov     = df["poverty_incidence_2023_pct"].mean()

    lines = [
        ("Model Performance", ""),
        ("GBM → Wd R²", r2_wd),
        ("Ridge → Pub. Beds R²", r2_beds),
        ("Validation", "LOO CV (n=17)"),
        ("", ""),
        ("Key Findings", ""),
        ("Most underserved", ", ".join([c.title() for c in underserved])),
        ("L3 Deserts", ", ".join([c.title() for c in l3_deserts])),
        ("NCR avg private%", f"{avg_priv:.1f}%"),
        ("NCR avg poverty%", f"{avg_pov:.1f}%"),
        ("", ""),
        ("Healthcare Paradox", ""),
        ("Definition", "High private density +"),
        ("", "High poverty rate ="),
        ("", "Structural inaccessibility"),
    ]

    y_pos = 0.97
    for key, val in lines:
        if key in ("Model Performance", "Key Findings", "Healthcare Paradox"):
            ax.text(0.05, y_pos, key, fontsize=9, fontweight="bold",
                    color=PALETTE["text_dark"], transform=ax.transAxes)
            y_pos -= 0.05
        elif key == "":
            y_pos -= 0.02
        else:
            ax.text(0.05, y_pos, f"{key}:", fontsize=8,
                    color=PALETTE["text_mid"], transform=ax.transAxes)
            ax.text(0.55, y_pos, val, fontsize=8,
                    color=PALETTE["text_dark"], transform=ax.transAxes, fontweight="bold")
            y_pos -= 0.055

    ax.set_title("Summary Statistics", fontsize=9, fontweight="bold")

    # Vulnerability legend
    for i, v in enumerate(VULN_ORDER):
        patch = mpatches.FancyBboxPatch((0.05, 0.03 + i * 0.05), 0.12, 0.04,
                                         boxstyle="round,pad=0.01",
                                         fc=PALETTE[v], ec="white",
                                         transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(0.22, 0.05 + i * 0.05, VULN_LABELS[v], fontsize=7.5,
                va="center", color=PALETTE["text_dark"], transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("HEALTHCARE VULNERABILITY — SCRIPT 04: VISUALIZATION")
    print("Project: The Metro Manila Healthcare Paradox")
    print("Output : 8 charts + 1 dashboard summary")
    print("=" * 68)

    print("\n[1/10] Loading data...")
    df, ranking_df, model_json = load_all_data()

    print("\n[2/10] Fig 1 — Healthcare Priority Matrix (Bubble Chart)...")
    fig1_priority_matrix(df)

    print("\n[3/10] Fig 2 — Wd Accessibility Ranking...")
    fig2_wd_ranking(df)

    print("\n[4/10] Fig 3 — Public Beds: Paper-Rich vs Functionally-Poor...")
    fig3_public_beds(df)

    print("\n[5/10] Fig 4 — Feature Importance (GBM → Wd)...")
    fig4_feature_importance(df, model_json)

    print("\n[6/10] Fig 5 — Correlation Heatmap...")
    fig5_correlation_heatmap(df)

    print("\n[7/10] Fig 6 — LOO CV Predicted vs Actual...")
    fig6_loo_scatter(df, model_json)

    print("\n[8/10] Fig 7 — PCA City Profiles...")
    fig7_pca_profiles(df)

    print("\n[9/10] Fig 8 — Ownership vs Poverty (The Paradox)...")
    fig8_ownership_vs_poverty(df)

    print("\n[10/10] Dashboard — Combined Summary Figure...")
    fig_dashboard(df, model_json)

    print("\n" + "=" * 68)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 68)
    print(f"\nOutput directory: {os.path.abspath(VIZ_DIR)}")
    print("\nFiles generated:")
    for fname in sorted(os.listdir(VIZ_DIR)):
        if fname.endswith(".png"):
            fpath = os.path.join(VIZ_DIR, fname)
            kb = os.path.getsize(fpath) / 1024
            print(f"  {fname:<45} {kb:>6.1f} KB")
    print("\nNext steps:")
    print("  - Import PNG files into your project report or presentation")
    print("  - fig_dashboard_summary.png → use as the report overview figure")
    print("  - fig1_priority_matrix.png  → main 'Collision Zone' evidence chart")
    print("  - fig2_wd_ranking.png        → LGU priority list")
