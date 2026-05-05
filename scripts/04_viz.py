"""
================================================================================
SCRIPT 04: Data Visualization & Storytelling
Project:   The Metro Manila Healthcare Paradox
Courses:   Data Mining & Wrangling | Machine Learning | Data Viz & Storytelling
================================================================================

CHARTS
------
  Fig 1 — PCA Supply Profile: Scatter (PC1×PC2) + Colour-coded Loadings Table
  Fig 2 — PCA Scree Plot + Cumulative Variance with 80% threshold
  Fig 3 — Gradient Boosting LOO Predicted vs Actual (Wd score)
  Fig 4 — Feature Importance — best predictor highlighted in coral
  Fig 5 — Recommendation Heatmap — cities × indicators, sorted by vulnerability

COLOUR SCHEME
-------------
  Navy  #1B2A4A  structure, titles         Teal  #0D7377  accessible/positive
  Coral #E84855  urgency/underserved        Amber #F4A261  medium/caution
  Slate #64748B  neutral text/grid          Ice   #E2E8F0  gridlines/borders

DESIGN: zero spines everywhere, hardcoded label offsets, 150 DPI output.
================================================================================
"""

import os, sys, json, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "../data/data_cleaning_output"
MODEL_DIR = "../data/model_output"
VIZ_DIR   = "../data/visualization_output"
DB_PATH   = "../data/database_output/healthcare_vulnerability.db"
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Colour system ──────────────────────────────────────────────────────────────
C = {
    "navy":  "#1B2A4A", "teal":  "#0D7377", "coral": "#E84855",
    "amber": "#F4A261", "slate": "#64748B", "ice":   "#E2E8F0",
    "white": "#FFFFFF", "bg":    "#F8FAFC",
}
VULN_COLOR  = {"High": C["coral"], "Medium": C["amber"], "Low": C["teal"]}
VULN_LABELS = {"High": "High Vulnerability", "Medium": "Medium Vulnerability", "Low": "Low Vulnerability"}
# Backwards compat alias
PALETTE = {
    "High": C["coral"], "Medium": C["amber"], "Low": C["teal"],
    "accent": C["navy"], "highlight": C["coral"],
    "bg_light": C["bg"], "grid": C["ice"],
    "text_dark": C["navy"], "text_mid": C["slate"],
}
VULN_ORDER = ["High", "Medium", "Low"]
SAN_JUAN_POVERTY_FALLBACK = 0.3

# ── Global rcParams: zero spines everywhere ────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": C["navy"], "axes.labelsize": 10,
    "axes.labelcolor": C["navy"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.grid": True, "axes.facecolor": C["bg"],
    "grid.color": C["ice"], "grid.linewidth": 0.7,
    "figure.facecolor": C["white"],
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.color": C["slate"], "ytick.color": C["slate"],
    "xtick.bottom": False, "ytick.left": False,
    "legend.fontsize": 8.5, "legend.framealpha": 0.95,
    "legend.edgecolor": C["ice"], "text.color": C["navy"],
})

# ── PCA columns ────────────────────────────────────────────────────────────────
PCA_INPUT_COLS = [
    "hospitals", "clinics", "rhu_count", "bhs_count",
    "birthing_homes", "dialysis_centers", "laboratories",
]
PCA_FEAT_LABELS = {
    "hospitals": "Hospitals", "clinics": "Clinics",
    "rhu_count": "Rural Health Units", "bhs_count": "Barangay Health Stations",
    "birthing_homes": "Birthing Homes", "dialysis_centers": "Dialysis Centers",
    "laboratories": "Laboratories",
}
PC_INTERPRET = {
    0: "Total supply volume\n(city size effect)",
    1: "Govt community health\n(BHS vs diagnostics)",
    2: "RHU vs BHS primary\ncare mix (residual)",
}

# ── GBM feature columns ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "nearest_public_tertiary_km", "beds_per_1000", "level3_per100k",
    "poverty_incidence_2023_pct", "private_ownership_pct",
    "nearest_km_x_poverty", "beds_per_poor_1000", "l3_per_poor_resident",
]
FEATURE_LABELS = {
    "nearest_public_tertiary_km": "Distance to Nearest Public L3 (km)",
    "beds_per_1000":              "Bed Capacity per 1,000 Residents",
    "level3_per100k":             "Level-3 Hospital Density (per 100k)",
    "poverty_incidence_2023_pct": "Poverty Incidence (%, 2023)",
    "private_ownership_pct":      "Private Ownership Share",
    "nearest_km_x_poverty":       "Distance × Poverty (compound barrier)",
    "beds_per_poor_1000":         "Effective Bed Supply (poverty-discounted)",
    "l3_per_poor_resident":       "Effective L3 Access (poverty-discounted)",
}

# ── Hardcoded PCA label offsets (PC1/PC2 space, tuned to real data) ────────────
PCA_OFFSETS = {
    "MANILA":       (-2.8,  0.35, "right",  "center"),
    "QUEZON CITY":  ( 0.30,  0.35, "left",   "bottom"),
    "CALOOCAN":     ( 0.30,  0.35, "left",   "bottom"),
    "LAS PINAS":    ( 0.30,  0.22, "left",   "center"),
    "MAKATI":       ( 0.30, -0.32, "left",   "top"),
    "MALABON":      (-0.35,  0.32, "right",  "bottom"),
    "MANDALUYONG":  ( 0.30,  0.22, "left",   "center"),
    "MARIKINA":     ( 0.30,  0.22, "left",   "center"),
    "MUNTINLUPA":   (-0.35, -0.32, "right",  "top"),
    "NAVOTAS":      (-0.35,  0.32, "right",  "bottom"),
    "PARANAQUE":    ( 0.30, -0.32, "left",   "top"),
    "PASAY":        ( 0.30,  0.32, "left",   "bottom"),
    "PASIG":        ( 0.30, -0.32, "left",   "top"),
    "PATEROS":      (-0.35, -0.32, "right",  "top"),
    "SAN JUAN":     ( 0.30,  0.32, "left",   "bottom"),
    "TAGUIG":       ( 0.30, -0.32, "left",   "top"),
    "VALENZUELA":   (-0.35, -0.32, "right",  "top"),
}

# ── LOO scatter offsets (Wd actual vs predicted space) ─────────────────────────
LOO_OFFSETS = {
    "MANILA":       (-0.045,  0.025, "right"),
    "QUEZON CITY":  ( 0.010,  0.025, "left"),
    "CALOOCAN":     (-0.045, -0.020, "right"),
    "LAS PINAS":    ( 0.010, -0.022, "left"),
    "MAKATI":       ( 0.010,  0.022, "left"),
    "MALABON":      (-0.045,  0.020, "right"),
    "MANDALUYONG":  ( 0.010, -0.020, "left"),
    "MARIKINA":     ( 0.010,  0.022, "left"),
    "MUNTINLUPA":   ( 0.010, -0.020, "left"),
    "NAVOTAS":      (-0.045,  0.020, "right"),
    "PARANAQUE":    (-0.045, -0.020, "right"),
    "PASAY":        ( 0.010,  0.022, "left"),
    "PASIG":        ( 0.010, -0.020, "left"),
    "PATEROS":      (-0.045, -0.022, "right"),
    "SAN JUAN":     ( 0.010,  0.022, "left"),
    "TAGUIG":       ( 0.010, -0.020, "left"),
    "VALENZUELA":   (-0.045,  0.020, "right"),
}

# ── Heatmap indicators ─────────────────────────────────────────────────────────
HEATMAP_COLS = {
    "nearest_public_tertiary_km":    "Distance to\nPublic L3 (km)",
    "poverty_incidence_2023_pct":    "Poverty\nIncidence (%)",
    "private_ownership_pct":         "Private\nOwnership (%)",
    "beds_per_1000":                 "Beds per\n1,000 People",
    "level3_per100k":                "L3 Hospitals\nper 100k",
    "effective_public_beds_per1000": "Public Beds\nper 1,000",
}
HEATMAP_HIGHER_IS_WORSE = {
    "nearest_public_tertiary_km": True,
    "poverty_incidence_2023_pct": True,
    "private_ownership_pct":      True,
    "beds_per_1000":              False,
    "level3_per100k":             False,
    "effective_public_beds_per1000": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    """Load merged feature matrix and optionally model results."""
    df = None
    for p in [os.path.join(DATA_DIR, "merged_metro_manila.csv"),
              "merged_metro_manila.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"  Data: {p}  ({len(df)} cities)")
            break
    if df is None and os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM fact_vulnerability", conn)
        conn.close()
    if df is None:
        raise FileNotFoundError("Cannot find merged_metro_manila.csv. Run 01_data_cleaning.py first.")

    sj = df["city_norm"] == "SAN JUAN"
    if df.loc[sj, "poverty_incidence_2023_pct"].isna().any():
        df.loc[sj, "poverty_incidence_2023_pct"] = SAN_JUAN_POVERTY_FALLBACK

    pov_frac = (df["poverty_incidence_2023_pct"].fillna(
        df["poverty_incidence_2023_pct"].median()) / 100.0).clip(0, 1)
    dist_sq = np.maximum(df["nearest_public_tertiary_km"].values ** 2, 0.5)
    pub_w   = df["weighted_score_per10k"].values * (1 - df["private_ownership_pct"].values)
    priv_w  = df["weighted_score_per10k"].values *     df["private_ownership_pct"].values
    wd_raw  = pub_w / dist_sq + priv_w / dist_sq * (1 - pov_frac.values)
    wd_log  = np.log1p(wd_raw)
    df["wd_score"] = ((wd_log - wd_log.min()) / (wd_log.max() - wd_log.min())).round(4)

    if "effective_public_beds_per1000" not in df.columns:
        df["effective_public_beds_per1000"] = (
            df["beds_per_1000"] * (1 - df["private_ownership_pct"])).round(4)

    pf = pov_frac.values
    df["nearest_km_x_poverty"] = (df["nearest_public_tertiary_km"] * pf).round(4)
    df["beds_per_poor_1000"]   = (df["beds_per_1000"] * (1 - pf)).round(4)
    df["l3_per_poor_resident"] = (df["level3_per100k"] * (1 - pf)).round(4)
    return df


def load_model_json():
    p = os.path.join(MODEL_DIR, "model_results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def clean_axes(ax):
    """Remove all four spines and tick marks."""
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def _title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=C["navy"], pad=14, loc="left")
    if subtitle:
        ax.text(0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=C["slate"], va="bottom")


def save_fig(fig, filename):
    fpath = os.path.join(VIZ_DIR, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight",
                facecolor=C["white"], edgecolor="none")
    plt.close(fig)
    kb = os.path.getsize(fpath) / 1024
    print(f"  → {filename}  ({kb:.0f} KB)")
    return fpath


def vuln_color(label):
    return VULN_COLOR.get(label, C["amber"])


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — PCA SCATTER + LOADINGS TABLE
# ══════════════════════════════════════════════════════════════════════════════
def fig1_pca_scatter_table(df):
    """
    Left: cities in PC1×PC2 space, coloured by vulnerability.
          Label offsets from PCA_OFFSETS dict — no overlap guaranteed.
    Right: colour-coded loadings table. Teal cells = strong positive loading,
           coral cells = strong negative. Interpretation row per PC.
    PC selection: minimum n to explain ≥80% variance.
    """
    X    = df[PCA_INPUT_COLS].fillna(0).values
    X_sc = StandardScaler().fit_transform(X)

    pca_full = PCA(n_components=len(PCA_INPUT_COLS), random_state=42)
    pca_full.fit(X_sc)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(int(np.searchsorted(cumvar, 0.80) + 1), 2)
    print(f"  PCA: {n_comp} PCs explain {cumvar[n_comp-1]*100:.1f}% variance (≥80% threshold)")

    pca    = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_sc)
    var    = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(17, 7.5))
    fig.patch.set_facecolor(C["white"])
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.06)
    ax_s = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])

    # ── Scatter ───────────────────────────────────────────────────────────────
    ax_s.set_facecolor(C["bg"])
    clean_axes(ax_s)
    ax_s.grid(True, color=C["ice"], linewidth=0.7, zorder=0)
    ax_s.axhline(0, color=C["ice"], lw=1.0, zorder=1)
    ax_s.axvline(0, color=C["ice"], lw=1.0, zorder=1)

    pop     = df["population_2020"].fillna(df["population_2020"].median()).values
    pop_n   = (pop - pop.min()) / (pop.max() - pop.min())
    sizes   = 100 + 2200 * pop_n

    for i, row in df.reset_index(drop=True).iterrows():
        vuln  = row.get("vulnerability_label", "Medium")
        color = vuln_color(vuln)
        x, y  = coords[i, 0], coords[i, 1]
        ax_s.scatter(x, y, s=sizes[i], color=color, alpha=0.82,
                     edgecolors=C["white"], linewidths=1.5, zorder=3)

        city = str(row["city_norm"])
        dx, dy, ha, va = PCA_OFFSETS.get(city, (0.3, 0.2, "left", "center"))
        ax_s.annotate(
            city.title(), xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=7.8, color=C["navy"], fontweight="medium",
            ha=ha, va=va, zorder=5,
            arrowprops=dict(arrowstyle="-", color=C["slate"], lw=0.5, alpha=0.55),
        )

    ax_s.set_xlabel(f"PC 1 — Total Supply Volume  ({var[0]*100:.1f}% variance)",
                    fontsize=9, color=C["slate"], labelpad=8)
    ax_s.set_ylabel(f"PC 2 — Govt Community Health  ({var[1]*100:.1f}% variance)",
                    fontsize=9, color=C["slate"], labelpad=8)
    _title(ax_s, "PCA City Supply Profiles",
           f"{n_comp} components explain {cumvar[n_comp-1]*100:.0f}% of infrastructure variance")

    ax_s.legend(
        handles=[mpatches.Patch(facecolor=VULN_COLOR[v], edgecolor=C["white"],
                                label=VULN_LABELS[v]) for v in ["High","Medium","Low"]],
        loc="lower right", framealpha=0.95, edgecolor=C["ice"],
        fontsize=8.5, title="Vulnerability", title_fontsize=8,
    )

    # Bubble size legend
    for pop_val, lbl in [(500_000, "500k"), (1_500_000, "1.5M"), (3_000_000, "3M")]:
        sz = 100 + 2200 * (pop_val - pop.min()) / (pop.max() - pop.min())
        ax_s.scatter([], [], s=sz, color=C["slate"], alpha=0.4,
                     edgecolors=C["white"], linewidths=1, label=f"Pop {lbl}")
    ax_s.legend(handles=ax_s.get_legend_handles_labels()[0],
                labels=ax_s.get_legend_handles_labels()[1],
                loc="lower right", fontsize=7.5, framealpha=0.93,
                edgecolor=C["ice"], title="Vulnerability / Population",
                title_fontsize=7.8)

    # ── Loadings table ────────────────────────────────────────────────────────
    ax_t.set_facecolor(C["white"])
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    clean_axes(ax_t)
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    ax_t.grid(False)

    loadings = pca.components_   # shape (n_comp, n_features)
    feats    = [PCA_FEAT_LABELS[c] for c in PCA_INPUT_COLS]
    n_f      = len(feats)
    n_col    = n_comp + 1         # feature name + one col per PC

    row_h  = 0.76 / (n_f + 3.5)
    col_w  = 0.97 / n_col
    x0     = 0.015
    y_start= 0.955

    # Title
    ax_t.text(0.5, y_start, "Principal Component Loadings",
              ha="center", va="top", fontsize=10, fontweight="bold", color=C["navy"])

    # Header row
    hy = y_start - 0.055
    ax_t.add_patch(mpatches.FancyBboxPatch(
        (x0, hy - row_h * 0.5), 0.97, row_h,
        boxstyle="round,pad=0.005", facecolor=C["navy"], edgecolor="none", zorder=1))
    ax_t.text(x0 + col_w * 0.5, hy, "Feature",
              ha="center", va="center", fontsize=7.8, fontweight="bold",
              color=C["white"], zorder=2)
    for j in range(n_comp):
        xc = x0 + (j + 1.5) * col_w
        ax_t.text(xc, hy, f"PC{j+1}\n({var[j]*100:.0f}%)",
                  ha="center", va="center", fontsize=7.5, fontweight="bold",
                  color=C["white"], zorder=2)

    # Feature rows
    for i, feat in enumerate(feats):
        ry  = hy - row_h * (i + 1.5)
        bg  = C["bg"] if i % 2 == 0 else C["white"]
        ax_t.add_patch(mpatches.FancyBboxPatch(
            (x0, ry - row_h * 0.5), 0.97, row_h,
            boxstyle="round,pad=0.002", facecolor=bg, edgecolor="none", zorder=1))
        ax_t.text(x0 + col_w * 0.5, ry, feat,
                  ha="center", va="center", fontsize=7.0, color=C["navy"], zorder=2)

        for j in range(n_comp):
            val  = loadings[j, i]
            xc   = x0 + (j + 1.5) * col_w
            cx   = x0 + (j + 1) * col_w

            if val > 0.25:
                t = min((val - 0.25) / 0.25, 1.0)
                r = int(13  + (255 - 13)  * (1 - t * 0.75))
                g = int(115 + (255 - 115) * (1 - t * 0.75))
                b = int(119 + (255 - 119) * (1 - t * 0.75))
                cell = f"#{r:02X}{g:02X}{b:02X}"
                tc   = C["white"] if t > 0.5 else C["navy"]
            elif val < -0.20:
                t = min((-val - 0.20) / 0.30, 1.0)
                r = int(232 + (255 - 232) * (1 - t * 0.75))
                g = int(72  + (255 - 72)  * (1 - t * 0.75))
                b = int(85  + (255 - 85)  * (1 - t * 0.75))
                cell = f"#{r:02X}{g:02X}{b:02X}"
                tc   = C["white"] if t > 0.5 else C["navy"]
            else:
                cell = bg
                tc   = C["slate"]

            ax_t.add_patch(mpatches.FancyBboxPatch(
                (cx + 0.004, ry - row_h * 0.44),
                col_w - 0.008, row_h * 0.88,
                boxstyle="round,pad=0.001", facecolor=cell, edgecolor="none", zorder=2))
            ax_t.text(xc, ry, f"{val:+.2f}",
                      ha="center", va="center", fontsize=7.8, fontweight="bold",
                      color=tc, zorder=3)

    # Interpretation row
    iy = hy - row_h * (n_f + 2.0)
    ax_t.add_patch(mpatches.FancyBboxPatch(
        (x0, iy - row_h * 0.5), 0.97, row_h * 1.4,
        boxstyle="round,pad=0.002", facecolor="#EBF5F5", edgecolor="none", zorder=1))
    ax_t.text(x0 + col_w * 0.5, iy + row_h * 0.2, "Interpretation",
              ha="center", va="center", fontsize=7.2, fontweight="bold",
              color=C["teal"], zorder=2)
    for j in range(n_comp):
        xc = x0 + (j + 1.5) * col_w
        ax_t.text(xc, iy, PC_INTERPRET.get(j, "—"),
                  ha="center", va="center", fontsize=6.5, color=C["navy"], zorder=2)

    # Colour note
    ax_t.text(0.5, iy - row_h * 1.3,
              "Teal = strong positive  ·  Coral = strong negative  ·  "
              "Threshold: |loading| > 0.25",
              ha="center", va="center", fontsize=6.5, color=C["slate"])

    save_fig(fig, "fig1_pca_scatter_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — PCA SCREE PLOT
# ══════════════════════════════════════════════════════════════════════════════
def fig2_pca_scree(df):
    """
    Left: bar chart of % variance per PC. Teal = selected (≥80%), slate = rest.
    Right: cumulative variance line with 80% threshold and crossing annotation.
    """
    X    = df[PCA_INPUT_COLS].fillna(0).values
    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=len(PCA_INPUT_COLS), random_state=42)
    pca.fit(X_sc)
    var    = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    n_pcs  = len(var)
    labels = [f"PC{i+1}" for i in range(n_pcs)]
    n_80   = int(np.searchsorted(cumvar, 80.0) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                   gridspec_kw={"wspace": 0.10})
    fig.patch.set_facecolor(C["white"])

    # Left: bar chart
    ax1.set_facecolor(C["bg"])
    clean_axes(ax1)
    ax1.grid(axis="y", color=C["ice"], linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)

    bar_colors = [C["teal"] if i < n_80 else C["slate"] for i in range(n_pcs)]
    bars = ax1.bar(labels, var, color=bar_colors, width=0.58,
                   zorder=3, edgecolor="none")
    for bar, v in zip(bars, var):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=8.5,
                 color=C["navy"], fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=9, color=C["slate"], labelpad=8)
    ax1.set_ylim(0, var.max() * 1.22)
    _title(ax1, "Scree Plot",
           "Variance explained by each principal component")
    ax1.legend(handles=[
        mpatches.Patch(facecolor=C["teal"], label=f"Selected (≥80%)"),
        mpatches.Patch(facecolor=C["slate"], alpha=0.7, label="Not selected"),
    ], fontsize=8, loc="upper right", framealpha=0.95, edgecolor=C["ice"])

    # Right: cumulative line
    ax2.set_facecolor(C["bg"])
    clean_axes(ax2)
    ax2.grid(axis="y", color=C["ice"], linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)

    x_idx = np.arange(n_pcs)
    ax2.fill_between(x_idx, cumvar,
                     where=[i < n_80 for i in x_idx],
                     alpha=0.12, color=C["teal"], zorder=2, interpolate=True)
    ax2.plot(labels, cumvar, color=C["navy"], lw=2.2,
             marker="o", markersize=6.5, markerfacecolor=C["teal"],
             markeredgecolor=C["white"], markeredgewidth=1.5, zorder=4)

    ax2.axhline(80, color=C["coral"], lw=1.5, ls="--", zorder=3, alpha=0.9)
    ax2.text(n_pcs - 0.08, 81.5, "80% threshold",
             ha="right", fontsize=8.5, color=C["coral"], fontweight="bold")

    ax2.annotate(
        f"PC{n_80}: {cumvar[n_80-1]:.1f}%",
        xy=(n_80 - 1, cumvar[n_80 - 1]),
        xytext=(min(n_80 + 0.5, n_pcs - 1.2), cumvar[n_80 - 1] - 9),
        fontsize=8.5, color=C["teal"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["teal"], lw=1.3),
    )
    for i, cv in enumerate(cumvar):
        ax2.text(i, cv + 1.8, f"{cv:.0f}%",
                 ha="center", va="bottom", fontsize=7.5, color=C["slate"])

    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=9,
                   color=C["slate"], labelpad=8)
    ax2.set_ylim(0, 112)
    _title(ax2, "Cumulative Variance",
           f"{n_80} components needed to reach ≥80% of supply variance")

    save_fig(fig, "fig2_pca_scree.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — GBM LOO PREDICTED VS ACTUAL
# ══════════════════════════════════════════════════════════════════════════════
def fig3_gbm_loo(df, model_json=None):
    """
    Scatter: actual Wd (x) vs LOO-predicted Wd (y).
    Points coloured by absolute error (teal→amber→coral gradient).
    Perfect-prediction diagonal in navy dashes.
    City labels use LOO_OFFSETS dict — no overlap.
    R² and MAE annotated in a clean info box.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    stds = X.std()
    X = X.drop(columns=stds[stds == 0].index.tolist(), errors="ignore")
    y = df["wd_score"].copy()

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=2, random_state=42)),
    ])

    # Use saved predictions if available
    y_pred_arr = None
    if model_json:
        for task in model_json.get("task_A_wd", []):
            if "Gradient" in task.get("model", ""):
                preds = task.get("city_preds", [])
                if len(preds) >= 10:
                    pmap = {p["city"]: p["pred"] for p in preds}
                    y_pred_arr = np.array([pmap.get(c, np.nan)
                                           for c in df["city_norm"]])
                    print("  Using saved LOO predictions.")
                    break

    if y_pred_arr is None:
        print("  Re-running GBM LOO CV...")
        mask = ~y.isna()
        y_pr = cross_val_predict(pipe, X[mask], y[mask], cv=LeaveOneOut())
        y_pred_arr = np.full(len(y), np.nan)
        y_pred_arr[~y.isna()] = y_pr

    mask    = ~(np.isnan(y.values) | np.isnan(y_pred_arr))
    y_act   = y.values[mask]
    y_pr    = y_pred_arr[mask]
    cities  = df["city_norm"].values[mask]
    errors  = np.abs(y_pr - y_act)
    r2      = r2_score(y_act, y_pr)
    mae     = mean_absolute_error(y_act, y_pr)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "err", [C["teal"], C["amber"], C["coral"]])

    fig, ax = plt.subplots(figsize=(9, 8.5))
    fig.patch.set_facecolor(C["white"])
    ax.set_facecolor(C["bg"])
    clean_axes(ax)
    ax.grid(True, color=C["ice"], linewidth=0.7, zorder=0)

    pad = 0.04
    lo  = min(y_act.min(), y_pr.min()) - pad
    hi  = max(y_act.max(), y_pr.max()) + pad
    ax.plot([lo, hi], [lo, hi], color=C["navy"], lw=1.5,
            ls="--", alpha=0.55, zorder=2, label="Perfect prediction")

    norm = mcolors.Normalize(errors.min(), errors.max())
    sc   = ax.scatter(y_act, y_pr, c=errors, cmap=cmap, norm=norm,
                      s=115, edgecolors=C["white"], linewidths=1.6,
                      zorder=4, alpha=0.93)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.68, pad=0.02)
    cbar.set_label("Absolute Error (Wd)", fontsize=8.5, color=C["slate"])
    cbar.ax.tick_params(labelsize=7.5)
    for sp in cbar.ax.spines.values():
        sp.set_visible(False)

    for city, xa, ya in zip(cities, y_act, y_pr):
        dx, dy, ha = LOO_OFFSETS.get(city, (0.010, 0.018, "left"))
        ax.text(xa + dx, ya + dy, city.title(),
                fontsize=7.5, color=C["navy"], ha=ha, va="center",
                fontweight="medium", zorder=5)

    ax.text(0.04, 0.94,
            f"LOO Cross-Validation    R² = {r2:.3f}    MAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            color=C["navy"], va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=C["white"],
                      edgecolor=C["ice"], alpha=0.97))

    ax.set_xlabel("Actual Wd Score (2SFCA gravity index)", fontsize=10,
                  color=C["slate"], labelpad=8)
    ax.set_ylabel("Predicted Wd Score (Leave-One-Out CV)", fontsize=10,
                  color=C["slate"], labelpad=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _title(ax, "Gradient Boosting — LOO Validation",
           "Each city is predicted by a model trained on the remaining 16 cities")
    ax.legend(fontsize=8.5, framealpha=0.95, edgecolor=C["ice"])

    save_fig(fig, "fig3_gbm_loo.png")
    return r2, mae


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def fig4_feature_importance(df, model_json=None):
    """
    Horizontal bars sorted descending. Best predictor in coral, others in teal.
    Value labels right of each bar. Zero spines. Clean grid on x-axis only.
    """
    importances, feat_names, labels = None, None, None

    if model_json:
        records = model_json.get("feat_imp_wd", [])
        if records:
            imp_df = pd.DataFrame(records).sort_values("importance", ascending=False)
            feat_names   = imp_df["feature"].tolist()
            importances  = imp_df["importance"].values
            labels       = [FEATURE_LABELS.get(f, f) for f in feat_names]
            print("  Using saved importances from model_results.json.")

    if importances is None:
        print("  Refitting GBM for feature importance...")
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].fillna(df[available].median())
        stds = X.std()
        X = X.drop(columns=stds[stds == 0].index.tolist(), errors="ignore")
        available = list(X.columns)
        y = df["wd_score"].fillna(df["wd_score"].median())
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   GradientBoostingRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                subsample=0.8, min_samples_leaf=2, random_state=42)),
        ])
        pipe.fit(X, y)
        imp_raw    = pipe.named_steps["model"].feature_importances_
        order      = np.argsort(imp_raw)[::-1]
        importances = imp_raw[order]
        feat_names  = [available[i] for i in order]
        labels      = [FEATURE_LABELS.get(f, f) for f in feat_names]

    n           = len(importances)
    imp_rev     = importances[::-1]
    lbl_rev     = labels[::-1]
    bar_colors  = [C["teal"]] * n
    bar_colors[n - 1] = C["coral"]   # top predictor

    fig, ax = plt.subplots(figsize=(11, max(5.5, n * 0.68 + 2.0)))
    fig.patch.set_facecolor(C["white"])
    ax.set_facecolor(C["bg"])
    clean_axes(ax)
    ax.grid(axis="x", color=C["ice"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    y_pos = np.arange(n)
    bars  = ax.barh(y_pos, imp_rev, color=bar_colors, height=0.60,
                    zorder=3, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(lbl_rev, fontsize=9.0, color=C["navy"])
    ax.tick_params(axis="y", length=0, pad=7)

    for bar, val in zip(bars, imp_rev):
        ax.text(bar.get_width() + importances.max() * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=8.5, color=C["slate"])

    ax.set_xlabel("Mean Impurity Decrease (GBM)", fontsize=9,
                  color=C["slate"], labelpad=8)
    ax.set_xlim(0, imp_rev.max() * 1.22)
    _title(ax, "Feature Importance — What Drives Inaccessibility?",
           "Gradient Boosting (n_estimators=50, depth=2) trained on Wd Accessibility Index")

    best = lbl_rev[-1]
    ax.annotate(
        f"← Best predictor:\n  {best}",
        xy=(imp_rev[-1], n - 1),
        xytext=(imp_rev[-1] * 0.52, n - 2.5),
        fontsize=8, color=C["coral"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["coral"], lw=1.2),
    )
    ax.legend(handles=[
        mpatches.Patch(facecolor=C["coral"], label="Top predictor"),
        mpatches.Patch(facecolor=C["teal"], label="Other features"),
    ], fontsize=8.5, loc="lower right", framealpha=0.95, edgecolor=C["ice"])

    save_fig(fig, "fig4_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — RECOMMENDATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def fig5_recommendation_heatmap(df):
    """
    Cities × indicator matrix. Each column normalised independently.
    Higher = worse for distance/poverty/private; higher = better for beds/L3.
    Colour: teal (good) → amber → coral (bad).
    Rows: sorted High → Medium → Low, then by Wd within group.
    Cell values printed with appropriate units.
    Coloured strip left of city name shows vulnerability class.
    """
    indicators = [c for c in HEATMAP_COLS if c in df.columns]
    labels_col  = [HEATMAP_COLS[c] for c in indicators]

    label_order = {"High": 0, "Medium": 1, "Low": 2}
    plot_df = df[["city_norm", "vulnerability_label", "wd_score"] +
                 indicators].copy()
    plot_df["_lo"] = plot_df["vulnerability_label"].map(label_order).fillna(1)
    plot_df = plot_df.sort_values(["_lo", "wd_score"]).reset_index(drop=True)

    city_labels = [c.title() for c in plot_df["city_norm"]]
    vuln_labels = plot_df["vulnerability_label"].tolist()
    n_cities    = len(plot_df)
    n_cols      = len(indicators)

    mat_norm   = np.zeros((n_cities, n_cols))
    disp_vals  = []

    for j, col in enumerate(indicators):
        vals = plot_df[col].fillna(plot_df[col].median()).values.astype(float)
        mn, mx = vals.min(), vals.max()
        norm   = (vals - mn) / (mx - mn) if mx > mn else np.full_like(vals, 0.5)
        mat_norm[:, j] = norm if HEATMAP_HIGHER_IS_WORSE[col] else (1 - norm)

        if "pct" in col or "ownership" in col:
            disp_vals.append([f"{v:.1f}%" for v in vals])
        elif "km" in col:
            disp_vals.append([f"{v:.2f}" for v in vals])
        else:
            disp_vals.append([f"{v:.2f}" for v in vals])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sev", [C["teal"], "#F4E1B5", C["coral"]], N=256)

    cell_h = 0.52
    cell_w = 1.65
    fig_h  = max(8.5, n_cities * cell_h + 4.0)
    fig_w  = n_cols * cell_w + 4.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(C["white"])
    ax.set_facecolor(C["white"])
    clean_axes(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw cells
    for i in range(n_cities):
        for j in range(n_cols):
            rgba  = cmap(mat_norm[i, j])
            rect  = mpatches.FancyBboxPatch(
                (j + 0.05, n_cities - i - 1 + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.02",
                facecolor=rgba, edgecolor=C["white"], linewidth=1.3, zorder=2)
            ax.add_patch(rect)

            bright = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc     = C["white"] if bright < 0.58 else C["navy"]
            val_s  = disp_vals[j][i] if j < len(disp_vals) else "—"
            ax.text(j + 0.5, n_cities - i - 0.5, val_s,
                    ha="center", va="center", fontsize=8.0,
                    color=tc, fontweight="bold", zorder=3)

    # Column headers
    for j, lbl in enumerate(labels_col):
        ax.text(j + 0.5, n_cities + 0.22, lbl,
                ha="center", va="bottom", fontsize=9.0,
                fontweight="bold", color=C["navy"])

    # Row labels + colour strips
    for i, (city, vuln) in enumerate(zip(city_labels, vuln_labels)):
        yc = n_cities - i - 0.5
        ax.add_patch(mpatches.FancyBboxPatch(
            (-1.5, yc - 0.42), 0.20, 0.84,
            boxstyle="round,pad=0.01",
            facecolor=VULN_COLOR.get(vuln, C["amber"]),
            edgecolor="none", zorder=2))
        ax.text(-1.25, yc, city,
                ha="left", va="center", fontsize=9.0,
                color=C["navy"], fontweight="medium")

    # Vulnerability legend
    for vi, (vl, vc) in enumerate(zip(
            ["High Vulnerability", "Medium Vulnerability", "Low Vulnerability"],
            [C["coral"], C["amber"], C["teal"]])):
        ax.text(-1.5, n_cities + 0.80 - vi * 0.42, f"■ {vl}",
                ha="left", va="center", fontsize=8.2,
                color=vc, fontweight="bold")

    # Gradient scale bar at bottom
    gx = np.linspace(0, n_cols, 300)
    ax.scatter(gx, np.full_like(gx, -0.9), c=gx / n_cols, cmap=cmap,
               s=13, linewidths=0, zorder=2)
    ax.text(0,    -1.18, "Better access", ha="left",  fontsize=7.8, color=C["teal"])
    ax.text(n_cols, -1.18, "Needs attention", ha="right", fontsize=7.8, color=C["coral"])

    ax.set_xlim(-1.8, n_cols + 0.15)
    ax.set_ylim(-1.45, n_cities + 1.35)

    ax.text(n_cols / 2, n_cities + 1.05,
            "Recommendation Heatmap — City Healthcare Indicator Summary",
            ha="center", va="bottom", fontsize=14,
            fontweight="bold", color=C["navy"])
    ax.text(n_cols / 2, n_cities + 0.62,
            "Sorted: High Vulnerability → Low  ·  Colour = relative severity within each indicator  ·  "
            "Values shown in original units",
            ha="center", va="bottom", fontsize=8.5, color=C["slate"])

    save_fig(fig, "fig5_recommendation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Wd ACCESSIBILITY RANKING BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
def fig9_wd_ranking_bar(df):
    """
    Horizontal bar chart: cities sorted worst-to-best by Wd accessibility score.

    Design
    ------
    Sorted ascending (worst at top) so the most urgent city is always the
    first thing the eye lands on — no scanning required.  Bars are coloured
    by vulnerability label so the colour carries information independently
    of position.  The rank number (#1 = worst) is printed inside the bar in
    white.  Cities with zero Level-3 hospitals are flagged with ★ and the
    plain-English label "No full-service hospital in this city" — not the
    jargon term "L3 Desert", which requires project context to decode.
    Secondary annotations (poverty %, distance) are placed in muted grey
    after the Wd value to add density without clutter.

    Story
    -----
    "Which cities need LGU intervention most urgently?" — answerable in
    10 seconds without reading any body text.
    """
    df_s   = df.copy().sort_values("wd_score", ascending=True).reset_index(drop=True)
    n      = len(df_s)
    l3d    = df_s["level3_hospitals"].fillna(0).values == 0
    pov_v  = df_s["poverty_incidence_2023_pct"].fillna(0.3).values
    km_v   = df_s["nearest_public_tertiary_km"].values

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(C["white"])
    ax.set_facecolor(C["bg"])
    clean_axes(ax)
    ax.grid(axis="x", color=C["ice"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    y_pos      = np.arange(n)
    bar_colors = [vuln_color(v) for v in df_s["vulnerability_label"]]
    bars       = ax.barh(y_pos, df_s["wd_score"], color=bar_colors,
                         height=0.64, zorder=3, edgecolor="none")

    for idx, bar in enumerate(bars):
        wd     = df_s["wd_score"].iloc[idx]
        rank_n = n - idx
        is_l3d = bool(l3d[idx])
        pov    = pov_v[idx]
        km     = km_v[idx]

        # Rank number — always inside bar if wide enough, else just right of bar end
        if wd > 0.07:
            ax.text(0.009, idx, f"#{rank_n}", va="center", ha="left",
                    fontsize=8, color=C["white"], fontweight="bold", zorder=5)
        elif wd > 0.025:
            # Short but visible bar: rank just right of bar, value further right
            ax.text(wd + 0.007, idx, f"#{rank_n}", va="center", ha="left",
                    fontsize=8, color=C["navy"], fontweight="bold", zorder=5)
        else:
            # Very short or zero bar: rank at fixed position
            ax.text(0.055, idx, f"#{rank_n}", va="center", ha="left",
                    fontsize=8, color=C["navy"], fontweight="bold", zorder=5)

        if is_l3d:
            ax.text(1.015, idx,
                    "★  No full-service hospital in this city",
                    va="center", ha="left", fontsize=8.5,
                    color=C["coral"], fontweight="bold", zorder=5,
                    transform=ax.get_yaxis_transform())
        else:
            ax.text(wd + 0.013, idx, f"{wd:.3f}",
                    va="center", ha="left", fontsize=9,
                    color=C["navy"], fontweight="bold", zorder=5)
            ax.text(wd + 0.072, idx,
                    f"  Pov {pov:.1f}%  ·  {km:.1f} km away",
                    va="center", ha="left", fontsize=7.8,
                    color=C["slate"], zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.title() for c in df_s["city_norm"]],
                       fontsize=9.5, color=C["navy"])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel(
        "Weighted Accessibility Index (Wd)"
        "     0 = functionally unreachable  ·  1 = best access",
        fontsize=9.5, color=C["slate"], labelpad=10)
    ax.set_xlim(0, 1.55)

    _title(ax, "City Accessibility Ranking — Who Needs Help Most Urgently?",
           "Sorted worst-to-best  ·  ★ = city has no full-service public hospital  ·  "
           "Colour = vulnerability level")

    handles = [mpatches.Patch(facecolor=VULN_COLOR[v], edgecolor="none",
                               label=VULN_LABELS[v]) for v in VULN_ORDER]
    handles.append(ax.scatter([], [], marker="*", color=C["coral"],
                               s=90, label="No full-service hospital"))
    ax.legend(handles=handles, loc="lower right",
              fontsize=8.5, framealpha=0.95, edgecolor=C["ice"])

    save_fig(fig, "fig9_wd_ranking_bar.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — POPULATION GROWTH vs SUPPLY QUALITY (FUTURE RISK QUADRANT)
# ══════════════════════════════════════════════════════════════════════════════
def fig10_growth_vs_supply(df):
    """
    Scatter: annual population growth rate 2020→2024 (x) vs quality-adjusted
    facility supply per 10k residents (y). Bubble size = population_2024.
    Colour = vulnerability label.

    Four quadrants (defined by NCR medians)
    ----------------------------------------
    Top-left    Stable:         slow growth, good supply
    Top-right   Watch:          fast growth but currently well-supplied
    Bottom-left Already strained: slow growth, thin supply
    Bottom-right FUTURE RISK:   fast growth + thin supply → coral callout

    Axis choices
    ------------
    X: pop_growth_rate_pct — annualised 2020→2024, PSA projections
    Y: weighted_score_per10k — quality-adjusted supply (L3 hospitals weighted
       higher than clinics, so a city of ICU beds ≠ a city of birthing homes)
    Bubble: population_2024 (scale of the problem, not just growth rate)

    Story
    -----
    Current rankings show TODAY's gaps.  This chart shows WHERE GAPS WILL
    WIDEN if infrastructure investment doesn't match population pressure.
    Taguig (1.61%/yr, supply below median) is the key policy priority for
    preventive infrastructure — before it becomes the next crisis.
    """
    df_p = df.copy()
    if "population_2024" not in df_p.columns:
        df_p["population_2024"] = (
            df_p["population_2020"] *
            (1 + df_p["pop_growth_rate_pct"] / 100) ** 4
        ).round(0)

    growth     = df_p["pop_growth_rate_pct"].values
    supply     = df_p["weighted_score_per10k"].values
    pop24      = df_p["population_2024"].fillna(df_p["population_2020"]).values
    cities     = df_p["city_norm"].values
    vuln       = df_p["vulnerability_label"].values
    growth_med = float(np.median(growth))
    supply_med = float(np.median(supply))

    pop_min, pop_max = float(pop24.min()), float(pop24.max())
    sizes = 80 + 2600 * (pop24 - pop_min) / (pop_max - pop_min + 1)

    x_lo = float(growth.min()) - 0.20
    x_hi = float(growth.max()) + 0.55
    y_lo = float(supply.min()) - 0.55
    y_hi = float(supply.max()) + 1.35

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(C["white"])
    ax.set_facecolor(C["bg"])
    clean_axes(ax)
    ax.grid(True, color=C["ice"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Coral tint in the future-risk quadrant
    ax.fill_between([growth_med, x_hi], [y_lo, y_lo], [supply_med, supply_med],
                    color=C["coral"], alpha=0.06, zorder=1)

    # Median reference lines
    ax.axvline(growth_med, color=C["slate"], lw=1.0, ls="--", alpha=0.50, zorder=2)
    ax.axhline(supply_med, color=C["slate"], lw=1.0, ls="--", alpha=0.50, zorder=2)
    ax.text(growth_med + 0.03, y_lo + 0.10,
            f"NCR median growth {growth_med:.2f}%",
            fontsize=7.5, color=C["slate"], va="bottom")
    ax.text(x_lo + 0.06, supply_med + 0.06,
            f"NCR median supply {supply_med:.2f}",
            fontsize=7.5, color=C["slate"], va="bottom")

    # Quadrant corner labels (subtle)
    quad_info = [
        (x_lo + 0.07, y_hi - 0.10, "Stable: slow growth + good supply",
         C["teal"], "left"),
        (x_hi - 0.06, y_hi - 0.10, "Watch: fast growth + good supply",
         C["amber"], "right"),
        (x_lo + 0.07, y_lo + 0.14, "Already strained: slow + thin supply",
         C["slate"], "left"),
        (x_hi - 0.06, y_lo + 0.14, "Future risk: fast + thin supply",
         C["coral"], "right"),
    ]
    for qx, qy, qt, qc, qha in quad_info:
        ax.text(qx, qy, qt, fontsize=8.0, color=qc, alpha=0.55,
                ha=qha, va="top", fontweight="bold")

    # Bubbles grouped by vulnerability
    for vg in ["Low", "Medium", "High"]:
        mask = vuln == vg
        ax.scatter(growth[mask], supply[mask], s=sizes[mask],
                   color=VULN_COLOR[vg], alpha=0.83,
                   edgecolors=C["white"], linewidths=1.6,
                   zorder=4, label=VULN_LABELS[vg])

    # Callout for the future-risk cluster
    fast_thin = (growth > growth_med) & (supply <= supply_med)
    if fast_thin.any():
        cx = float(growth[fast_thin].mean())
        cy = float(supply[fast_thin].mean())
        ax.annotate(
            "Fast-growing cities with\nthin supply today\n"
            "→  Future risk if investment\n   doesn't keep pace",
            xy=(cx, cy),
            xytext=(cx + 0.28, cy - 0.65),
            fontsize=9, color=C["coral"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["coral"], lw=1.5),
            bbox=dict(boxstyle="round,pad=0.45", facecolor=C["white"],
                      edgecolor=C["coral"], alpha=0.95),
            zorder=6,
        )

    # City labels — hardcoded offsets to avoid overlap
    label_off = {
        "MANILA":       (-0.07, -0.24, "center"),
        "QUEZON CITY":  ( 0.04,  0.17, "left"),
        "CALOOCAN":     (-0.07, -0.24, "center"),
        "LAS PINAS":    ( 0.04, -0.22, "left"),
        "MAKATI":       ( 0.04,  0.17, "left"),
        "MALABON":      (-0.08, -0.24, "right"),
        "MANDALUYONG":  ( 0.04,  0.17, "left"),
        "MARIKINA":     ( 0.04,  0.17, "left"),
        "MUNTINLUPA":   (-0.09,  0.17, "right"),
        "NAVOTAS":      (-0.07, -0.24, "center"),
        "PARANAQUE":    ( 0.04, -0.22, "left"),
        "PASAY":        (-0.07, -0.24, "center"),
        "PASIG":        ( 0.04,  0.17, "left"),
        "PATEROS":      ( 0.04, -0.22, "left"),
        "SAN JUAN":     ( 0.04,  0.17, "left"),
        "TAGUIG":       ( 0.04, -0.22, "left"),
        "VALENZUELA":   (-0.07, -0.24, "center"),
    }
    for i, (city, gv, sv) in enumerate(zip(cities, growth, supply)):
        dx, dy, ha = label_off.get(city, (0.04, 0.15, "left"))
        ax.annotate(
            city.title(), xy=(gv, sv),
            xytext=(gv + dx, sv + dy),
            fontsize=7.8, color=C["navy"], ha=ha, va="center",
            fontweight="medium", zorder=5,
            arrowprops=dict(arrowstyle="-", color=C["slate"], lw=0.5, alpha=0.45),
        )

    ax.set_xlabel("Annual Population Growth Rate  (%, 2020→2024)",
                  fontsize=10, color=C["slate"], labelpad=10)
    ax.set_ylabel(
        "Quality-Adjusted Facility Supply\n(weighted facilities per 10,000 residents)",
        fontsize=10, color=C["slate"], labelpad=10)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    _title(ax, "Future Risk: Which Cities Will Fall Behind?",
           "Fast-growing cities with thin supply today will face worsening access gaps "
           "unless infrastructure investment keeps pace with population")

    # Population size legend
    for pref, plbl in [(500_000, "500k pop"), (1_500_000, "1.5M pop"), (3_000_000, "3M pop")]:
        sz = 80 + 2600 * (pref - pop_min) / (pop_max - pop_min + 1)
        ax.scatter([], [], s=sz, color=C["slate"], alpha=0.35,
                   edgecolors=C["white"], linewidths=1, label=f"Pop {plbl}")

    ax.legend(fontsize=8.5, framealpha=0.95, edgecolor=C["ice"],
              loc="upper left", title="Vulnerability / Population",
              title_fontsize=8)

    save_fig(fig, "fig10_growth_vs_supply.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("HEALTHCARE VULNERABILITY — SCRIPT 04: VISUALIZATION")
    print("Project : The Metro Manila Healthcare Paradox")
    print("Outputs : fig1 PCA scatter+table  |  fig2 Scree")
    print("          fig3 GBM LOO            |  fig4 Feature Importance")
    print("          fig5 Recommendation Heatmap")
    print("          fig9 Wd Ranking Bar     |  fig10 Growth vs Supply")
    print("Colours : Navy/Teal/Coral  |  Zero spines  |  150 DPI")
    print("=" * 68)

    print("\n[1/6] Loading data...")
    df         = load_data()
    model_json = load_model_json()
    print(f"  Model JSON: {'loaded' if model_json else 'not found — models will refit'}")

    print("\n[2/6] Fig 1 — PCA Scatter + Loadings Table...")
    fig1_pca_scatter_table(df)

    print("\n[3/6] Fig 2 — PCA Scree Plot...")
    fig2_pca_scree(df)

    print("\n[4/6] Fig 3 — Gradient Boosting LOO Plot...")
    r2, mae = fig3_gbm_loo(df, model_json)
    print(f"  LOO R²={r2:.4f}  MAE={mae:.4f}")

    print("\n[5/6] Fig 4 — Feature Importance...")
    fig4_feature_importance(df, model_json)

    print("\n[6/8] Fig 5 — Recommendation Heatmap...")
    fig5_recommendation_heatmap(df)

    print("\n[7/8] Fig 9 — Wd Accessibility Ranking Bar...")
    fig9_wd_ranking_bar(df)

    print("\n[8/8] Fig 10 — Population Growth vs Supply Quality...")
    fig10_growth_vs_supply(df)

    print("\n" + "=" * 68)
    print("ALL VISUALIZATIONS COMPLETE")
    print(f"Output: {os.path.abspath(VIZ_DIR)}")
    print("=" * 68)
    files = sorted(f for f in os.listdir(VIZ_DIR) if f.endswith(".png"))
    for f in files:
        kb = os.path.getsize(os.path.join(VIZ_DIR, f)) / 1024
        print(f"  {f:<48}  {kb:>6.0f} KB")
    print("\nChart → Report section mapping:")
    print("  fig1  methodology  (PCA justification + component table)")
    print("  fig2  methodology  (scree, n_components selection)")
    print("  fig3  results      (GBM model validation)")
    print("  fig4  findings     (top drivers of inaccessibility)")
    print("  fig5  conclusions  (LGU priority briefing heatmap)")
    print("  fig9  findings    (Wd ranking — immediate intervention priority)")
    print("  fig10 forward     (growth vs supply — future risk identification)")