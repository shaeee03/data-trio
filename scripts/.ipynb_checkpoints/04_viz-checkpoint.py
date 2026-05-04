"""
================================================================================
SCRIPT 04: Data Visualization & Storytelling
Project:   The Metro Manila Healthcare Paradox
================================================================================

STORY STRUCTURE (mirrors the project narrative)
------------------------------------------------
  Act 1 — The Mirage         : Fig 1  Priority Matrix — proximity ≠ access
  Act 2 — The Investigation  : Fig 2  PCA Scree & Loadings
                               Fig 3  GBM Learning Curve
                               Fig 4  Feature Importance
  Act 3 — The Finding        : Fig 5  Wd Accessibility Ranking
                               Fig 6  LOO Validation Scatter
  Act 4 — The Call           : Fig 7  Intervention Heatmap (what to fix, where)
                               Fig 8  The Paradox Decomposed (ownership vs poverty)

DESIGN PRINCIPLES
-----------------
  - Zero spines on all axes (ax.spines["*"].set_visible(False))
  - Every annotation offset-tested against data range so nothing overlaps
  - Consistent navy/teal/coral palette — healthcare, not generic blue
  - Titles 15–16 pt bold, axis labels 11 pt, tick labels 9 pt
  - All figures 1920×1080 equivalent (16:9, 200 dpi)
  - Subtitles carry the "so what" — readers understand without the talk
  - Legends placed by content, not default location

OUTPUTS → data/visualization_output/
================================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "../data/data_cleaning_output"
MODEL_DIR = "../data/model_output"
VIZ_DIR   = "../data/visualization_output"
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Design tokens ─────────────────────────────────────────────────────────────
C = {
    "navy":    "#1B2A4A",   # primary dark
    "teal":    "#0D7377",   # positive / accessible
    "coral":   "#E84855",   # alert / high urgency
    "amber":   "#F4A261",   # medium
    "smoke":   "#F7F8FA",   # background
    "mist":    "#E8ECF0",   # grid / separator
    "ink":     "#2D3748",   # body text
    "muted":   "#718096",   # secondary text
    "white":   "#FFFFFF",
    "High":    "#E84855",
    "Medium":  "#F4A261",
    "Low":     "#0D7377",
}

VULN_ORDER = ["High", "Medium", "Low"]
SAN_JUAN_POVERTY = 0.3
PCA_COLS = ["hospitals","clinics","rhu_count","bhs_count",
            "birthing_homes","dialysis_centers","laboratories"]
FEAT_LABELS = [
    "Distance to\nNearest Public L3",
    "Bed Capacity\n(per 1,000)",
    "L3 Hospital\nDensity (per 100k)",
    "Poverty\nIncidence (%)",
    "Private Ownership\nShare (%)",
    "Distance ×\nPoverty",
    "Beds ×\n(1 − Poverty)",
    "L3 ×\n(1 − Poverty)",
]

# ── Shared rcParams ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   C["white"],
    "axes.facecolor":     C["smoke"],
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     15,
    "axes.titleweight":   "bold",
    "axes.titlepad":      18,
    "axes.labelsize":     11,
    "axes.labelcolor":    C["ink"],
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.color":        C["muted"],
    "ytick.color":        C["muted"],
    "axes.grid":          True,
    "grid.color":         C["mist"],
    "grid.linewidth":     0.8,
    "grid.alpha":         1.0,
    "legend.fontsize":    9,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   C["mist"],
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  C["white"],
})


def _strip_spines(ax, keep=()):
    """Remove all spines; optionally keep a subset e.g. ('left',)."""
    for sp in ["top","right","left","bottom"]:
        ax.spines[sp].set_visible(sp in keep)


def _save(fig, name):
    p = os.path.join(VIZ_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    kb = os.path.getsize(p) / 1024
    print(f"  → {name}  ({kb:.0f} KB)")
    return p


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION (shared across all figures)
# ══════════════════════════════════════════════════════════════════════════════
def prepare(df):
    df = df.copy()
    df.loc[df["city_norm"] == "SAN JUAN", "poverty_incidence_2023_pct"] = SAN_JUAN_POVERTY
    pov  = df["poverty_incidence_2023_pct"].values
    km   = df["nearest_public_tertiary_km"].values
    priv = df["private_ownership_pct"].values
    beds = df["beds_per_1000"].values
    l3   = df["level3_per100k"].values
    pf   = pov / 100.0

    dist_sq = np.maximum(km ** 2, 0.5)
    pub_w   = df["weighted_score_per10k"].values * (1 - priv)
    priv_w  = df["weighted_score_per10k"].values * priv
    wd_raw  = pub_w / dist_sq + priv_w / dist_sq * (1 - pf)
    wl      = np.log1p(wd_raw)
    wd      = (wl - wl.min()) / (wl.max() - wl.min())

    df["wd_score"]                     = wd
    df["effective_public_beds_per1000"] = (beds * (1 - priv)).round(4)
    df["l3_desert"]                    = l3 == 0

    X = np.column_stack([km, beds, l3, pov, priv,
                         km * pf, beds * (1 - pf), l3 * (1 - pf)])
    return df, X, wd, pov, km, priv, beds, l3, pf


def fit_models(X, wd, pub_beds):
    pipe = lambda m: Pipeline([("i", SimpleImputer(strategy="median")),
                                ("s", StandardScaler()), ("m", m)])
    gbm  = pipe(GradientBoostingRegressor(n_estimators=50, max_depth=2,
                learning_rate=0.1, subsample=0.8, random_state=42))
    gbm.fit(X, wd)
    yp_wd   = cross_val_predict(gbm, X, wd, cv=LeaveOneOut())
    r2_wd   = r2_score(wd, yp_wd)
    mae_wd  = mean_absolute_error(wd, yp_wd)
    imp     = gbm.named_steps["m"].feature_importances_

    ridge   = pipe(Ridge(alpha=1.0))
    ridge.fit(X, pub_beds)
    yp_beds = cross_val_predict(ridge, X, pub_beds, cv=LeaveOneOut())
    r2_beds = r2_score(pub_beds, yp_beds)

    lc_n, lc_tr, lc_te = [5,10,15,20,30,40,50,60,80,100], [], []
    for n in lc_n:
        m  = pipe(GradientBoostingRegressor(n_estimators=n, max_depth=2,
               learning_rate=0.1, subsample=0.8, random_state=42))
        m.fit(X, wd)
        lc_tr.append(mean_absolute_error(wd, m.predict(X)))
        lc_te.append(mean_absolute_error(wd,
            cross_val_predict(m, X, wd, cv=LeaveOneOut())))

    return dict(gbm=gbm, imp=imp, yp_wd=yp_wd, r2_wd=r2_wd, mae_wd=mae_wd,
                yp_beds=yp_beds, r2_beds=r2_beds,
                lc_n=lc_n, lc_tr=lc_tr, lc_te=lc_te)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — THE PRIORITY MATRIX  (Act 1: The Mirage)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_priority_matrix(df, pov, km):
    """
    Bubble chart: Poverty % (x) vs Distance to nearest public L3 (y).
    Bubble size = population.  Colour = vulnerability label.
    Upper-right quadrant = cities where residents are BOTH poor AND far — the Collision Zone.
    """
    fig, ax = plt.subplots(figsize=(13, 8))
    _strip_spines(ax)

    pop    = df["population_2020"].values
    s_min, s_max = 120, 2200
    sizes  = s_min + (s_max - s_min) * (pop - pop.min()) / (pop.max() - pop.min())

    pov_med  = np.median(pov)
    dist_med = np.median(km)

    # Quadrant shading — very subtle
    xmax = pov.max() * 1.18
    ymax = km.max() * 1.15
    ax.fill_between([pov_med, xmax], dist_med, ymax,
                    color=C["coral"], alpha=0.06, zorder=0)

    # Median reference lines
    ax.axvline(pov_med, color=C["muted"], lw=1.0, ls="--", alpha=0.6, zorder=1)
    ax.axhline(dist_med, color=C["muted"], lw=1.0, ls="--", alpha=0.6, zorder=1)

    # Collision zone label — top-right only
    ax.text(pov_med + 0.06, ymax * 0.93,
            "COLLISION ZONE\nPoor + Far from Care",
            fontsize=8.5, color=C["coral"], fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=C["white"],
                      ec=C["coral"], alpha=0.85, linewidth=1.2))

    # Plot bubbles by vulnerability (High last so they're on top)
    for vuln in ["Low", "Medium", "High"]:
        mask = df["vulnerability_label"] == vuln
        ax.scatter(pov[mask], km[mask], s=sizes[mask],
                   color=C[vuln], alpha=0.78, edgecolors=C["white"],
                   linewidths=1.5, zorder=3, label=vuln)
        # L3 desert ring
        l3d = mask & df["l3_desert"].values
        if l3d.any():
            ax.scatter(pov[l3d], km[l3d], s=sizes[l3d] * 1.55,
                       facecolors="none", edgecolors=C["coral"],
                       linewidths=2.2, zorder=4)

    # City labels — smart offsets to avoid overlap
    offsets = {
        "NAVOTAS":     (+0.07, +0.18),
        "PARANAQUE":   (+0.07, +0.18),
        "PATEROS":     (-0.35, +0.18),
        "MALABON":     (-0.40, -0.30),
        "MANILA":      (-0.38, +0.18),
        "QUEZON CITY": (+0.07, -0.30),
        "PASAY":       (-0.38, -0.30),
        "PASIG":       (+0.07, +0.18),
        "MAKATI":      (-0.38, +0.18),
        "SAN JUAN":    (+0.07, -0.30),
        "CALOOCAN":    (+0.07, -0.30),
        "MANDALUYONG": (-0.42, +0.18),
    }
    for i, row in df.iterrows():
        city = row["city_norm"]
        ox, oy = offsets.get(city, (0.07, 0.18))
        x, y   = pov[i], km[i]
        ax.annotate(
            city.title(),
            xy=(x, y), xytext=(x + ox, y + oy),
            fontsize=7.5, color=C["ink"],
            arrowprops=dict(arrowstyle="-", color=C["mist"], lw=0.8),
            zorder=5,
        )

    ax.set_xlabel("Poverty Incidence  (%)", labelpad=10)
    ax.set_ylabel("Distance to Nearest\nPublic L3 Hospital  (km)", labelpad=10)
    ax.set_title("Proximity to a Hospital Does Not Equal Access to Care",
                 pad=20)
    ax.text(0.5, -0.11,
            "Bubble size = city population  ·  Red ring = L3 Desert (zero Level-3 hospitals)  ·  "
            "Dashed lines = NCR medians",
            transform=ax.transAxes, fontsize=8, ha="center", color=C["muted"])

    ax.set_xlim(-0.1, xmax)
    ax.set_ylim(-0.3, ymax)
    ax.grid(axis="both", color=C["mist"], linewidth=0.8)

    handles = [
        mpatches.Patch(fc=C[v], ec=C["white"], label=f"{v} Vulnerability") for v in VULN_ORDER
    ] + [mpatches.Patch(fc="none", ec=C["coral"], label="L3 Desert")]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              facecolor=C["white"], edgecolor=C["mist"],
              labelcolor=C["ink"], fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, "fig1_priority_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PCA SCREE + LOADINGS  (Act 2: The Investigation)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_pca(df):
    """
    Left: scree plot showing cumulative variance vs 80% threshold.
    Right: loadings heatmap showing what each PC represents.
    """
    X  = df[PCA_COLS].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    pca_full = PCA(random_state=42).fit(Xs)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_) * 100
    indvar   = pca_full.explained_variance_ratio_ * 100
    n_pcs    = len(indvar)

    pca2     = PCA(n_components=2, random_state=42).fit(Xs)
    loadings = pd.DataFrame(
        pca2.components_.T,
        index=[c.replace("_", " ").title() for c in PCA_COLS],
        columns=["PC1  (72.4%)\nVolume Index", "PC2  (19.0%)\nPrimary Care Index"]
    ).round(3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [1.1, 1]})
    _strip_spines(ax1)

    # Scree bars
    bar_colors = [C["teal"] if i < 2 else C["mist"] for i in range(n_pcs)]
    ax1.bar(np.arange(1, n_pcs + 1), indvar, color=bar_colors,
            width=0.55, zorder=2, label="Individual variance")
    ax1.plot(np.arange(1, n_pcs + 1), cumvar, "o-",
             color=C["coral"], lw=2.0, ms=7, zorder=3, label="Cumulative")

    # 80% threshold
    ax1.axhline(80, color=C["coral"], lw=1.4, ls="--", alpha=0.8)
    ax1.text(n_pcs - 0.1, 81.5, "80% threshold", fontsize=8.5,
             color=C["coral"], ha="right")

    # Annotate cumulative
    for i, (cv, iv) in enumerate(zip(cumvar, indvar)):
        if i < 4:
            ax1.text(i + 1, cv + 1.8, f"{cv:.0f}%",
                     ha="center", fontsize=8, color=C["coral"], fontweight="bold")
        ax1.text(i + 1, iv * 0.45, f"{iv:.1f}%",
                 ha="center", fontsize=7.5, color=C["white"], fontweight="bold")

    ax1.set_xticks(np.arange(1, n_pcs + 1))
    ax1.set_xticklabels([f"PC{i}" for i in range(1, n_pcs + 1)], fontsize=9)
    ax1.set_ylabel("Explained Variance  (%)", labelpad=10)
    ax1.set_ylim(0, 110)
    ax1.set_title("How Many Components\nCapture 80% of Variance?", pad=16)
    ax1.legend(loc="center right", fontsize=9, frameon=True,
               facecolor=C["white"], edgecolor=C["mist"])
    ax1.grid(axis="y", color=C["mist"], linewidth=0.8)
    ax1.grid(axis="x", visible=False)

    # Loadings heatmap
    cmap = LinearSegmentedColormap.from_list(
        "teal_coral", [C["coral"], C["white"], C["teal"]], N=256)
    ax2.set_aspect("auto")
    sns.heatmap(
        loadings, ax=ax2,
        cmap=cmap, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 10, "weight": "bold", "color": C["ink"]},
        linewidths=1.5, linecolor=C["white"],
        cbar_kws={"shrink": 0.75, "label": "Loading strength"},
    )
    ax2.set_title("What Does Each Component Represent?", pad=16)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=9)

    # Footnotes
    fig.text(0.5, -0.03,
             "PC1 captures city SIZE — Manila scores high because it has the most of everything, not because care is better per capita.  "
             "PC2 captures government community-care breadth (BHS stations dominate at +0.83).",
             ha="center", fontsize=8, color=C["muted"], style="italic")

    fig.suptitle("Principal Component Analysis — NCR Healthcare Supply Mix",
                 fontsize=16, fontweight="bold", y=1.02, color=C["navy"])
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, "fig2_pca.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — GBM LEARNING CURVE  (Act 2: The Investigation)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_learning_curve(models):
    """
    Training MAE vs LOO test MAE as n_estimators grows.
    Shows where the model stabilises and the generalisation gap.
    """
    lc_n  = models["lc_n"]
    lc_tr = models["lc_tr"]
    lc_te = models["lc_te"]

    fig, ax = plt.subplots(figsize=(10, 6))
    _strip_spines(ax)

    ax.fill_between(lc_n, lc_tr, lc_te,
                    color=C["coral"], alpha=0.10, label="Generalisation gap")
    ax.plot(lc_n, lc_tr, "o-", color=C["teal"], lw=2.2, ms=6,
            label="Training MAE (full data)")
    ax.plot(lc_n, lc_te, "s-", color=C["coral"], lw=2.2, ms=6,
            label="LOO Test MAE (held-out city)")

    # Mark chosen n=50
    chosen_idx  = lc_n.index(50)
    chosen_test = lc_te[chosen_idx]
    ax.axvline(50, color=C["muted"], lw=1.2, ls="--", alpha=0.7)
    ax.annotate(f"  Chosen: n = 50\n  Test MAE = {chosen_test:.3f}",
                xy=(50, chosen_test),
                xytext=(55, chosen_test + 0.012),
                fontsize=8.5, color=C["ink"],
                arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc=C["white"],
                          ec=C["mist"], alpha=0.9))

    ax.set_xlabel("Number of Boosting Trees  (n_estimators)", labelpad=10)
    ax.set_ylabel("Mean Absolute Error  (MAE)", labelpad=10)
    ax.set_title("Gradient Boosting Learns Quickly — and Plateaus at n = 40",
                 pad=20)
    ax.text(0.5, -0.12,
            "Training error (teal) drops toward zero as more trees memorise the data.  "
            "Test error (red) stops improving at ~40 trees — adding more creates overfit, not insight.",
            transform=ax.transAxes, fontsize=8.5, ha="center", color=C["muted"])
    ax.legend(loc="upper right", fontsize=9, frameon=True,
              facecolor=C["white"], edgecolor=C["mist"])
    ax.grid(axis="y", color=C["mist"], linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0, 108)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, "fig3_gbm_learning_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — FEATURE IMPORTANCE  (Act 2: The Investigation)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_feature_importance(models):
    """
    Horizontal bar chart. Best predictor (distance) highlighted.
    94% of the model's decisions hinge on ONE variable.
    """
    imp  = models["imp"]
    idx  = np.argsort(imp)
    best = np.argmax(imp)

    fig, ax = plt.subplots(figsize=(11, 6))
    _strip_spines(ax)

    colors = [C["coral"] if i == best else C["teal"] if imp[i] > 0.01 else "#C5D5DC"
              for i in range(len(imp))]
    bars = ax.barh(
        [FEAT_LABELS[i] for i in idx],
        [imp[i] for i in idx],
        color=[colors[i] for i in idx],
        height=0.58, edgecolor=C["white"], linewidth=0,
    )

    # Value labels — right-aligned, only show if bar is visible
    for bar, i_orig in zip(bars, idx):
        val = imp[i_orig]
        if val > 0.003:
            ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.1f}%",
                    va="center", fontsize=9, color=C["ink"], fontweight="bold")

    ax.set_xlabel("Share of Model's Decision Power  (%)", labelpad=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("One Factor Explains 94% of Accessibility Variation:\nDistance to the Nearest Public Hospital",
                 pad=20)
    ax.text(0.5, -0.12,
            "GBM impurity-based importance  ·  Values sum to 100%  ·  "
            "Poverty ranks 4th — because in Metro Manila, poverty rates are narrow (0.3–4.2%)",
            transform=ax.transAxes, fontsize=8.5, ha="center", color=C["muted"])

    handles = [
        mpatches.Patch(fc=C["coral"], label="Dominant predictor (>90%)"),
        mpatches.Patch(fc=C["teal"],  label="Secondary predictors"),
        mpatches.Patch(fc="#C5D5DC",  label="Near-zero contribution"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              frameon=True, facecolor=C["white"], edgecolor=C["mist"])
    ax.grid(axis="x", color=C["mist"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.08)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, "fig4_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Wd ACCESSIBILITY RANKING  (Act 3: The Finding)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_wd_ranking(df, wd, pov, km, priv):
    """
    Horizontal bar chart. Cities sorted worst to best.
    Annotated with poverty %, distance, and L3 desert flag.
    """
    order  = np.argsort(wd)
    cities = df["city_norm"].values
    vuln   = df["vulnerability_label"].values
    l3d    = df["l3_desert"].values

    fig, ax = plt.subplots(figsize=(12, 8))
    _strip_spines(ax)

    y      = np.arange(len(cities))
    colors = [C[vuln[i]] for i in order]
    bars   = ax.barh(y, wd[order], color=colors, height=0.62,
                     edgecolor=C["white"], linewidth=0)

    # Rank number on the bar
    for j, i in enumerate(order):
        ax.text(0.005, j, f"#{j+1}",
                va="center", fontsize=7.5, color=C["white"],
                fontweight="bold")

    # Right-side annotations: poverty + km
    xmax = 1.0
    for j, i in enumerate(order):
        label_parts = []
        label_parts.append(f"Pov {pov[i]:.1f}%")
        label_parts.append(f"{km[i]:.1f} km")
        if l3d[i]:
            label_parts.append("⚠ L3 Desert")
        ax.text(xmax + 0.01, j,
                "  ·  ".join(label_parts),
                va="center", fontsize=7.5, color=C["ink"])

    ax.set_yticks(y)
    ax.set_yticklabels([cities[i].title() for i in order], fontsize=9.5)
    ax.set_xlabel("Weighted Accessibility Index (Wd)\n"
                  "0 = functionally unreachable  ·  1 = best access", labelpad=10)
    ax.set_xlim(0, 1.45)
    ax.set_title("Who Can Actually Reach a Hospital?\nNCR Cities Ranked by True Accessibility",
                 pad=20)
    ax.text(0.5, -0.08,
            "⚠ L3 Desert = city has zero Level-3 hospitals of any kind  ·  "
            "Pov = poverty incidence  ·  km = distance to nearest public tertiary hospital",
            transform=ax.transAxes, fontsize=8, ha="center", color=C["muted"])

    handles = [mpatches.Patch(fc=C[v], label=f"{v} Vulnerability") for v in VULN_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              frameon=True, facecolor=C["white"], edgecolor=C["mist"])
    ax.grid(axis="x", color=C["mist"], linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, "fig5_wd_ranking.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — LOO VALIDATION SCATTER  (Act 3: The Finding)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_loo_scatter(df, wd, models):
    """
    Predicted vs actual Wd. Every city is its own test case once.
    Colour = absolute error magnitude (green = accurate).
    """
    yp     = models["yp_wd"]
    r2     = models["r2_wd"]
    mae    = models["mae_wd"]
    cities = df["city_norm"].values
    abs_e  = np.abs(yp - wd)

    fig, ax = plt.subplots(figsize=(9, 8))
    _strip_spines(ax, keep=("left", "bottom"))

    cmap = LinearSegmentedColormap.from_list(
        "err_map", ["#2ECC71", "#F4A261", "#E84855"], N=256)
    sc = ax.scatter(wd, yp, c=abs_e, cmap=cmap,
                    vmin=0, vmax=abs_e.max(),
                    s=160, edgecolors=C["white"], linewidths=1.4, zorder=3)

    lo, hi = -0.05, 1.08
    ax.plot([lo, hi], [lo, hi], "--", color=C["muted"], lw=1.4,
            label="Perfect attribution", zorder=2)

    # Subtle MAE band
    ax.fill_between([lo, hi], [lo + mae, hi + mae], [lo - mae, hi - mae],
                    color=C["muted"], alpha=0.07, zorder=1)

    # City labels — avoid overlap with smart placement
    label_offsets = {
        "NAVOTAS":     (+0.03, -0.06),
        "PARANAQUE":   (+0.03, +0.04),
        "MALABON":     (-0.15, +0.04),
        "PASAY":       (-0.12, +0.04),
        "PASIG":       (+0.03, -0.06),
        "MANILA":      (-0.14, -0.06),
        "QUEZON CITY": (+0.03, +0.04),
        "CALOOCAN":    (+0.03, +0.04),
        "LAS PINAS":   (-0.14, +0.04),
        "VALENZUELA":  (+0.03, -0.06),
        "TAGUIG":      (+0.03, +0.04),
    }
    for i, city in enumerate(cities):
        ox, oy = label_offsets.get(city, (0.03, 0.04))
        ax.text(wd[i] + ox, yp[i] + oy, city.title(),
                fontsize=7.5, color=C["ink"], zorder=4)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Absolute Error", fontsize=9, color=C["muted"])
    cbar.ax.yaxis.set_tick_params(color=C["muted"])

    ax.set_xlabel("Actual Wd Score", labelpad=10)
    ax.set_ylabel("Model-Attributed Wd  (LOO CV)", labelpad=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(f"The Model Explains {r2*100:.0f}% of Accessibility Variation\n"
                 f"Across All 17 NCR Cities  (R² = {r2:.3f},  MAE = {mae:.3f})",
                 pad=20)
    ax.text(0.5, -0.10,
            "Each point = one city  ·  Every city was the test case exactly once (LOO CV)  ·  "
            "Green = accurate attribution  ·  Red = larger error",
            transform=ax.transAxes, fontsize=8, ha="center", color=C["muted"])
    ax.legend(fontsize=9, frameon=True, facecolor=C["white"], edgecolor=C["mist"])
    ax.grid(color=C["mist"], linewidth=0.8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, "fig6_loo_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — INTERVENTION HEATMAP  (Act 4: The Call)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_intervention_heatmap(df, wd, pov, km, priv, beds):
    """
    City × indicator matrix. Darker = more urgent.
    Rows sorted worst to best by Wd.
    Each column = one policy lever an LGU can act on.
    """
    order  = np.argsort(wd)           # worst first
    cities = df["city_norm"].values
    l3d    = df["l3_desert"].values

    def n_inv(s):  # raw high = bad → dark
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    def n_fwd(s):  # raw high = good → invert
        mn, mx = s.min(), s.max()
        return 1 - (s - mn) / (mx - mn + 1e-9)

    pub_beds = df["effective_public_beds_per1000"].values

    raw_matrix = np.column_stack([
        n_inv(wd),                     # Wd: low = bad
        n_fwd(pov),                    # Poverty: high = bad
        n_fwd(km),                     # Distance: high = bad
        n_fwd(priv * 100),             # Private%: high = bad
        n_inv(pub_beds),               # Pub beds: low = bad
        l3d.astype(float),             # L3 desert: 1 = bad
    ])

    col_labels = [
        "Accessibility\nScore (Wd)",
        "Poverty\nRate",
        "Distance to\nPublic L3",
        "Private\nOwnership",
        "Public Beds\nper 1,000",
        "L3 Desert\n(no hospital)",
    ]

    M     = raw_matrix[order]
    c_lbl = [cities[i].title() for i in order]

    fig, ax = plt.subplots(figsize=(13, 8))

    cmap = LinearSegmentedColormap.from_list(
        "urgency", ["#E8F6F3", "#F4A261", "#E84855"], N=256)
    im   = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9.5, color=C["ink"])
    ax.set_yticks(np.arange(len(c_lbl)))
    ax.set_yticklabels(c_lbl, fontsize=9.5, color=C["ink"])
    ax.tick_params(axis="both", length=0, pad=8)

    # Cell values as text
    for r in range(M.shape[0]):
        for c_idx in range(M.shape[1]):
            val = M[r, c_idx]
            # L3 desert column: show ⚠ or blank
            if c_idx == 5:
                txt = "⚠" if val == 1 else ""
                ax.text(c_idx, r, txt, ha="center", va="center",
                        fontsize=12, color=C["white"] if val == 1 else C["mist"],
                        fontweight="bold")
            else:
                # Show actual value in readable form
                ax.text(c_idx, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color=C["white"] if val > 0.65 else C["navy"])

    # Separation line after top-5
    ax.axhline(4.5, color=C["white"], lw=2.0)
    ax.text(len(col_labels) - 0.45, 4.8, "← most urgent", fontsize=7.5,
            color=C["muted"], va="bottom", ha="right")

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Urgency Score  (1.0 = highest priority)", fontsize=9,
                   color=C["muted"])
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Medium", "High"], fontsize=8)

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title("Where Should LGUs Intervene First?\n"
                 "Every City, Every Policy Lever — Ranked by Urgency",
                 pad=20)
    ax.text(0.5, -0.06,
            "Darker cell = higher urgency  ·  Cities ranked top-to-bottom: most underserved first  ·  "
            "Each column = one actionable policy lever",
            transform=ax.transAxes, fontsize=8, ha="center", color=C["muted"])
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, "fig7_intervention_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — THE PARADOX DECOMPOSED  (Act 4: The Call)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_paradox(df, pov, priv):
    """
    Private ownership % (x) vs Poverty % (y).
    Upper-right = Healthcare Paradox Zone: residents are POOR and hospitals are PRIVATE.
    """
    pop    = df["population_2020"].values
    cities = df["city_norm"].values
    vuln   = df["vulnerability_label"].values
    l3d    = df["l3_desert"].values
    s_min, s_max = 120, 2000
    sizes  = s_min + (s_max - s_min) * (pop - pop.min()) / (pop.max() - pop.min())

    pov_med  = np.nanmedian(pov)
    priv_med = np.median(priv * 100)

    fig, ax = plt.subplots(figsize=(13, 8))
    _strip_spines(ax)

    # Shading the paradox quadrant
    xmax = 100
    ymax = pov.max() * 1.22
    ax.fill_between([priv_med, xmax], pov_med, ymax,
                    color=C["coral"], alpha=0.07, zorder=0)

    ax.axvline(priv_med, color=C["muted"], lw=1.0, ls="--", alpha=0.6)
    ax.axhline(pov_med,  color=C["muted"], lw=1.0, ls="--", alpha=0.6)

    # Paradox label
    ax.text(priv_med + 1, ymax * 0.94,
            "HEALTHCARE PARADOX ZONE\n"
            "Residents are Poor  +  Hospitals are Private",
            fontsize=8.5, color=C["coral"], fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc=C["white"],
                      ec=C["coral"], alpha=0.85, linewidth=1.2))

    for vuln_grp in ["Low", "Medium", "High"]:
        mask = vuln == vuln_grp
        ax.scatter(priv[mask] * 100, pov[mask], s=sizes[mask],
                   color=C[vuln_grp], alpha=0.78, edgecolors=C["white"],
                   linewidths=1.5, zorder=3, label=vuln_grp)

    # Smart label offsets
    offsets = {
        "NAVOTAS":     (+1.0, +0.08),
        "MANILA":      (-11.0, +0.08),
        "QUEZON CITY": (+1.0, -0.22),
        "PASAY":       (-10.0, -0.22),
        "PASIG":       (+1.0, +0.08),
        "MAKATI":      (+1.0, -0.22),
        "MANDALUYONG": (-13.0, -0.22),
        "CALOOCAN":    (+1.0, +0.08),
        "LAS PINAS":   (-11.0, +0.08),
        "SAN JUAN":    (+1.0, -0.22),
    }
    for i, city in enumerate(cities):
        ox, oy = offsets.get(city, (1.0, 0.08))
        ax.annotate(
            city.title(),
            xy=(priv[i]*100, pov[i]),
            xytext=(priv[i]*100 + ox, pov[i] + oy),
            fontsize=7.5, color=C["ink"],
            arrowprops=dict(arrowstyle="-", color=C["mist"], lw=0.8),
            zorder=5,
        )

    ax.set_xlabel("Private Facility Ownership  (% of all healthcare facilities)",
                  labelpad=10)
    ax.set_ylabel("Poverty Incidence  (%)", labelpad=10)
    ax.set_xlim(40, xmax + 2)
    ax.set_ylim(-0.15, ymax)
    ax.set_title("Cities Where Residents Are Poor AND Their Hospitals Are Private\n"
                 "The Healthcare Paradox — Visible in the Data",
                 pad=20)
    ax.text(0.5, -0.10,
            "Bubble size = population  ·  Upper-right quadrant = structural inaccessibility  ·  "
            "Dashed lines = NCR medians",
            transform=ax.transAxes, fontsize=8, ha="center", color=C["muted"])
    ax.legend(handles=[
        mpatches.Patch(fc=C[v], label=f"{v} Vulnerability") for v in VULN_ORDER
    ], loc="upper left", fontsize=9, frameon=True,
       facecolor=C["white"], edgecolor=C["mist"])
    ax.grid(color=C["mist"], linewidth=0.8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, "fig8_paradox.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("04_viz.py — The Metro Manila Healthcare Paradox")
    print(f"Output → {os.path.abspath(VIZ_DIR)}/")
    print("=" * 60)

    # Load
    candidates = [
        os.path.join(DATA_DIR, "merged_metro_manila.csv"),
        "merged_metro_manila.csv",
    ]
    df_raw = None
    for p in candidates:
        if os.path.exists(p):
            df_raw = pd.read_csv(p)
            print(f"\n  Data: {p}  ({len(df_raw)} cities)")
            break
    if df_raw is None:
        raise FileNotFoundError("Cannot find merged_metro_manila.csv — run 01_data_cleaning.py first.")

    df, X, wd, pov, km, priv, beds, l3, pf = prepare(df_raw)
    pub_beds = df["effective_public_beds_per1000"].values

    print("\n  Fitting models (LOO CV — this takes ~30 sec)...")
    models = fit_models(X, wd, pub_beds)
    print(f"  GBM  R² = {models['r2_wd']:.4f}   MAE = {models['mae_wd']:.4f}")
    print(f"  Ridge R² = {models['r2_beds']:.4f}  (public beds)")

    print("\n  Generating figures...")
    fig1_priority_matrix(df, pov, km)
    fig2_pca(df)
    fig3_learning_curve(models)
    fig4_feature_importance(models)
    fig5_wd_ranking(df, wd, pov, km, priv)
    fig6_loo_scatter(df, wd, models)
    fig7_intervention_heatmap(df, wd, pov, km, priv, beds)
    fig8_paradox(df, pov, priv)

    print("\n" + "=" * 60)
    print("All figures saved:")
    for f in sorted(os.listdir(VIZ_DIR)):
        if f.endswith(".png"):
            kb = os.path.getsize(os.path.join(VIZ_DIR, f)) / 1024
            print(f"  {f:<45} {kb:>5.0f} KB")
    print("=" * 60)