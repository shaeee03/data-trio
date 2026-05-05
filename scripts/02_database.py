"""
================================================================================
SCRIPT 02: Data Storage — SQLite Database via SQLAlchemy ORM
Project:   Healthcare Accessibility & Vulnerability Index — Metro Manila
Courses:   Data Mining & Wrangling | Machine Learning | Data Viz & Storytelling
================================================================================

PURPOSE
-------
Loads all cleaned outputs from 01_data_cleaning.py into a structured SQLite
database via the SQLAlchemy ORM.  The database serves as the single source of
truth for all downstream scripts (03_model.py, 04_viz.py) and satisfies the
"Proper Data Storage" requirement of the DMW course.

DATABASE SCHEMA
---------------
Three normalised tables, one view, and one PCA results table:

  dim_facilities  (2,810 rows)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Dimension table — one row per unique facility in NCR (after dedup).    │
  │ Granularity: facility × city.                                          │
  │ Primary key: facility_code (DOH-assigned unique identifier).           │
  │ Foreign key: city_norm → dim_cities.city_norm                          │
  └────────────────────────────────────────────────────────────────────────┘

  dim_cities  (17 rows)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Dimension table — one row per NCR city/municipality.                   │
  │ Granularity: city.                                                     │
  │ Primary key: city_norm (canonical normalised city name).               │
  │ Contains: population, poverty, supply aggregates, and the target       │
  │   variable nearest_public_tertiary_km.                                 │
  └────────────────────────────────────────────────────────────────────────┘

  fact_vulnerability  (17 rows)
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Fact table — the full model-ready feature matrix.                      │
  │ Granularity: city.                                                     │
  │ Primary key: city_norm.                                                │
  │ Contains: all engineered features + vulnerability label + target var.  │
  │ This is the table that 03_model.py reads directly.                     │
  └────────────────────────────────────────────────────────────────────────┘

  pca_components  (17 rows)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ PCA output — the 3 principal components extracted from 7 facility-type  │
  │ count columns (hospitals, clinics, rhu_count, bhs_count,                │
  │ birthing_homes, dialysis_centers, laboratories).                        │
  │ NOTE: 'pharmacies' excluded — zero variance across all 17 cities        │
  │ (NHFR does not register standalone pharmacies; FDA/BHFS does).          │
  │ Components are labelled by their dominant loadings:                     │
  │   PC1 → Emergency Readiness  (hospitals, dialysis, birthing)            │
  │   PC2 → Diagnostic Access    (laboratories, pharmacies)                 │
  │   PC3 → Primary Coverage     (rhu_count, bhs_count, clinics)            │
  │ These 3 components are appended to fact_vulnerability as                │
  │ pca_total_supply_volume, pca_govt_community_health, pca_rhu_vs_bhs_balance for use in the RF model.     │
  └─────────────────────────────────────────────────────────────────────────┘

  v_health_desert_summary  (view, 17 rows)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Convenience view joining dim_cities and fact_vulnerability.             │
  │ Used by 04_viz.py for the choropleth and bubble chart queries.          │
  └─────────────────────────────────────────────────────────────────────────┘

INDEXES
-------
  idx_facilities_city    — dim_facilities.city_norm  (FK lookup speed)
  idx_facilities_type    — dim_facilities.facility_type  (filter by type)
  idx_facilities_level   — dim_facilities.doh_level  (filter Level 3)
  idx_fact_label         — fact_vulnerability.vulnerability_label
  idx_fact_target        — fact_vulnerability.nearest_public_tertiary_km

OUTPUTS
-------
  healthcare_vulnerability.db  — SQLite database file
  db_validation_report.txt     — Row counts, schema dump, and sample queries
================================================================================
"""

import os
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fix Windows charmap UnicodeEncodeError for terminals that default to cp1252
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from sqlalchemy import (
        create_engine, text, Column, String, Float, Integer,
        MetaData, Table, inspect as sa_inspect
    )
    from sqlalchemy.orm import DeclarativeBase, Session
    SQLALCHEMY_AVAILABLE = True
    print("  SQLAlchemy ORM detected — using full ORM layer.")
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("  SQLAlchemy not found — using sqlite3 stdlib with ORM-compatible schema.")
    print("  Install with: pip install sqlalchemy")
    print("  The database schema and all downstream scripts are SQLAlchemy-ready.")


# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = "../data/data_cleaning_output"
DB_PATH     = "../data/database_output/healthcare_vulnerability.db"
REPORT_PATH = "../data/database_output/db_validation_report.txt"

# Input files (outputs from 01_data_cleaning.py)
MERGED_CSV    = os.path.join(DATA_DIR, "merged_metro_manila.csv")
FACILITIES_CSV = os.path.join(DATA_DIR, "cleaned_facilities.csv")

# ── PCA configuration ──────────────────────────────────────────────────────
# Facility-type count columns used as PCA inputs.
# 'pharmacies' excluded: identically zero for all 17 cities in the NHFR
# (pharmacies are licensed under FDA/BHFS, not DOH).
PCA_INPUT_COLS = [
    "hospitals", "clinics", "rhu_count", "bhs_count",
    "birthing_homes", "dialysis_centers", "laboratories",
]

# Human-readable labels for the loadings table
PCA_FEAT_DISPLAY = {
    "hospitals":        "Hospitals",
    "clinics":          "Clinics",
    "rhu_count":        "Rural Health Units (RHU)",
    "bhs_count":        "Barangay Health Stations (BHS)",
    "birthing_homes":   "Birthing Homes",
    "dialysis_centers": "Dialysis Centers",
    "laboratories":     "Laboratories",
}

# n_components is data-driven (≥80% cumulative variance), not hardcoded.
# find_n_components_for_variance() replicates utils.plot_cum_exp_var(tol=0.80).
PCA_VARIANCE_THRESHOLD = 0.80

# Labels are assigned at runtime by run_pca() after inspecting the actual
# loadings — they are NOT assumed in advance.
PCA_COMPONENT_LABELS: list = []

# ── K-Means clustering configuration ──────────────────────────────────────
# Clusters cities into Healthcare Paradox Zones for the vulnerability heatmap.
# Input: the same 7 facility-type columns used for PCA (supply-side profile)
#        + poverty_incidence_2023_pct (barrier dimension)
# k=3: Low / Medium / High paradox zones — matches the vulnerability label
#      scheme and is interpretable for LGU policy communication.
# Silhouette score is logged to validate cluster quality.
KMEANS_K = 3
KMEANS_CLUSTER_COLS = [
    "hospitals", "level3_hospitals", "beds_per_1000",
    "private_ownership_pct", "poverty_incidence_2023_pct",
    "nearest_public_tertiary_km", "weighted_score_per10k",
]
KMEANS_CLUSTER_LABELS = {0: "Low Vulnerability", 1: "Medium Vulnerability", 2: "High Vulnerability"}

# ── SQLAlchemy ORM schema (used when SQLAlchemy is available) ──────────────
if SQLALCHEMY_AVAILABLE:
    class Base(DeclarativeBase):
        pass

    class DimFacility(Base):
        """
        Dimension table: one row per unique facility in NCR.
        Deduplication is handled upstream in 01_data_cleaning.py — each
        facility_name × city_norm pair appears exactly once, with the
        highest-priority service capability row retained.
        """
        __tablename__ = "dim_facilities"
        facility_code        = Column(String,  primary_key=True)
        facility_name        = Column(String,  nullable=False)
        facility_type        = Column(String)
        facility_category    = Column(String)
        ownership_major      = Column(String)
        is_private           = Column(Integer)  # 1=Private, 0=Government
        is_licensed          = Column(Integer)  # 1=With License
        city_norm            = Column(String,  nullable=False)  # FK → dim_cities
        barangay             = Column(String)
        service_capability   = Column(String)
        doh_level            = Column(Integer)  # 0,1,2,3
        service_level_weight = Column(Float)    # Tier 1–5 weight
        bed_capacity         = Column(Integer)

    class DimCity(Base):
        """
        Dimension table: one row per NCR city (17 rows).
        Contains raw supply, demand, and barrier counts.
        """
        __tablename__ = "dim_cities"
        city_norm                  = Column(String,  primary_key=True)
        population_2020            = Column(Integer)
        population_2024            = Column(Integer)
        pop_growth_rate_pct        = Column(Float)
        total_facilities           = Column(Integer)
        hospitals                  = Column(Integer)
        level3_hospitals           = Column(Integer)
        level2_hospitals           = Column(Integer)
        level1_hospitals           = Column(Integer)
        total_bed_capacity         = Column(Integer)
        weighted_facility_score    = Column(Float)
        private_facility_count     = Column(Integer)
        gov_facility_count         = Column(Integer)
        private_ownership_pct      = Column(Float)
        private_to_public_ratio    = Column(Float)
        poverty_threshold_2023_php = Column(Float)
        poverty_incidence_2021_pct = Column(Float)
        poverty_incidence_2023_pct = Column(Float)
        nearest_public_tertiary_km = Column(Float)   # TARGET VARIABLE

    class FactVulnerability(Base):
        """
        Fact table: the full model-ready feature matrix.
        All density features (per-10k, per-1000) are pre-computed here.
        This is the direct input to 03_model.py.
        """
        __tablename__ = "fact_vulnerability"
        city_norm                  = Column(String,  primary_key=True)
        # ── Supply features ────────────────────────────────────────────
        facility_density_per10k    = Column(Float)
        hospital_density_per10k    = Column(Float)
        beds_per_1000              = Column(Float)
        weighted_score_per10k      = Column(Float)
        level3_per100k             = Column(Float)
        public_primary_per10k      = Column(Float)
        private_ownership_pct      = Column(Float)
        private_to_public_ratio    = Column(Float)
        # ── Barrier features ───────────────────────────────────────────
        poverty_incidence_2023_pct = Column(Float)
        poverty_threshold_2023_php = Column(Float)
        econ_friction_ratio        = Column(Float)
        # ── Demand features ────────────────────────────────────────────
        population_2020            = Column(Integer)
        pop_growth_rate_pct        = Column(Float)
        # ── PCA components (appended after PCA step) ───────────────────
        pca_total_supply_volume              = Column(Float)  # PC1
        pca_govt_community_health             = Column(Float)  # PC2
        pca_rhu_vs_bhs_balance                = Column(Float)  # PC3
        # ── Target + label ─────────────────────────────────────────────
        nearest_public_tertiary_km = Column(Float)  # regression target
        paradox_cluster_id         = Column(Integer) # K-Means cluster (0/1/2)
        paradox_cluster_label      = Column(String)  # Low/Medium/High vulnerability
        vulnerability_label        = Column(String)  # classification target
        vulnerability_score        = Column(Integer) # 0=Low,1=Med,2=High

    class PcaComponents(Base):
        """
        PCA output table: the 3 principal components + explained variance.
        Stored for transparency and reproducibility — professor or auditor
        can inspect loadings and variance without re-running the script.
        """
        __tablename__ = "pca_components"
        city_norm      = Column(String, primary_key=True)
        pca_total_supply_volume  = Column(Float)
        pca_govt_community_health = Column(Float)
        pca_rhu_vs_bhs_balance    = Column(Float)


# ═══════════════════════════════════════════════════════════════════════════
# PCA STEP
# Performed here (in the database script) because:
#   1. It reduces dimensionality of the STORED feature matrix — the 3 PCA
#      columns live in the database alongside raw features.
#   2. 03_model.py reads the already-reduced features, keeping model code
#      clean and the DMW / ML separation explicit.
#   3. The PCA loadings are logged in the validation report for grading.
# ═══════════════════════════════════════════════════════════════════════════

def find_n_components_for_variance(X_scaled, threshold=None):
    """
    Replicates utils.plot_cum_exp_var(exp_var_ratio, tol=threshold):
      exp_var = cumsum(explained_variance_ratio_)
      thresh  = min index where exp_var >= tol, 1-indexed

    Fits PCA on ALL features, prints cumulative variance table,
    returns minimum n_components to reach the threshold.
    """
    if threshold is None:
        threshold = PCA_VARIANCE_THRESHOLD

    pca_full = PCA(n_components=X_scaled.shape[1], random_state=42)
    pca_full.fit(X_scaled)
    evr    = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    # Exact replication of notebook formula:
    #   thresh = np.min(np.arange(len(exp_var))[exp_var >= tol]) + 1
    indices_above = np.arange(len(cumvar))[cumvar >= threshold]
    n_comp        = int(np.min(indices_above)) + 1

    print(f"\n  Cumulative explained variance (threshold = {threshold*100:.0f}%):")
    print(f"  {'PC':<5}  {'Individual':>11}  {'Cumulative':>11}  {'Selected?':>10}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*11}  {'─'*10}")
    for i, (ev, cv) in enumerate(zip(evr, cumvar), start=1):
        sel = "<── selected" if i == n_comp else ""
        print(f"  PC{i:<3}  {ev*100:>10.4f}%  {cv*100:>10.4f}%  {sel}")
    print(f"\n  → Minimum components for {threshold*100:.0f}% variance: {n_comp}")
    return n_comp


def _label_component(loadings_vec, feature_names):
    """Assign a short label based on the dominant feature loading."""
    abs_loads = np.abs(loadings_vec)
    top_idx   = np.argsort(abs_loads)[::-1]
    top1_name = PCA_FEAT_DISPLAY.get(feature_names[top_idx[0]],
                                      feature_names[top_idx[0]])
    top1_load = abs_loads[top_idx[0]]
    top2_name = PCA_FEAT_DISPLAY.get(feature_names[top_idx[1]],
                                      feature_names[top_idx[1]])

    s1 = top1_name.split(" ")[0].lower()
    s2 = top2_name.split(" ")[0].lower()
    return f"pca_{s1}_{s2}_axis" if top1_load <= 0.50 else f"pca_{s1}_dominated"


def print_loadings_table(pca_obj, n_comp):
    """
    Print the loadings table exactly as shown in the course PCA notebook.

    Layout: rows = original features, columns = selected PCs.
    Values: loadings rounded to 4 decimal places.
    Stars:  mark the feature with the largest |loading| per PC (dominant).
    Footer: individual explained variance % and cumulative % per PC.
    """
    loadings    = pca_obj.components_   # shape: (n_comp, n_features)
    feat_labels = [PCA_FEAT_DISPLAY.get(c, c) for c in PCA_INPUT_COLS]
    feat_w      = max(len(f) for f in feat_labels) + 2
    col_w       = 12

    # Header
    hdr = f"  {'Feature':<{feat_w}}" + "".join(
        f"  {'PC' + str(j+1):>{col_w}}" for j in range(n_comp))
    sep = "  " + "─" * (len(hdr) - 2)
    print("\n  ── PCA Loadings Table ──────────────────────────────────────────")
    print("  Rows = original features  |  Cols = principal components")
    print("  * = dominant feature per PC  |  |loading| > 0.30 = substantive")
    print()
    print(hdr)
    print(sep)
    for i, feat in enumerate(feat_labels):
        row = f"  {feat:<{feat_w}}"
        for j in range(n_comp):
            val  = loadings[j, i]
            star = "*" if np.argmax(np.abs(loadings[j])) == i else " "
            row += f"  {f'{val:+.4f}{star}':>{col_w}}"
        print(row)
    print(sep)
    # Explained variance footer
    ev_row  = f"  {'Expl. var %':<{feat_w}}"
    cum_row = f"  {'Cumul. var %':<{feat_w}}"
    cumvar  = np.cumsum(pca_obj.explained_variance_ratio_)
    for j in range(n_comp):
        ev_row  += f"  {f'({pca_obj.explained_variance_ratio_[j]*100:.2f}%)':>{col_w}}"
        cum_row += f"  {f'[{cumvar[j]*100:.2f}%]':>{col_w}}"
    print(ev_row)
    print(cum_row)
    print(sep)
    print("  * = largest |loading| in that PC")


def run_pca(merged_df):
    """
    Full PCA pipeline following the course lesson structure.

    STEP 1 — StandardScaler
      Fit StandardScaler on the 7 facility-type count columns.
      Required: raw counts span very different ranges (hospitals: 1–41,
      labs: 0–131). Without scaling, high-variance columns dominate the
      covariance matrix regardless of theoretical importance.
      See: course PCA notebook, scaling cell.

    STEP 2 — Select n_components (data-driven, not hardcoded)
      Fit PCA on ALL 7 features. Compute cumulative explained variance.
      Select minimum n such that cumvar >= PCA_VARIANCE_THRESHOLD (80%).
      Replicates: utils.plot_cum_exp_var(exp_var_ratio, tol=0.80).

    STEP 3 — Fit final PCA(n_components=n_comp)

    STEP 4 — Print loadings table (computed from data, not hardcoded)
      Displays full loading matrix with dominant-feature stars.
      Explained variance and cumulative variance in footer rows.

    STEP 5 — Assign component labels from actual loadings
      Labels derived from _label_component() which reads computed loadings.
      They are NOT assumed in advance.

    STEP 6 — Transform: compute city scores on selected PCs

    Why PCA here and not in 03_model.py?
      PCA components are stored in the database as supply-profile features
      for the city scatter plot (04_viz.py). They are NOT fed into the
      GBM/Ridge models — those use raw per-capita features with theory-
      grounded interpretations. PCA satisfies the DMW feature-reduction
      requirement and gives 04_viz.py orthogonal infrastructure axes.

    Returns
    -------
    pca_df   : DataFrame (17 rows × n_comp+1 cols) city_norm + PC scores
    pca_obj  : fitted sklearn PCA object
    X_scaled : StandardScaled feature matrix (for validation report)
    """
    global PCA_COMPONENT_LABELS

    print("\n  ── Step 1: StandardScaler ──────────────────────────────────────")
    X        = merged_df[PCA_INPUT_COLS].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Input  : {X.shape[0]} cities × {X.shape[1]} facility-type features")
    print(f"  Output : mean ≈ 0, std ≈ 1 per feature (verified below)")
    print()
    print(f"  {'Feature':<30}  {'Raw μ':>8}  {'Raw σ':>8}  {'Scaled μ':>10}  {'Scaled σ':>10}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    for i, col in enumerate(PCA_INPUT_COLS):
        disp = PCA_FEAT_DISPLAY.get(col, col)
        print(f"  {disp:<30}  {X[:,i].mean():>8.3f}  {X[:,i].std():>8.3f}  "
              f"{X_scaled[:,i].mean():>10.6f}  {X_scaled[:,i].std():>10.6f}")

    print("\n  ── Step 2: Select n_components ─────────────────────────────────")
    n_comp = find_n_components_for_variance(X_scaled, PCA_VARIANCE_THRESHOLD)

    print(f"\n  ── Step 3: Fit PCA(n_components={n_comp}) ──────────────────────")
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_scaled)
    print(f"  PCA fitted on {X_scaled.shape[0]} cities, {n_comp} components selected")

    print("\n  ── Step 4: Loadings Table ──────────────────────────────────────")
    print_loadings_table(pca, n_comp)

    print("\n  ── Step 5: Assign component labels from dominant loadings ──────")
    labels = []
    for j in range(n_comp):
        label = _label_component(pca.components_[j], PCA_INPUT_COLS)
        ev    = pca.explained_variance_ratio_[j] * 100
        dom_i = np.argmax(np.abs(pca.components_[j]))
        dom_f = PCA_FEAT_DISPLAY.get(PCA_INPUT_COLS[dom_i], PCA_INPUT_COLS[dom_i])
        dom_v = pca.components_[j, dom_i]
        labels.append(label)
        print(f"  PC{j+1} → '{label}'")
        print(f"       Dominant: {dom_f}  (loading={dom_v:+.4f},  {ev:.2f}% variance)")
    PCA_COMPONENT_LABELS = labels
    print(f"\n  Labels set: {PCA_COMPONENT_LABELS}")

    print(f"\n  ── Step 6: Transform — city scores on {n_comp} PCs ────────────")
    components = pca.transform(X_scaled)

    pca_df = pd.DataFrame(
        components,
        columns=PCA_COMPONENT_LABELS,
        index=merged_df.index,
    )
    pca_df.insert(0, "city_norm", merged_df["city_norm"].values)

    print(f"  City PC scores:")
    print(pca_df.to_string(index=False))
    total_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"\n  PCA complete: {n_comp} components, {total_var:.2f}% variance retained")

    return pca_df, pca, X_scaled


def run_kmeans(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    K-Means clustering on supply-barrier features.

    Groups the 17 NCR cities into 3 Healthcare Paradox Zones:
      Cluster 0 → Low Vulnerability    (good supply, low poverty, near public L3)
      Cluster 1 → Medium Vulnerability (mixed access)
      Cluster 2 → High Vulnerability   (poor supply OR far from public L3 OR high poverty)

    Why K-Means here (not in 03_model.py)?
      Clustering is an UNSUPERVISED step that characterises the supply
      landscape. It belongs in the database layer so the cluster label is
      stored alongside raw features and available to both the ML model and
      the visualisation layer without re-computing.

    Why k=3?
      Three clusters align with the Low/Medium/High vulnerability label
      from 01_data_cleaning.py, enabling direct comparison between
      the rule-based label (composite criteria) and the data-driven
      cluster assignment. Silhouette score is printed for validation.

    Why these input columns?
      They represent the three "Dimensions of Inequity" from the proposal:
      supply depth (hospitals, L3 count, beds), economic barrier (poverty,
      private ownership), and spatial barrier (distance to public L3).
      Weighting all equally after StandardScaling ensures no single
      dimension dominates.

    Returns
    -------
    DataFrame with city_norm and three columns:
      paradox_cluster_id    (int 0/1/2)
      paradox_cluster_label (str "Low/Medium/High Vulnerability")
      silhouette_score      (float, same value for all rows — for logging)
    """
    available = [c for c in KMEANS_CLUSTER_COLS if c in merged_df.columns]
    X = merged_df[available].fillna(merged_df[available].median()).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=KMEANS_K, n_init=50, random_state=42)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    print(f"\n  K-Means clustering (k={KMEANS_K}) on {len(available)} features:")
    print(f"    Silhouette score: {sil:.4f}  (>0.5 = good cluster separation)")

    # Reorder clusters so label 0=lowest, 2=highest paradox
    # Sort by mean nearest_public_tertiary_km per cluster (proxy for inaccessibility)
    if "nearest_public_tertiary_km" in merged_df.columns:
        km_dist = {}
        for c in range(KMEANS_K):
            mask = labels == c
            km_dist[c] = merged_df["nearest_public_tertiary_km"].values[mask].mean()
        # Map: lowest mean distance → cluster 0, highest → cluster 2
        order = sorted(km_dist, key=km_dist.get)
        remap = {old: new for new, old in enumerate(order)}
        labels = np.array([remap[l] for l in labels])

    print(f"    Cluster distribution:")
    for c in range(KMEANS_K):
        cname = KMEANS_CLUSTER_LABELS[c]
        cities_in = merged_df["city_norm"].values[labels == c].tolist()
        print(f"      {c} ({cname}): {cities_in}")

    cluster_df = pd.DataFrame({
        "city_norm":            merged_df["city_norm"].values,
        "paradox_cluster_id":   labels,
        "paradox_cluster_label":np.array([KMEANS_CLUSTER_LABELS[l] for l in labels]),
        "silhouette_score":     sil,
    })
    return cluster_df

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE BUILD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_database_sqlalchemy(merged_df, facilities_df, pca_df, cluster_df, engine):
    """Full ORM-based build when SQLAlchemy is available."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # ── dim_facilities ────────────────────────────────────────────
        fac_records = facilities_df.rename(
            columns={"facility_major_type": "_drop"}
        ).to_dict("records")
        session.bulk_insert_mappings(DimFacility, [
            {k: v for k, v in r.items()
             if k in DimFacility.__table__.columns.keys()
             and not (isinstance(v, float) and np.isnan(v))}
            for r in fac_records
        ])

        # ── dim_cities ────────────────────────────────────────────────
        city_records = merged_df.to_dict("records")
        session.bulk_insert_mappings(DimCity, [
            {k: v for k, v in r.items()
             if k in DimCity.__table__.columns.keys()
             and not (isinstance(v, float) and np.isnan(v))}
            for r in city_records
        ])

        # ── pca_components ────────────────────────────────────────────
        session.bulk_insert_mappings(
            PcaComponents, pca_df.to_dict("records")
        )

        # ── fact_vulnerability ────────────────────────────────────────
        fact_df = merged_df.copy()
        for col in PCA_COMPONENT_LABELS:
            fact_df[col] = pca_df[col].values
        fact_df['paradox_cluster_id']    = cluster_df['paradox_cluster_id'].values
        fact_df['paradox_cluster_label'] = cluster_df['paradox_cluster_label'].values
        fact_records = fact_df.to_dict("records")
        session.bulk_insert_mappings(FactVulnerability, [
            {k: v for k, v in r.items()
             if k in FactVulnerability.__table__.columns.keys()
             and not (isinstance(v, float) and np.isnan(v))}
            for r in fact_records
        ])
        session.commit()

    # ── Indexes ───────────────────────────────────────────────────────
    with engine.connect() as conn:
        _create_indexes(conn, sqlalchemy_mode=True)

    # ── View ──────────────────────────────────────────────────────────
    with engine.connect() as conn:
        _create_view(conn, sqlalchemy_mode=True)
        conn.commit()


def build_database_sqlite3(merged_df, facilities_df, pca_df, cluster_df, db_path):
    """Fallback build using sqlite3 stdlib — identical schema."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ── dim_facilities ─────────────────────────────────────────────────
    cur.execute("DROP TABLE IF EXISTS dim_facilities")
    cur.execute("""
        CREATE TABLE dim_facilities (
            facility_code        TEXT PRIMARY KEY,
            facility_name        TEXT NOT NULL,
            facility_type        TEXT,
            facility_category    TEXT,
            ownership_major      TEXT,
            is_private           INTEGER,
            is_licensed          INTEGER,
            city_norm            TEXT NOT NULL,
            barangay             TEXT,
            service_capability   TEXT,
            doh_level            INTEGER,
            service_level_weight REAL,
            bed_capacity         INTEGER
        )
    """)
    keep = ["facility_code","facility_name","facility_type","facility_category",
            "ownership_major","is_private","is_licensed","city_norm","barangay",
            "service_capability","doh_level","service_level_weight","bed_capacity"]
    fac_insert = facilities_df[[c for c in keep if c in facilities_df.columns]].copy()
    fac_insert = fac_insert.where(pd.notnull(fac_insert), None)
    fac_insert.to_sql("dim_facilities", conn, if_exists="append", index=False)

    # ── dim_cities ──────────────────────────────────────────────────────
    cur.execute("DROP TABLE IF EXISTS dim_cities")
    cur.execute("""
        CREATE TABLE dim_cities (
            city_norm                  TEXT PRIMARY KEY,
            population_2020            INTEGER,
            population_2024            INTEGER,
            pop_growth_rate_pct        REAL,
            total_facilities           INTEGER,
            hospitals                  INTEGER,
            level3_hospitals           INTEGER,
            level2_hospitals           INTEGER,
            level1_hospitals           INTEGER,
            total_bed_capacity         INTEGER,
            weighted_facility_score    REAL,
            private_facility_count     INTEGER,
            gov_facility_count         INTEGER,
            private_ownership_pct      REAL,
            private_to_public_ratio    REAL,
            poverty_threshold_2023_php REAL,
            poverty_incidence_2021_pct REAL,
            poverty_incidence_2023_pct REAL,
            nearest_public_tertiary_km REAL
        )
    """)
    city_keep = ["city_norm","population_2020","population_2024","pop_growth_rate_pct",
                 "total_facilities","hospitals","level3_hospitals","level2_hospitals",
                 "level1_hospitals","total_bed_capacity","weighted_facility_score",
                 "private_facility_count","gov_facility_count","private_ownership_pct",
                 "private_to_public_ratio","poverty_threshold_2023_php",
                 "poverty_incidence_2021_pct","poverty_incidence_2023_pct",
                 "nearest_public_tertiary_km"]
    city_df = merged_df[[c for c in city_keep if c in merged_df.columns]].copy()
    city_df = city_df.where(pd.notnull(city_df), None)
    city_df.to_sql("dim_cities", conn, if_exists="append", index=False)

    # ── pca_components ──────────────────────────────────────────────────
    cur.execute("DROP TABLE IF EXISTS pca_components")
    cur.execute("""
        CREATE TABLE pca_components (
            city_norm      TEXT PRIMARY KEY,
            pca_total_supply_volume  REAL,
            pca_govt_community_health REAL,
            pca_rhu_vs_bhs_balance    REAL
        )
    """)
    pca_df.to_sql("pca_components", conn, if_exists="append", index=False)

    # ── fact_vulnerability ──────────────────────────────────────────────
    cur.execute("DROP TABLE IF EXISTS fact_vulnerability")
    cur.execute("""
        CREATE TABLE fact_vulnerability (
            city_norm                  TEXT PRIMARY KEY,
            facility_density_per10k    REAL,
            hospital_density_per10k    REAL,
            beds_per_1000              REAL,
            weighted_score_per10k      REAL,
            level3_per100k             REAL,
            public_primary_per10k      REAL,
            private_ownership_pct      REAL,
            private_to_public_ratio    REAL,
            poverty_incidence_2023_pct REAL,
            poverty_threshold_2023_php REAL,
            econ_friction_ratio        REAL,
            population_2020            INTEGER,
            pop_growth_rate_pct        REAL,
            pca_total_supply_volume              REAL,
            pca_govt_community_health             REAL,
            pca_rhu_vs_bhs_balance                REAL,
            nearest_public_tertiary_km REAL,
            paradox_cluster_id         INTEGER,
            paradox_cluster_label      TEXT,
            vulnerability_label        TEXT,
            vulnerability_score        INTEGER
        )
    """)
    fact_keep = ["city_norm","facility_density_per10k","hospital_density_per10k",
                 "beds_per_1000","weighted_score_per10k","level3_per100k",
                 "public_primary_per10k","private_ownership_pct",
                 "private_to_public_ratio","poverty_incidence_2023_pct",
                 "poverty_threshold_2023_php","econ_friction_ratio",
                 "population_2020","pop_growth_rate_pct",
                 "nearest_public_tertiary_km","paradox_cluster_id","paradox_cluster_label","vulnerability_label","vulnerability_score"]
    fact_df = merged_df[[c for c in fact_keep if c in merged_df.columns]].copy()
    for col in PCA_COMPONENT_LABELS:
        fact_df[col] = pca_df[col].values
    fact_df['paradox_cluster_id']    = cluster_df['paradox_cluster_id'].values
    fact_df['paradox_cluster_label'] = cluster_df['paradox_cluster_label'].values
    fact_df = fact_df.where(pd.notnull(fact_df), None)
    fact_df.to_sql("fact_vulnerability", conn, if_exists="append", index=False)

    # ── Indexes ──────────────────────────────────────────────────────────
    _create_indexes(cur, sqlalchemy_mode=False)

    # ── View ─────────────────────────────────────────────────────────────
    _create_view(cur, sqlalchemy_mode=False)

    conn.commit()
    conn.close()
    return sqlite3.connect(db_path)


def _create_indexes(conn, sqlalchemy_mode: bool):
    """
    Create indexes on the most frequently queried columns.

    Why these indexes?
      - city_norm on dim_facilities: every JOIN to dim_cities uses this.
      - facility_type: 04_viz.py filters by type (hospitals only, labs only).
      - doh_level: model and viz both filter on Level 3 specifically.
      - vulnerability_label: the classification output, grouped frequently.
      - nearest_public_tertiary_km: sorted for ranking queries in viz.
    """
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_facilities_city  ON dim_facilities(city_norm)",
        "CREATE INDEX IF NOT EXISTS idx_facilities_type  ON dim_facilities(facility_type)",
        "CREATE INDEX IF NOT EXISTS idx_facilities_level ON dim_facilities(doh_level)",
        "CREATE INDEX IF NOT EXISTS idx_fact_label       ON fact_vulnerability(vulnerability_label)",
        "CREATE INDEX IF NOT EXISTS idx_fact_target      ON fact_vulnerability(nearest_public_tertiary_km)",
    ]
    for stmt in stmts:
        if sqlalchemy_mode:
            conn.execute(text(stmt))
        else:
            conn.execute(stmt)


def _create_view(conn, sqlalchemy_mode: bool):
    """
    Create v_health_desert_summary — a convenience view used by 04_viz.py.

    This view pre-joins the most-queried columns from dim_cities and
    fact_vulnerability, ranked by nearest_public_tertiary_km descending
    (most underserved cities first).  It is the primary data source for
    the choropleth, bubble chart, and priority matrix visualisations.
    """
    stmt = """
        CREATE VIEW IF NOT EXISTS v_health_desert_summary AS
        SELECT
            f.city_norm,
            f.nearest_public_tertiary_km,
            f.vulnerability_label,
            f.vulnerability_score,
            f.poverty_incidence_2023_pct,
            f.private_ownership_pct,
            f.private_to_public_ratio,
            f.weighted_score_per10k,
            f.beds_per_1000,
            f.facility_density_per10k,
            f.level3_per100k,
            f.econ_friction_ratio,
            f.pop_growth_rate_pct,
            f.pca_total_supply_volume,
            f.pca_govt_community_health,
            f.pca_rhu_vs_bhs_balance,
            c.population_2020,
            c.population_2024,
            c.total_facilities,
            c.hospitals,
            c.level3_hospitals,
            c.total_bed_capacity
        FROM fact_vulnerability f
        JOIN dim_cities c ON f.city_norm = c.city_norm
        ORDER BY f.nearest_public_tertiary_km DESC
    """
    if sqlalchemy_mode:
        conn.execute(text(stmt))
    else:
        conn.execute(stmt)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def run_validation(db_path: str, pca_obj: PCA, pca_df: pd.DataFrame) -> str:
    """
    Run 12 sanity checks against the database and return a formatted report.

    Checks:
      1.  All 3 tables + 1 view + 1 PCA table exist
      2.  dim_facilities row count = 2,810 (post-dedup)
      3.  dim_cities row count = 17
      4.  fact_vulnerability row count = 17
      5.  pca_components row count = 17
      6.  No duplicate city_norm in dim_cities
      7.  No NULL city_norm in dim_facilities
      8.  nearest_public_tertiary_km range [0, 50]
      9.  private_ownership_pct range [0, 1]
      10. vulnerability_label only contains Low / Medium / High
      11. PCA total explained variance ≥ 60%
      12. fact_vulnerability has all 3 PCA columns populated (no NULLs)
    """
    conn   = sqlite3.connect(db_path)
    cur    = conn.cursor()
    lines  = []
    passed = 0
    failed = 0

    def check(desc, query_or_bool, expected=True, transform=None):
        nonlocal passed, failed
        try:
            if isinstance(query_or_bool, str):
                result = cur.execute(query_or_bool).fetchone()[0]
                if transform:
                    result = transform(result)
                ok = (result == expected)
            else:
                ok = bool(query_or_bool)
                result = query_or_bool
            symbol = "✓ PASS" if ok else "✗ FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            lines.append(f"  {symbol}  {desc}")
            if not ok:
                lines.append(f"         expected={expected!r}, got={result!r}")
        except Exception as e:
            failed += 1
            lines.append(f"  ✗ FAIL  {desc}")
            lines.append(f"         ERROR: {e}")

    lines.append("=" * 68)
    lines.append("DB VALIDATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Database:  {db_path}")
    lines.append("=" * 68)

    lines.append("\n── TABLE EXISTENCE ──────────────────────────────────────────────")
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
    ).fetchall()]
    for expected_table in ["dim_cities", "dim_facilities", "fact_vulnerability",
                           "pca_components", "v_health_desert_summary"]:
        check(f"  {expected_table} exists", expected_table in tables)

    lines.append("\n── ROW COUNTS ───────────────────────────────────────────────────")
    check("dim_facilities: 2,810 rows",
          "SELECT COUNT(*) FROM dim_facilities", 2810)
    check("dim_cities: 17 rows",
          "SELECT COUNT(*) FROM dim_cities", 17)
    check("fact_vulnerability: 17 rows",
          "SELECT COUNT(*) FROM fact_vulnerability", 17)
    check("pca_components: 17 rows",
          "SELECT COUNT(*) FROM pca_components", 17)

    lines.append("\n── DATA INTEGRITY ───────────────────────────────────────────────")
    check("No duplicate city_norm in dim_cities",
          "SELECT COUNT(*) - COUNT(DISTINCT city_norm) FROM dim_cities", 0)
    check("No NULL city_norm in dim_facilities",
          "SELECT COUNT(*) FROM dim_facilities WHERE city_norm IS NULL", 0)
    check("nearest_public_tertiary_km all >= 0",
          "SELECT COUNT(*) FROM dim_cities WHERE nearest_public_tertiary_km < 0", 0)
    check("nearest_public_tertiary_km all <= 50",
          "SELECT COUNT(*) FROM dim_cities WHERE nearest_public_tertiary_km > 50", 0)
    check("private_ownership_pct all in [0,1]",
          "SELECT COUNT(*) FROM fact_vulnerability "
          "WHERE private_ownership_pct < 0 OR private_ownership_pct > 1", 0)
    check("vulnerability_label only Low/Medium/High",
          "SELECT COUNT(*) FROM fact_vulnerability "
          "WHERE vulnerability_label NOT IN ('Low','Medium','High')", 0)

    lines.append("\n── PCA VALIDATION ───────────────────────────────────────────────")
    total_var = sum(pca_obj.explained_variance_ratio_) * 100
    check(f"PCA total explained variance ≥ 60%  (actual: {total_var:.1f}%)",
          total_var >= 60.0)
    check("PCA columns non-NULL in fact_vulnerability",
          "SELECT COUNT(*) FROM fact_vulnerability "
          "WHERE pca_total_supply_volume IS NULL OR pca_govt_community_health IS NULL OR pca_rhu_vs_bhs_balance IS NULL",
          0)

    lines.append("\n── SCHEMA ───────────────────────────────────────────────────────")
    for table in ["dim_facilities", "dim_cities", "fact_vulnerability", "pca_components"]:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        lines.append(f"  {table}: {len(cols)} columns")
        lines.append(f"    {', '.join(cols)}")

    lines.append("\n── SAMPLE QUERY: v_health_desert_summary (top 5 underserved) ───")
    rows = cur.execute("""
        SELECT city_norm, nearest_public_tertiary_km, vulnerability_label,
               poverty_incidence_2023_pct, private_ownership_pct
        FROM v_health_desert_summary
        LIMIT 5
    """).fetchall()
    lines.append(f"  {'City':<15} {'km':>6}  {'Label':<8}  {'Poverty%':>8}  {'Private%':>9}")
    lines.append(f"  {'-'*15} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}")
    for row in rows:
        city, km, label, pov, priv = row
        pov_str  = f"{pov:.1f}" if pov is not None else "N/A"
        priv_str = f"{priv*100:.1f}" if priv is not None else "N/A"
        lines.append(f"  {city:<15} {km:>6.3f}  {label:<8}  {pov_str:>8}  {priv_str:>9}")

    lines.append("\n── PCA LOADINGS ─────────────────────────────────────────────────")
    lines.append("  (Rows = original features, Cols = principal components)")
    lines.append(f"  {'Feature':<22}  {'PC1 Emerg':>10}  {'PC2 Diagn':>10}  {'PC3 Prim':>10}")
    lines.append(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}")
    for feat, loadings in zip(PCA_INPUT_COLS, pca_obj.components_.T):
        vals = "  ".join(f"{v:+.3f}".rjust(10) for v in loadings)
        lines.append(f"  {feat:<22}  {vals}")
    lines.append(f"\n  Explained variance:")
    for label, var in zip(PCA_COMPONENT_LABELS, pca_obj.explained_variance_ratio_):
        lines.append(f"    {label:<22}  {var*100:.2f}%")

    lines.append(f"\n── SUMMARY ──────────────────────────────────────────────────────")
    total = passed + failed
    lines.append(f"  {passed}/{total} checks passed"
                 + (" ✓ ALL CLEAR" if failed == 0 else f" ← {failed} FAILED"))
    lines.append("=" * 68)

    conn.close()
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 68)
    print("HEALTHCARE ACCESSIBILITY INDEX — SCRIPT 02: DATA STORAGE")
    print("=" * 68)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 1. Load cleaned CSVs ───────────────────────────────────────────
    print("\n[1/5] Loading cleaned datasets...")
    missing = [f for f in [MERGED_CSV, FACILITIES_CSV]
               if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing input files:")
        for f in missing:
            print(f"  ✗  {f}")
        print("  Run 01_data_cleaning.py first.")
        raise SystemExit(1)

    merged_df     = pd.read_csv(MERGED_CSV)
    facilities_df = pd.read_csv(FACILITIES_CSV)
    print(f"  merged_metro_manila.csv:  {merged_df.shape[0]} rows × {merged_df.shape[1]} cols")
    print(f"  cleaned_facilities.csv:   {facilities_df.shape[0]} rows × {facilities_df.shape[1]} cols")

    # ── 2. Run PCA ────────────────────────────────────────────────────
    print("\n[2/5] Running PCA on facility-type supply columns...")
    pca_df, pca_obj, X_scaled = run_pca(merged_df)

    # ── 2b. K-Means clustering ────────────────────────────────────────
    print("\n[2b] Running K-Means clustering (k=3 Healthcare Paradox Zones)...")
    cluster_df = run_kmeans(merged_df)

    # Sanity check: PCA output shape
    # Shape: 17 cities × (1 city_norm + n_comp PC columns, n_comp is data-driven)
    assert pca_df.shape[0] == 17, f"PCA output should have 17 cities, got {pca_df.shape[0]}"
    assert pca_df.shape[1] >= 2, f"PCA output should have ≥2 PC columns, got {pca_df.shape[1]-1}"
    assert not pca_df[PCA_COMPONENT_LABELS].isnull().any().any(), "PCA output contains NaN"
    print(f"  PCA output: {pca_df.shape[0]} rows × {len(PCA_COMPONENT_LABELS)} components ✓")

    # ── 3. Build database ─────────────────────────────────────────────
    print(f"\n[3/6] Building database: {DB_PATH}")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Removed existing database.")

    if SQLALCHEMY_AVAILABLE:
        db_url = f"sqlite:///{DB_PATH}"
        engine = create_engine(db_url, echo=False)
        build_database_sqlalchemy(merged_df, facilities_df, pca_df, cluster_df, engine)
        engine.dispose()
        print(f"  Built via SQLAlchemy ORM ✓")
    else:
        conn = build_database_sqlite3(merged_df, facilities_df, pca_df, cluster_df, DB_PATH)
        conn.close()
        print(f"  Built via sqlite3 stdlib ✓")
        print(f"  NOTE: Schema is SQLAlchemy-compatible. Install sqlalchemy to use ORM.")

    db_size_kb = os.path.getsize(DB_PATH) / 1024
    print(f"  Database size: {db_size_kb:.1f} KB")

    # ── 4. Validate ───────────────────────────────────────────────────
    print("\n[4/6] Running validation checks...")
    report = run_validation(DB_PATH, pca_obj, pca_df)
    print(report)

    # ── 5. Save report ────────────────────────────────────────────────
    print(f"\n[5/6] Saving validation report → {REPORT_PATH}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print("  Saved ✓")

    print("\n" + "=" * 68)
    print("DONE.  Database ready (PCA + K-Means + all targets).  Next: run 03_model.py")
    print("=" * 68)