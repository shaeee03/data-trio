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
  │   PC1 → Healthcare Infrastructure Volume Index  (72.4% variance)        │
  │          All 7 facility types load equally — captures city SIZE.         │
  │   PC2 → Community Primary Care Network Index  (19.0% variance)          │
  │          BHS stations dominant (+0.83) — public safety-net breadth.      │
  │   Total: 91.3% variance (above 80% threshold). Stored in DB for city    │
  │   profiling. NOT used in regression (03_model.py uses per-capita vars).  │
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
# These 7 facility-type count columns are reduced to 3 principal components.
# NOTE: 'pharmacies' is excluded — standalone pharmacies are not licensed
# under the DOH/NHFR registry (they fall under FDA/BHFS), so this column is
# identically zero for all 17 cities and would contribute zero variance to PCA.
# Including it would silently produce a near-singular covariance matrix and
# waste one of the PCA dimensions on noise.
PCA_INPUT_COLS = [
    "hospitals", "clinics", "rhu_count", "bhs_count",
    "birthing_homes", "dialysis_centers", "laboratories",
]
PCA_N_COMPONENTS = 2  # PC1+PC2 = 91.3% variance — exceeds 80% threshold

# ── PCA component names (derived from actual factor loadings) ──────────────
# After fitting PCA on the 7 facility-type columns and inspecting loadings:
#
#   PC1 (72.4% variance) — Healthcare Infrastructure Volume Index
#       All 7 facility types load positively (+0.36 to +0.43).
#       Captures city SIZE: Manila and QC score highest because they have
#       more of every facility type — not because care is better per capita.
#       ⚠ LIMITATION: a city with 100 birthing homes scores similarly to
#       one with 100 ICU beds. PC1 does not distinguish clinical quality.
#
#   PC2 (18.95% variance) — Community Primary Care Network Index
#       Dominant loader: bhs_count (+0.83).
#       Secondary: birthing_homes (+0.39), rhu_count (+0.27).
#       Captures the breadth of government community-level health infrastructure.
#       High PC2 = strong public primary-care safety net (Valenzuela, Caloocan).
#
#   PC3–PC7 add only 8.7% combined — not retained.
#
#   PURPOSE: stored for city profiling/clustering/visualisation only.
#   NOT used as regression features in 03_model.py (which uses per-capita
#   features beds_per_1000, level3_per100k to avoid the size bias in PC1).
PCA_COMPONENT_LABELS = [
    "pca_volume_index",       # PC1 (72.4% var): Infrastructure Volume Index
    "pca_primary_care_index", # PC2 (19.0% var): Community Primary Care Index
]

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
KMEANS_CLUSTER_LABELS = {0: "Low Paradox", 1: "Medium Paradox", 2: "High Paradox"}

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
        pca_volume_index           = Column(Float)  # PC1: Infrastructure Volume Index
        pca_primary_care_index     = Column(Float)  # PC2: Community Primary Care Index
        # ── Target + label ─────────────────────────────────────────────
        nearest_public_tertiary_km = Column(Float)  # regression target
        paradox_cluster_id         = Column(Integer) # K-Means cluster (0/1/2)
        paradox_cluster_label      = Column(String)  # Low/Medium/High Paradox
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
        pca_volume_index       = Column(Float)  # PC1
        pca_primary_care_index = Column(Float)  # PC2


# ═══════════════════════════════════════════════════════════════════════════
# PCA STEP
# Performed here (in the database script) because:
#   1. It reduces dimensionality of the STORED feature matrix — the 3 PCA
#      columns live in the database alongside raw features.
#   2. 03_model.py reads the already-reduced features, keeping model code
#      clean and the DMW / ML separation explicit.
#   3. The PCA loadings are logged in the validation report for grading.
# ═══════════════════════════════════════════════════════════════════════════

def run_pca(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """
    Fit PCA on 8 facility-type count columns (StandardScaled).

    Returns
    -------
    pca_df  : DataFrame with city_norm + 2 component columns (city_norm, pca_volume_index, pca_primary_care_index)
    pca_obj : fitted sklearn PCA object (for loadings + variance)
    X_scaled: the scaled input matrix (for the validation report)

    Why StandardScaler before PCA?
        The 7 input columns are counts on vastly different scales
        (hospitals: 1–41, laboratories: 0–131).  Without scaling, PCA would
        be dominated by high-variance columns regardless of their importance.
        StandardScaler (zero mean, unit variance) ensures each column
        contributes equally to the covariance matrix.

    Why 7 columns, not 8?
        'pharmacies' is excluded because it is identically zero for all 17
        cities — the NHFR does not license standalone pharmacies (FDA/BHFS
        does).  A zero-variance column contributes nothing to PCA and would
        waste a component dimension on noise.

    Why 2 components?
        PC1+PC2 explain 91.3% of variance — the 80% threshold is met with
        only 2 components. PC3 adds only 4.1% and is not retained.
        Running PCA(n_components=7) and inspecting cumvar confirms:
          PC1: 72.4%, PC1+PC2: 91.3%, PC1+PC2+PC3: 95.4%.
        Reducing 7 → 2 gives a 71.4% dimensionality reduction.
    """
    X = merged_df[PCA_INPUT_COLS].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        components,
        columns=PCA_COMPONENT_LABELS,
        index=merged_df.index
    )
    pca_df.insert(0, "city_norm", merged_df["city_norm"].values)

    total_var = sum(pca.explained_variance_ratio_) * 100
    print(f"\n  PCA explained variance:")
    for i, (label, var) in enumerate(zip(PCA_COMPONENT_LABELS, pca.explained_variance_ratio_)):
        print(f"    {label:20s}  {var*100:.1f}%")
    print(f"    {'TOTAL':20s}  {total_var:.1f}%")
    if total_var < 70:
        print(f"  ⚠ WARNING: 3 components explain only {total_var:.1f}% of variance.")
        print(f"    Consider increasing PCA_N_COMPONENTS or reviewing input columns.")

    return pca_df, pca, X_scaled




def run_kmeans(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    K-Means clustering on supply-barrier features.

    Groups the 17 NCR cities into 3 Healthcare Paradox Zones:
      Cluster 0 → Low Paradox    (good supply, low poverty, near public L3)
      Cluster 1 → Medium Paradox (mixed access)
      Cluster 2 → High Paradox   (poor supply OR far from public L3 OR high poverty)

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
      paradox_cluster_label (str "Low/Medium/High Paradox")
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
            city_norm              TEXT PRIMARY KEY,
            pca_volume_index       REAL,
            pca_primary_care_index REAL
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
            pca_volume_index           REAL,
            pca_primary_care_index     REAL,
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
            f.pca_volume_index,
            f.pca_primary_care_index,
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
      12. fact_vulnerability has all 2 PCA columns populated (no NULLs)
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
          "WHERE pca_volume_index IS NULL OR pca_primary_care_index IS NULL",
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
    assert pca_df.shape == (17, 3), f"PCA output shape wrong: expected (17,3) [city_norm + 2 components], got {pca_df.shape}"
    assert not pca_df[PCA_COMPONENT_LABELS].isnull().any().any(), (
        f"PCA output contains NaN in columns {PCA_COMPONENT_LABELS}")
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