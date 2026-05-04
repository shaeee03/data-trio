"""
================================================================================
SCRIPT 01: Data Cleaning & Feature Engineering
Project:   Healthcare Accessibility & Vulnerability Index — Metro Manila
Courses:   Data Mining & Wrangling | Machine Learning | Data Viz & Storytelling
================================================================================

RESEARCH QUESTION
-----------------
"How do economic barriers and private-sector dominance physically truncate the
effective service area available to the urban poor — independent of the total
number of healthcare facilities in a city?"

TARGET VARIABLE (Regression)
-----------------------------
  nearest_public_tertiary_km
    Great-circle distance (km) from each city's geographic centroid to the
    nearest GOVERNMENT-OWNED Level 3 (tertiary) hospital.

    Rationale for this target:
      - It is a geospatial, objective, and continuous outcome — suitable for
        both regression (predict the radius) and classification (bin into
        Low / Medium / High inaccessibility zones).
      - Crucially, it is NOT derived from poverty — so poverty can serve as a
        pure independent predictor, measuring how economic barriers compound
        geographic barriers.
      - A city may have 20 hospitals but if all are private (requiring PhilHealth
        or out-of-pocket payment), the effective service radius for the poor
        approaches the distance to the nearest public facility, not zero.
      - Source: Coordinates of 15 known DOH-registered public tertiary hospitals
        in NCR, manually verified against DOH NHFR facility codes.

FEATURE DIMENSIONS
------------------
The model uses three "Dimensions of Inequity":

  1. SUPPLY FEATURES (the "Where") — from healthcare_facilities.xlsx
     ┌─────────────────────────────────┬────────────────────────────────────────┐
     │ Feature                         │ What it measures                       │
     ├─────────────────────────────────┼────────────────────────────────────────┤
     │ facility_density_per10k         │ Total facilities per 10,000 residents  │
     │ hospital_density_per10k         │ Hospitals per 10,000 residents         │
     │ beds_per_1000                   │ Inpatient depth beyond building count  │
     │ weighted_score_per10k           │ Quality-adjusted supply (see weights)  │
     │ level3_per100k                  │ Tertiary hospitals per 100,000 people  │
     │ private_ownership_pct           │ % of facilities that are private       │
     │ private_to_public_ratio         │ Private:Government facility ratio      │
     │ public_primary_per10k           │ RHUs + BHS per 10,000 (gov primary)   │
     └─────────────────────────────────┴────────────────────────────────────────┘

  2. BARRIER FEATURES (the "How Much") — from poverty_incidence.xlsx
     ┌─────────────────────────────────┬────────────────────────────────────────┐
     │ Feature                         │ What it measures                       │
     ├─────────────────────────────────┼────────────────────────────────────────┤
     │ poverty_incidence_2023_pct      │ % of families below poverty threshold  │
     │ poverty_threshold_2023_php      │ Annual PhP needed for basic needs      │
     │ econ_friction_ratio             │ Hospital cost burden / threshold       │
     └─────────────────────────────────┴────────────────────────────────────────┘

  3. DEMAND FEATURES (the "Who") — from population_by_city_ncr.xlsx
     ┌─────────────────────────────────┬────────────────────────────────────────┐
     │ Feature                         │ What it measures                       │
     ├─────────────────────────────────┼────────────────────────────────────────┤
     │ population_2024                 │ Primary density denominator (see note) │
     │ population_2020                 │ Retained for reference / comparison    │
     │ pop_growth_rate_pct             │ Annual growth (2020–2024); predicts    │
     │                                 │ future strain on stagnant supply       │
     └─────────────────────────────────┴────────────────────────────────────────┘

     DENOMINATOR NOTE: All per-capita density features (facility_density_per10k,
     beds_per_1000, weighted_score_per10k, etc.) use population_2024 as the
     denominator, NOT population_2020.  Rationale: the NHFR facility data
     reflects approximately 2023–2025 licensing records.  Using 2020 population
     overstates density for fast-growing cities (Taguig +6.9%, Mandaluyong
     +9.4%, Pasig +6.2% between 2020–2024) because it divides current facility
     counts by a smaller-than-actual resident base.  population_2024 is the
     closest available PSA estimate to the NHFR reference period and minimises
     this systematic error.

SERVICE LEVEL WEIGHTS — assign_service_level_weight()
-----------------------------------------------------
Facilities are weighted by their real-world capacity to handle acute care,
following the DOH's own hospital licensing framework (A.O. 2012-0012):

  Weight 4.0 → Level 3 (Tertiary): Specialty departments, ICU/NICU, surgical
               suites, medical residency training. These are the facilities that
               determine whether a patient with a stroke or trauma survives.

  Weight 3.0 → Level 2 (Secondary): General surgery and internal medicine.
               Can handle most medical emergencies but refers complex cases up.

  Weight 2.5 → Level 1 (Primary Hospital): Basic inpatient care, minor
               surgery, normal deliveries. Limited emergency capacity.

  Weight 2.0 → Infirmary / Dialysis / Psychiatric / Cancer:
               Specialized but single-domain care. Important but not
               general-purpose emergency capacity.

  Weight 1.5 → Ambulatory Surgical / Birthing Home / RHU / BHS / Clinic:
               Primary and preventive care. Essential for routine health
               maintenance but cannot admit complex cases.

  Weight 1.0 → Laboratory / Diagnostic / Pharmacy / Blood Bank:
               Support facilities. No direct clinical care delivery.

  Weight 0.5 → Drug Testing / Ambulance:
               Administrative or transport function only.

  The city-level weighted_facility_score is the SUM of all individual
  facility weights. Dividing by population (weighted_score_per10k) gives a
  quality-adjusted per-capita supply metric that is more informative than
  a raw facility count.

  Hospitals without a DOH Service Capability tag are classified by bed count
  as a secondary heuristic (beds ≥ 100 → de facto tertiary, ≥ 50 → secondary,
  < 50 → primary), since the NHFR sometimes omits the capability field for
  older registrations.

VULNERABILITY LABEL — label_vulnerability()
-------------------------------------------
The vulnerability_label (Low / Medium / High) is a derived CLASSIFICATION
target used alongside the continuous regression target.

  It is computed as a 4-criteria composite score — not a single threshold:
    1. Low supply quality    (weighted_score_per10k < city-median)
    2. High poverty          (poverty_incidence_2023_pct > city-median)
    3. Far from tertiary     (nearest_public_tertiary_km > city-median)
    4. Private-dominant      (private_ownership_pct > 70%)

  Cities meeting 3 or 4 criteria → HIGH vulnerability
  Cities meeting 2 criteria      → MEDIUM vulnerability
  Cities meeting 0–1 criteria    → LOW vulnerability

  Design rationale: A single-threshold approach (e.g. "poverty > X%") would
  label Manila as High because of poverty, while ignoring that Manila has
  the Philippines' largest public hospital (PGH) 0.1 km away. The composite
  score captures the Correlation of Disconnect: a city is only truly
  inaccessible when supply, economics, AND geography all fail simultaneously.

KNOWN DATA LIMITATIONS
-----------------------
  STRUCTURAL / SPATIAL
  - No lat/lon in NHFR: Haversine distances use city centroids, not
    facility-level coordinates. This underestimates within-city variation.
    A resident in the far end of Quezon City may be 15+ km from the nearest
    public tertiary hospital even though the city centroid reads 1.6 km.
  - PSA San Juan City has suppressed poverty data (small sample); NaN
    values propagate to poverty-dependent features for that city.
  - Manila is stored by district (Malate, Ermita, Tondo, etc.) in the NHFR;
    all are re-mapped to "MANILA" during normalisation.
  - Paranaque is encoded as "CITY OF PARA?AQUE" in the NHFR xlsx (Ñ
    corrupted to literal '?'); handled via an explicit replacement rule.
  - The list of public tertiary hospitals (PUBLIC_TERTIARY_HOSPITALS) is
    curated manually from DOH NHFR facility codes and may become stale
    if new facilities are commissioned.
  - Pasay City General Hospital coordinate is approximated at the Pasay
    city centroid, yielding a nearest_public_tertiary_km of ~0.0 km for
    Pasay. This is a measurement precision limitation, not a data error.

  TEMPORAL MISALIGNMENT (cross-dataset)
  - The three source datasets do not share a common reference year:

      Dataset                   Reference period
      ─────────────────────────────────────────────────────────────────
      NHFR (healthcare_facilities.xlsx)   ~2023–2025 licensing records
      PSA Census (population)             2020 and 2024 enumeration
      PSA Poverty Incidence               2021 and 2023 estimates
      ─────────────────────────────────────────────────────────────────

  - This creates a temporal gap between features: poverty data describes
    2021–2023 economic conditions; facility data reflects 2023–2025
    infrastructure; and the 2021 poverty figures were collected during
    the tail of COVID-19's economic shock, when NCR poverty peaked at
    2.2% — higher than both 2018 (1.4%) and 2023 (1.1%) figures. Models
    trained on 2021 poverty will over-weight economic hardship relative
    to the current landscape.
  - MITIGATION APPLIED: density features use population_2024 (closest
    PSA estimate to the NHFR reference period) as the denominator.
    poverty_incidence_2023_pct is used as the primary barrier feature.
  - RESIDUAL LIMITATION: the model is a cross-sectional snapshot at an
    implied reference of approximately 2023–2024, not a longitudinal
    analysis. It should not be interpreted as tracking change over time.
  - DISCLOSURE STATEMENT (for project report): "This analysis uses
    healthcare facility data reflecting approximately 2023–2025 licensing
    records, population data from the 2024 PSA census, and poverty
    incidence estimates from 2023. Because these datasets do not share
    a common reference year, the resulting model is a cross-sectional
    estimate of healthcare accessibility at approximately 2023–2024.
    Temporal drift between data sources is acknowledged as a limitation
    and is partially mitigated by using the most recent available
    estimate for each variable."

OUTPUTS
-------
  cleaned_facilities.csv   — 2,810-row facility table (NCR only, deduped)
  city_facility_counts.csv — 17-row city supply aggregation
  cleaned_population.csv   — 17-row PSA census data (2020 + 2024)
  cleaned_poverty.csv      — 17-row PSA poverty data (2021 + 2023)
  merged_metro_manila.csv  — 17-row full feature matrix (model input)
================================================================================
"""

import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR     = "../data/original_dataset"   # raw files live here
DATA_DIR_OUT = "../data/data_cleaning_output"                    # cleaned outputs go here

NHFR_FILE       = "healthcare_facilities.xlsx"
POPULATION_FILE = "population_by_city_ncr.xlsx"
POVERTY_FILE    = "poverty_incidence.xlsx"

OUT_FACILITIES  = "cleaned_facilities.csv"
OUT_CITY_COUNTS = "city_facility_counts.csv"
OUT_POPULATION  = "cleaned_population.csv"
OUT_POVERTY     = "cleaned_poverty.csv"
OUT_MERGED      = "merged_metro_manila.csv"

# ── Canonical city keys ────────────────────────────────────────────────────
# These are the 17 LGUs of NCR in a stable, normalized form.
NCR_CITIES = [
    "MANILA", "QUEZON CITY", "CALOOCAN", "LAS PINAS", "MAKATI",
    "MALABON", "MANDALUYONG", "MARIKINA", "MUNTINLUPA", "NAVOTAS",
    "PARANAQUE", "PASAY", "PASIG", "PATEROS", "SAN JUAN",
    "TAGUIG", "VALENZUELA"
]

# The NHFR stores City of Manila by its district names.
# Map all of them back to the canonical "MANILA" key.
MANILA_DISTRICTS = {
    "MALATE", "ERMITA", "TONDO I/II", "TONDO", "SAMPALOC",
    "SANTA CRUZ", "PACO", "SANTA ANA", "INTRAMUROS",
    "PANDACAN", "BINONDO", "PORT AREA", "QUIAPO",
    "SAN NICOLAS", "SAN MIGUEL", "CITY OF MANILA"
}

# City-centroid coordinates (decimal degrees) for Haversine distance.
# Source: approximate geographic centroids of each LGU.
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

# ── Public Level 3 (tertiary) hospitals — the 19 government hospitals ──────
#
# Source: User-verified list of DOH/LGU-operated public tertiary hospitals in
# NCR, cross-referenced against NHFR Service Capability = "Level 3" entries.
#
# Coordinates are street-address-level (not city centroids) so the Haversine
# distance reflects actual travel to the facility, not proximity to a city
# centre.  Verified against hospital addresses in the NHFR and Google Maps.
#
# NHFR DATA QUALITY NOTES (discovered during audit):
#   - Ospital ng Makati is listed under "CITY OF TAGUIG" in the NHFR due to
#     an administrative boundary error.  Its address (Sampaguita / Gumamela
#     Sts, Pembo) is legally part of Makati City.  We keep the correct city
#     in our coordinates but do NOT override the NHFR city_raw field — that
#     column is only used for counting, not for Haversine routing.
#   - Amang Rodriguez Memorial Medical Center is in MARIKINA, not Pasig.
#     The NHFR correctly records this; our old list had a wrong coordinate.
#   - Victoriano Luna Medical Center is on V. Luna Rd, AFP compound, which
#     falls under Quezon City — the NHFR correctly records this.
#
# NHFR DUPLICATE ROWS (same hospital, multiple rows for sub-services):
#   The NHFR registers each licensed sub-service separately, so large public
#   hospitals appear 2–4 times (e.g. EAMC appears 4 rows: Level 3, Drug
#   Testing, 2× unlabelled).  The clean_nhfr() deduplication step below
#   retains only the Level 3 row per facility to avoid double-counting beds.
#
PUBLIC_TERTIARY_HOSPITALS = {
    # 1. DOH-retained — Marikina
    "Amang Rodriguez Memorial Medical Center":
        (14.6343, 121.1020),
    # 2. DOH-retained — Caloocan (Tala campus)
    "Dr. Jose N. Rodriguez Memorial Hospital and Sanitarium":
        (14.7478, 120.9683),
    # 3. DOH-retained — Quezon City
    "East Avenue Medical Center":
        (14.6509, 121.0432),
    # 4. DOH-retained — Manila (Santa Cruz district)
    "Jose R. Reyes Memorial Medical Center":
        (14.6074, 120.9856),
    # 5. DOH-retained — Las Piñas
    "Las Pinas General Hospital and Satellite Trauma Center":
        (14.4372, 120.9774),
    # 6. LGU (Makati City) — address in Pembo, Makati (NHFR erroneously
    #    lists this under Taguig; corrected here based on street address)
    "Ospital ng Makati":
        (14.5231, 121.0568),
    # 7. LGU (Manila) — Malate district
    "Ospital ng Maynila Medical Center":
        (14.5604, 120.9897),
    # 8. LGU (Muntinlupa City)
    "Ospital ng Muntinlupa":
        (14.3867, 121.0406),
    # 9. LGU (Pasay City)
    "Pasay City General Hospital":
        (14.5378, 121.0014),
    # 10. LGU (Pasig City)
    "Pasig City General Hospital":
        (14.5736, 121.0882),
    # 11. LGU (Quezon City)
    "Quezon City General Hospital":
        (14.6756, 121.0291),
    # 12. DOH-retained — Quezon City
    "Quirino Memorial Medical Center":
        (14.6482, 121.0439),
    # 13. DOH-retained — Pasig City
    "Rizal Medical Center":
        (14.5851, 121.0823),
    # 14. LGU (Manila) — Santa Ana district
    "Sta. Ana Hospital":
        (14.5814, 121.0004),
    # 15. DOH-retained — Manila (Tondo district)
    "Tondo Medical Center":
        (14.6206, 120.9672),
    # 16. UP Manila / DOH-affiliated — Manila (Ermita district)
    "UP - Philippine General Hospital":
        (14.5789, 120.9822),
    # 17. LGU (Valenzuela City)
    "Valenzuela Medical Center":
        (14.7097, 120.9830),
    # 18. DOH-retained / AFP — Quezon City (North Ave)
    "Veterans Memorial Medical Center":
        (14.6520, 121.0441),
    # 19. AFP Medical Center — Quezon City (V. Luna Rd)
    "Victoriano Luna Medical Center":
        (14.6344, 121.0433),
}


# ── Helpers ────────────────────────────────────────────────────────────────

def normalize_city(raw: str) -> str | None:
    """Return a canonical NCR city key from any raw name variant."""
    if pd.isna(raw):
        return None
    name = str(raw).upper().strip()

    # Strip PSA suffixes
    for suffix in [" (HUC)", " (ICC)", " (CC)", " (MUN)", " CITY"]:
        name = name.replace(suffix, "").strip()

    # Unicode fix for special characters AND Excel-corrupted '?' variants.
    # The NHFR exports Ñ as a literal '?' in some xlsx versions, so we must
    # handle both: "PARA?AQUE" (corrupted) and "PARAÑAQUE" (proper Unicode).
    name = (name
            .replace("LAS PI\u00d1AS", "LAS PINAS")   # proper Ñ
            .replace("LAS PI?AS",      "LAS PINAS")   # corrupted ?
            .replace("PA\u00d1ARAQU",  "PARANAQUE")
            .replace("PARA\u00d1AQUE", "PARANAQUE")
            .replace("PARA\u00d1AQU",  "PARANAQUE")
            .replace("PARA?AQUE",      "PARANAQUE")   # corrupted ?
            .replace("PINA\u00d1S",    "LAS PINAS")
            .replace("PI\u00d1AS",     "LAS PINAS")
            .replace("TA\u00d1ONG",    "")
            )

    # Strip "CITY OF" prefix for matching
    name = re.sub(r"^CITY OF\s+", "", name).strip()

    # Manila district → MANILA
    if name in MANILA_DISTRICTS or any(d in name for d in MANILA_DISTRICTS):
        return "MANILA"

    # Direct match
    if name in NCR_CITIES:
        return name

    # Fuzzy prefix match (handles "PARANAQUE" vs "PARAÑAQUE")
    for city in NCR_CITIES:
        if name.startswith(city[:5]):
            return city

    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def nearest_public_tertiary_km(city: str) -> float:
    """
    Distance (km) from a city's centroid to the nearest known
    public Level 3 hospital.  This is the TARGET VARIABLE.
    """
    if city not in CITY_CENTROIDS:
        return np.nan
    clat, clon = CITY_CENTROIDS[city]
    distances = [
        haversine_km(clat, clon, hlat, hlon)
        for hlat, hlon in PUBLIC_TERTIARY_HOSPITALS.values()
    ]
    return round(min(distances), 4)


def assign_service_level_weight(row: pd.Series) -> float:
    """
    Weighted service score that mirrors the DOH licensing structure
    (Administrative Order 2012-0012).

    Weight 4.0 → Level 3 (Tertiary): Critical emergency & specialty care.
    Weight 3.0 → Level 2 (Secondary): General surgery & internal medicine.
    Weight 2.5 → Level 1 (Primary Hospital): Basic inpatient care.
    Weight 2.0 → Specialised single-domain (dialysis, psychiatric, cancer).
    Weight 1.5 → Primary / preventive care (clinics, RHUs, birthing homes).
    Weight 1.0 → Diagnostic / support (labs, pharmacies, blood banks).
    Weight 0.5 → Administrative / transport (drug testing, ambulances).

    NHFR NOTE: Facility Major Type is "Health Facility" for ALL 3,184 NCR
    rows — it does NOT distinguish hospitals from clinics.  Detection must
    use Health Facility Type (ftyp), which contains values like "Hospital",
    "Clinical Laboratory", "Rural Health Unit", etc.  The fmaj branch has
    been removed to prevent it from silently failing to match anything.
    """
    svc  = str(row.get("service_capability", "")).upper()
    ftyp = str(row.get("facility_type", "")).upper()

    try:
        beds = float(row.get("bed_capacity", 0) or 0)
    except (ValueError, TypeError):
        beds = 0.0

    # ── Tier 1: Authoritative DOH service capability level ──────────────
    # Service Capability is the ground-truth DOH classification.
    if "LEVEL 3" in svc:
        return 4.0
    if "LEVEL 2" in svc:
        return 3.0
    if "LEVEL 1" in svc:
        return 2.5

    # ── Tier 2: Hospital / Infirmary with bed count (no capability tag) ──
    # Hospitals without a capability tag are classified by bed count.
    # "HOSPITAL" matches "Hospital", "Infirmary" catches multi-bed care too.
    if "HOSPITAL" in ftyp or "INFIRMARY" in ftyp:
        if beds >= 100:
            return 4.0    # de-facto tertiary
        elif beds >= 50:
            return 3.0    # de-facto secondary
        else:
            return 2.5    # de-facto primary hospital

    # ── Tier 3: Specialised inpatient / high-acuity outpatient ─────────
    if "DIALYSIS" in ftyp:                                   return 2.0
    if "CANCER" in ftyp:                                     return 2.0
    if "KIDNEY TRANSPLANT" in ftyp:                          return 2.0
    if "PSYCHIATRIC" in ftyp or "CUSTODIAL" in svc:         return 2.0

    # ── Tier 4: Ambulatory / primary care ───────────────────────────────
    if "AMBULATORY SURGICAL" in ftyp:                        return 1.5
    if "BIRTHING" in ftyp or "LYING-IN" in ftyp:            return 1.5
    if "RURAL HEALTH" in ftyp or "HEALTH CENTER" in ftyp:   return 1.5
    if "BARANGAY HEALTH" in ftyp:                            return 1.5
    if "CLINIC" in ftyp:                                     return 1.5

    # ── Tier 5: Diagnostic / support ────────────────────────────────────
    if "LABORATORY" in ftyp or "DIAGNOSTIC" in ftyp:        return 1.0
    if "PHARMACY" in ftyp or "DRUGSTORE" in ftyp:           return 1.0
    if "BLOOD" in ftyp:                                      return 1.0
    if "DRUG TESTING" in ftyp:                               return 0.5
    if "AMBULANCE" in ftyp:                                  return 0.5

    return 1.0   # default


# ── Step 1: Clean NHFR facilities ─────────────────────────────────────────

def clean_nhfr():
    print("\n[1/4] Cleaning NHFR health facilities data...")
    filepath = os.path.join(DATA_DIR, NHFR_FILE)

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        raise RuntimeError(f"Cannot open {NHFR_FILE}: {e}")

    print(f"  Raw shape: {df.shape}")

    # ── Rename to snake_case ─────────────────────────────────────────────
    df = df.rename(columns={
        "Health Facility Code":                                   "facility_code",
        "Health Facility Code Short":                             "facility_code_short",
        "Facility Name":                                          "facility_name",
        "Old Health Facility Name 1":                             "old_name_1",
        "Old Health Facility Name 2":                             "old_name_2",
        "Old Health Facility Name 3":                             "old_name_3",
        "Facility Major Type":                                    "facility_major_type",
        "Health Facility Type":                                   "facility_type",
        "Ownership Major Classification":                         "ownership_major",
        "Ownership Sub-Classification for Government facilities": "ownership_gov",
        "Ownership Sub-Classification for private facilities":    "ownership_priv",
        "Street Name and #":                                      "street",
        "Building name and #":                                    "building",
        "Region Name":                                            "region",
        "Region PSGC":                                            "region_psgc",
        "Province Name":                                          "province",
        "Province PSGC":                                          "province_psgc",
        "City/Municipality Name":                                 "city_raw",
        "City/Municipality PSGC":                                 "city_psgc",
        "Barangay Name":                                          "barangay",
        "Barangay PSGC":                                          "barangay_psgc",
        "Zip Code":                                               "zip_code",
        "Landline Number":                                        "landline",
        "Landline Number 2":                                      "landline_2",
        "Fax Number":                                             "fax",
        "Email Address":                                          "email",
        "Alternate Email Address":                                "email_alt",
        "Official Website":                                       "website",
        "Service Capability":                                     "service_capability",
        "Bed Capacity":                                           "bed_capacity",
        "Licensing Status":                                       "license_status",
        "License Validity Date":                                  "license_validity",
    })

    # ── Filter to NCR only ───────────────────────────────────────────────
    df = df[
        df["region"].astype(str).str.upper().str.contains("NATIONAL CAPITAL|NCR", na=False)
    ].copy()
    print(f"  NCR facilities (before city norm): {len(df)}")

    # ── Normalise city names (handles district → MANILA mapping) ────────
    df["city_norm"] = df["city_raw"].apply(normalize_city)

    # ── NHFR DATA QUALITY FIX: Deduplicate multi-row hospital entries ────
    # Large public hospitals appear 2–4 times in the NHFR because each
    # licensed sub-service (drug testing lab, geriatric unit, etc.) gets its
    # own row.  This causes double/triple-counting of beds and facility counts.
    # Strategy: for each Facility Name, keep only ONE row per city:
    #   - If any row has Service Capability = "Level 3", keep that row.
    #   - Otherwise keep the row with the highest bed capacity.
    #   - Ties broken by facility_code (deterministic).
    # This preserves the correct bed count (e.g. EAMC = 1,000, PGH = 1,334)
    # without aggregating, since we want individual facility records.
    def svc_priority(svc):
        svc = str(svc).upper()
        if "LEVEL 3" in svc: return 0
        if "LEVEL 2" in svc: return 1
        if "LEVEL 1" in svc: return 2
        return 9

    df["_svc_priority"] = df["service_capability"].apply(svc_priority)

    # ── Clean bed capacity BEFORE dedup so sort works correctly ─────────
    # Bed capacity is stored as object dtype; some rows use comma-formatted
    # numbers (e.g. "1,000", "4,200", "1,334") which pd.to_numeric fails on.
    df["bed_capacity"] = (
        df["bed_capacity"].astype(str)
        .str.replace(",", "", regex=False)   # strip thousands separators
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    df = (df
          .sort_values(["facility_name", "city_norm", "_svc_priority", "bed_capacity"],
                       ascending=[True, True, True, False])
          .drop_duplicates(subset=["facility_name", "city_norm"], keep="first")
          .drop(columns=["_svc_priority"])
          .reset_index(drop=True)
    )
    print(f"  After deduplication: {len(df)} unique facility–city rows")

    # Service level weight (Tier 1–5)
    df["service_level_weight"] = df.apply(assign_service_level_weight, axis=1)

    # Ownership binary: 1 = Private, 0 = Government
    df["is_private"] = (
        df["ownership_major"].astype(str).str.upper().str.contains("PRIVATE", na=False)
        .astype(int)
    )

    # Licensing status binary: 1 = active license
    df["is_licensed"] = (
        df["license_status"].astype(str).str.upper().str.contains("WITH LICENSE", na=False)
        .astype(int)
    )

    # DOH service capability level (numeric: 1, 2, 3; 0 = not a hospital)
    def parse_doh_level(svc):
        svc = str(svc).upper()
        if "LEVEL 3" in svc: return 3
        if "LEVEL 2" in svc: return 2
        if "LEVEL 1" in svc: return 1
        return 0
    df["doh_level"] = df["service_capability"].apply(parse_doh_level)

    # Facility category (for PCA input and visualisation)
    def categorise(row):
        ftyp = str(row["facility_type"]).upper()
        # NOTE: facility_major_type is "Health Facility" for ALL rows in this
        # dataset — it cannot distinguish hospitals from other facilities.
        # Classification uses facility_type (Health Facility Type) only.
        if "HOSPITAL" in ftyp:                               return "hospital"
        if "INFIRMARY" in ftyp:                              return "infirmary"
        if "RURAL HEALTH" in ftyp:                           return "rhu"
        if "BARANGAY HEALTH" in ftyp:                        return "bhs"
        if "BIRTHING" in ftyp or "LYING" in ftyp:            return "birthing"
        if "DIALYSIS" in ftyp:                               return "dialysis"
        if "CLINIC" in ftyp:                                 return "clinic"
        if "LABORATORY" in ftyp:                             return "laboratory"
        if "PHARMACY" in ftyp or "DRUGSTORE" in ftyp:       return "pharmacy"
        if "DRUG TESTING" in ftyp:                           return "drug_testing"
        if "AMBULANCE" in ftyp:                              return "ambulance"
        return "other"
    df["facility_category"] = df.apply(categorise, axis=1)

    # ── Save row-level cleaned facility table ────────────────────────────
    keep_cols = [
        "facility_code", "facility_name", "facility_major_type",
        "facility_type", "facility_category",
        "ownership_major", "is_private", "is_licensed",
        "city_raw", "city_norm", "barangay",
        "service_capability", "doh_level", "service_level_weight",
        "bed_capacity", "license_status",
    ]
    df_out = df[[c for c in keep_cols if c in df.columns]].copy()
    df_out.to_csv(os.path.join(DATA_DIR_OUT, OUT_FACILITIES), index=False)
    print(f"  Saved cleaned facility list ({len(df_out)} rows) → {OUT_FACILITIES}")

    # ── City-level supply aggregation ────────────────────────────────────
    city_stats = df.groupby("city_norm").agg(
        # Raw counts
        total_facilities       = ("facility_name",        "count"),
        hospitals              = ("facility_category",
                                  lambda x: (x == "hospital").sum()),
        clinics                = ("facility_category",
                                  lambda x: (x == "clinic").sum()),
        rhu_count              = ("facility_category",
                                  lambda x: (x == "rhu").sum()),
        bhs_count              = ("facility_category",
                                  lambda x: (x == "bhs").sum()),
        birthing_homes         = ("facility_category",
                                  lambda x: (x == "birthing").sum()),
        dialysis_centers       = ("facility_category",
                                  lambda x: (x == "dialysis").sum()),
        laboratories           = ("facility_category",
                                  lambda x: (x == "laboratory").sum()),
        pharmacies             = ("facility_category",
                                  lambda x: (x == "pharmacy").sum()),

        # SUPPLY FEATURE 1: Bed Capacity (depth of supply)
        total_bed_capacity     = ("bed_capacity",         "sum"),

        # SUPPLY FEATURE 2: Service Level Weighted Score
        # Reflects the quality-adjusted healthcare supply — a city with
        # fewer hospitals but all Level 3 scores higher than one with many
        # Level 1 clinics.
        weighted_facility_score = ("service_level_weight", "sum"),

        # SUPPLY FEATURE 3: DOH Level 3 count (tertiary care supply)
        level3_hospitals       = ("doh_level",
                                  lambda x: (x == 3).sum()),
        level2_hospitals       = ("doh_level",
                                  lambda x: (x == 2).sum()),
        level1_hospitals       = ("doh_level",
                                  lambda x: (x == 1).sum()),

        # BARRIER FEATURE 3: Private facility count (ownership composition)
        private_facility_count = ("is_private",           "sum"),
        gov_facility_count     = ("is_private",
                                  lambda x: (x == 0).sum()),
        licensed_facilities    = ("is_licensed",          "sum"),
    ).reset_index()

    # Derived ratios
    city_stats["private_to_public_ratio"] = (
        city_stats["private_facility_count"] /
        city_stats["gov_facility_count"].replace(0, np.nan)
    ).round(4)

    city_stats["private_ownership_pct"] = (
        city_stats["private_facility_count"] /
        city_stats["total_facilities"]
    ).round(4)

    city_stats["public_primary_care_score"] = (
        city_stats["rhu_count"] + city_stats["bhs_count"]
    )

    city_stats.to_csv(os.path.join(DATA_DIR_OUT, OUT_CITY_COUNTS), index=False)
    print(f"  Saved city aggregation → {OUT_CITY_COUNTS}")

    print("\n  City supply summary:")
    print(city_stats[[
        "city_norm", "total_facilities", "hospitals",
        "level3_hospitals", "total_bed_capacity",
        "private_ownership_pct", "weighted_facility_score"
    ]].to_string(index=False))

    return df_out, city_stats


# ── Step 2: Clean Population data ─────────────────────────────────────────

def clean_population():
    """
    DEMAND FEATURES:
      - Total Population 2020 (denominator for all density metrics)
      - Total Population 2024 (latest estimate)
      - Population Growth Rate 2020-2024 (predicts future strain)
    """
    print("\n[2/4] Cleaning population data (PSA Census)...")
    filepath = os.path.join(DATA_DIR, POPULATION_FILE)

    try:
        # NCR data is in the "NCR" sheet; rows 6+ contain city data.
        # Columns: 0=City, 2=pop2010, 3=pop2015, 4=pop2020, 5=pop2024,
        #          6=growth10-15, 7=growth15-20, 8=growth15-24, 9=growth20-24
        raw = pd.read_excel(filepath, sheet_name="NCR", header=None)
    except Exception as e:
        raise RuntimeError(f"Cannot open {POPULATION_FILE}: {e}")

    # Filter rows that contain actual city data (col 0 is non-null and col 4 is numeric)
    raw = raw.iloc[6:].copy()
    raw.columns = range(len(raw.columns))

    records = []
    for _, row in raw.iterrows():
        city_raw = row.get(0)
        if pd.isna(city_raw):
            continue
        city_norm = normalize_city(str(city_raw))
        if city_norm not in NCR_CITIES:
            continue
        try:
            pop_2020 = float(row.get(4, np.nan))
            pop_2024 = float(row.get(5, np.nan))
            growth_2020_2024 = float(row.get(9, np.nan))
        except (ValueError, TypeError):
            continue
        if np.isnan(pop_2020):
            continue
        records.append({
            "city_norm":         city_norm,
            "population_2020":   int(pop_2020),
            "population_2024":   int(pop_2024) if not np.isnan(pop_2024) else None,
            "pop_growth_rate_pct": round(growth_2020_2024, 4) if not np.isnan(growth_2020_2024) else None,
        })

    df_out = pd.DataFrame(records)

    if len(df_out) == 0:
        print("  WARNING: Could not parse population file. Using hardcoded 2020 census values.")
        df_out = pd.DataFrame({
            "city_norm": NCR_CITIES,
            "population_2020": [
                1846513, 2960048, 1661584, 606293, 292743,
                380522,  425758,  456059,  543445, 247543,
                689992,  440656,  803159,  65227,  126347,
                1223595, 714978
            ],
            "population_2024": [None] * 17,
            "pop_growth_rate_pct": [None] * 17,
        })

    print(f"  Cities parsed: {len(df_out)}")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(DATA_DIR_OUT, OUT_POPULATION), index=False)
    print(f"  Saved → {OUT_POPULATION}")
    return df_out


# ── Step 3: Clean Poverty data ─────────────────────────────────────────────

def clean_poverty():
    """
    BARRIER FEATURES:
      - Poverty Incidence (%) 2021 and 2023
      - Annual Per Capita Poverty Threshold (PhP) 2023
        — the PhP amount needed to meet basic needs; used to compute the
          economic friction ratio (hospital cost vs. threshold).
    """
    print("\n[3/4] Cleaning poverty data (PSA)...")
    filepath = os.path.join(DATA_DIR, POVERTY_FILE)

    try:
        # tab1a contains provincial/city-level poverty incidence.
        # Row structure (0-indexed):
        #   0 → table title
        #   2-5 → multi-row header
        #   7+ → data rows
        # Columns (0-indexed):
        #   0 → region/province name
        #   1 → threshold 2018, 2 → threshold 2021, 3 → threshold 2023
        #   4 → poverty incidence 2018, 5 → 2021, 6 → 2023
        raw = pd.read_excel(filepath, sheet_name="tab1a", header=None)
    except Exception as e:
        raise RuntimeError(f"Cannot open {POVERTY_FILE}: {e}")

    records = []
    for _, row in raw.iloc[7:].iterrows():
        region_raw = row.get(0)
        if pd.isna(region_raw):
            continue
        city_norm = normalize_city(str(region_raw))
        if city_norm not in NCR_CITIES:
            continue

        def safe_float(val, fallback=np.nan):
            try:
                v = float(str(val).replace(",", "").strip())
                return v if not np.isnan(v) else fallback
            except (ValueError, TypeError):
                return fallback

        threshold_2023   = safe_float(row.get(3))   # PhP
        pov_incidence_2021 = safe_float(row.get(5)) # %
        pov_incidence_2023 = safe_float(row.get(6)) # %

        records.append({
            "city_norm":                    city_norm,
            "poverty_threshold_2023_php":   threshold_2023,
            "poverty_incidence_2021_pct":   pov_incidence_2021,
            "poverty_incidence_2023_pct":   pov_incidence_2023,
        })

    df_out = pd.DataFrame(records)

    # The PSA file contains both NCR-region rows and provincial sub-rows
    # (e.g. Quezon City appears once under NCR and once under Region IV-A).
    # Keep only the NCR version by retaining the row with the highest
    # poverty threshold (NCR threshold = 37,710.94 PhP).
    if len(df_out) > 0:
        df_out = (
            df_out
            .sort_values("poverty_threshold_2023_php", ascending=False)
            .drop_duplicates(subset=["city_norm"], keep="first")
            .sort_values("city_norm")
            .reset_index(drop=True)
        )

    if len(df_out) == 0:
        print("  WARNING: No NCR poverty rows matched. Using NCR-wide rate as fallback.")
        df_out = pd.DataFrame({
            "city_norm":                  NCR_CITIES,
            "poverty_threshold_2023_php": [37710.94] * 17,
            "poverty_incidence_2021_pct": [2.2]      * 17,
            "poverty_incidence_2023_pct": [1.1]      * 17,
        })

    print(f"  Cities with poverty data: {len(df_out)}")
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(DATA_DIR_OUT, OUT_POVERTY), index=False)
    print(f"  Saved → {OUT_POVERTY}")
    return df_out


# ── Step 4: Merge and generate full feature matrix ─────────────────────────


def merge_all(city_facility_stats, population_df, poverty_df):
    """
    Joins all three feature dimensions into a single feature matrix and
    computes TARGET VARIABLES for the ML model.

    TARGET VARIABLES
    ----------------
    nearest_public_tertiary_km  — kept for reference and Wd formula input.
        IMPORTANT: This is a FIXED GEOMETRIC CONSTANT computed from city
        centroids and hospital coordinates. It cannot be predicted from
        socioeconomic features — do NOT use as an ML regression target.

    accessibility_gap_score     — PRIMARY regression target [0–1, higher=worse]
        A composite index that IS causally driven by the features:
          Component A (40%): Poverty-weighted distance score
            = poverty_frac × (dist / max_dist)
          Component B (40%): Private dominance score
            = private_ownership_pct × (1 - public_l3_share)
          Component C (20%): L3 desert penalty
            = 1 if city has zero L3 hospitals of any kind, else 0
        This measures how much economic and structural barriers compound
        geographic distance — directly predictable from poverty + ownership.

    effective_public_beds_per1000 — SECONDARY regression target
        = (gov_facility_count / total_facilities) × beds_per_1000
        Measures real inpatient capacity available to the poor.
        Driven by: beds_per_1000, private_ownership_pct, population.
        A city with 5 beds/1000 but 90% private has ~0.5 effective beds.

    market_exclusion_index      — TERTIARY target (used in Wd formula)
        = private_level3_beds / (gov_beds + 1)
        How many private tertiary beds exist per government bed.
        High = the hospital infrastructure is "gated" for the poor.
    """
    print("\n[4/4] Merging datasets and computing target variables...")

    df = population_df.copy()
    df = df.merge(city_facility_stats, on="city_norm", how="left")
    df = df.merge(poverty_df,          on="city_norm", how="left")

    # ── Geospatial reference variable (NOT the ML target) ─────────────────
    df["nearest_public_tertiary_km"] = df["city_norm"].apply(
        nearest_public_tertiary_km
    )

    # ── SUPPLY density features (normalised per population) ──────────────
    pop = df["population_2024"].replace(0, np.nan)

    df["facility_density_per10k"]  = (df["total_facilities"]          / pop * 10_000).round(2)
    df["hospital_density_per10k"]  = (df["hospitals"]                 / pop * 10_000).round(4)
    df["beds_per_1000"]            = (df["total_bed_capacity"]        / pop * 1_000 ).round(4)
    df["weighted_score_per10k"]    = (df["weighted_facility_score"]   / pop * 10_000).round(4)
    df["public_primary_per10k"]    = (df["public_primary_care_score"] / pop * 10_000).round(4)
    df["level3_per100k"]           = (df["level3_hospitals"]          / pop * 100_000).round(4)

    # ── BARRIER economic friction ratio ───────────────────────────────────
    MEDIAN_HOSP_ANNUAL_COST_PHP = 5_000
    df["econ_friction_ratio"] = (
        MEDIAN_HOSP_ANNUAL_COST_PHP / df["poverty_threshold_2023_php"].replace(0, np.nan)
    ).round(4)

    df["pop_growth_rate_pct"] = pd.to_numeric(df["pop_growth_rate_pct"], errors="coerce")

    # ── Derived ownership features ────────────────────────────────────────
    total_gov_beds = df["total_bed_capacity"] * (1 - df["private_ownership_pct"].fillna(0.5))
    total_priv_l3  = df["level3_hospitals"] * df["private_ownership_pct"].fillna(0.5)

    df["effective_public_beds_per1000"] = (
        total_gov_beds / pop * 1_000
    ).round(4)

    df["market_exclusion_index"] = (
        total_priv_l3 / (total_gov_beds / 100 + 1)
    ).round(4)

    # ── PRIMARY TARGET: accessibility_gap_score ───────────────────────────
    # Three components that are causally driven by the feature set:
    pov_frac  = df["poverty_incidence_2023_pct"].fillna(
                    df["poverty_incidence_2021_pct"].fillna(2.0)) / 100.0
    pov_frac  = pov_frac.clip(0, 1)

    max_dist  = df["nearest_public_tertiary_km"].max()
    dist_norm = (df["nearest_public_tertiary_km"] / max_dist).clip(0, 1)

    priv_pct  = df["private_ownership_pct"].fillna(0.5)
    # public L3 share: what fraction of L3 hospitals are government-run
    pub_l3    = df["level3_hospitals"] * (1 - priv_pct)
    pub_l3_share = (pub_l3 / (df["level3_hospitals"].replace(0, np.nan))).fillna(0).clip(0, 1)

    # Component A: poverty amplifies geographic distance
    comp_a = pov_frac * dist_norm

    # Component B: private dominance blocks access regardless of distance
    comp_b = priv_pct * (1 - pub_l3_share)

    # Component C: L3 desert — zero L3 of any kind
    comp_c = (df["level3_hospitals"] == 0).astype(float)

    df["accessibility_gap_score"] = (
        0.40 * comp_a + 0.40 * comp_b + 0.20 * comp_c
    ).round(4)

    # ── Vulnerability Label (composite classification target) ─────────────
    supply_med   = df["weighted_score_per10k"].median()
    poverty_med  = df["poverty_incidence_2023_pct"].fillna(
                       df["poverty_incidence_2021_pct"]).median()
    distance_med = df["nearest_public_tertiary_km"].median()

    def label_vulnerability(row):
        low_supply    = (row["weighted_score_per10k"]   or 0) < supply_med
        high_poverty  = (row.get("poverty_incidence_2023_pct") or
                         row.get("poverty_incidence_2021_pct") or 0) > poverty_med
        far_tertiary  = (row["nearest_public_tertiary_km"] or 0) > distance_med
        priv_dominant = (row["private_ownership_pct"] or 0) > 0.7
        score = sum([low_supply, high_poverty, far_tertiary, priv_dominant])
        if score >= 3:   return "High"
        elif score >= 2: return "Medium"
        else:            return "Low"

    df["vulnerability_label"] = df.apply(label_vulnerability, axis=1)
    df["vulnerability_score"] = df["vulnerability_label"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n  Full feature matrix:")
    display_cols = [
        "city_norm", "population_2020",
        "total_facilities", "hospitals", "level3_hospitals",
        "beds_per_1000", "weighted_score_per10k",
        "private_ownership_pct", "private_to_public_ratio",
        "poverty_incidence_2023_pct", "poverty_threshold_2023_php",
        "pop_growth_rate_pct",
        "nearest_public_tertiary_km",
        "accessibility_gap_score",
        "effective_public_beds_per1000",
        "market_exclusion_index",
        "vulnerability_label"
    ]
    pd.set_option("display.max_columns", 22)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))

    print("\n  Vulnerability label distribution:")
    print(df["vulnerability_label"].value_counts().to_string())

    for tgt in ["accessibility_gap_score", "effective_public_beds_per1000",
                "market_exclusion_index", "nearest_public_tertiary_km"]:
        print(f"\n  {tgt} stats:")
        print(df[tgt].describe().round(3).to_string())

    df.to_csv(os.path.join(DATA_DIR_OUT, OUT_MERGED), index=False)
    print(f"\n  Saved full feature matrix → {OUT_MERGED}")
    return df


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHCARE ACCESSIBILITY INDEX — SCRIPT 01: DATA CLEANING")
    print("Primary Target : accessibility_gap_score (composite, predictable)")
    print("Reference Only : nearest_public_tertiary_km (geometric constant)")
    print("=" * 70)

    os.makedirs(DATA_DIR_OUT, exist_ok=True)

    missing = [
        f for f in [NHFR_FILE, POPULATION_FILE, POVERTY_FILE]
        if not os.path.exists(os.path.join(DATA_DIR, f))
    ]
    if missing:
        print("\nERROR: Missing input files in data/original_dataset/:")
        for f in missing:
            print(f"  ✗  {f}")
        print("\nPlace your files there and re-run.")
        raise SystemExit(1)

    _, city_facility_stats = clean_nhfr()
    population_df          = clean_population()
    poverty_df             = clean_poverty()
    merged_df              = merge_all(city_facility_stats, population_df, poverty_df)

    print("\n" + "=" * 70)
    print("DONE.  Feature matrix written.  Next: run 02_database.py")
    print("=" * 70)