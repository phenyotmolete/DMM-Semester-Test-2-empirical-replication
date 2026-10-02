"""
Build a firm-quarter financial panel from SEC EDGAR for the firms listed
in VIOLATIONS_CSV (set below), and merge in the Bind indicator.

Currently configured for: fulltext_violations_events_FINAL.csv
(the full-text-search-derived sample, 2010-2019, 54 companies).

If you want to instead rebuild the ORIGINAL Roberts/Sufi-based sample
(2009-2011, 259 companies), change VIOLATIONS_CSV below to
"violations_2009_2011_for_edgar.csv" and remember to also change
CUTOFF_DATE back to "2011-12-31" in run_full_pipeline.py.

Run locally:
    pip install requests pandas
    python build_edgar_panel.py

REQUIRED: replace USER_AGENT below with your real name + email.
SEC blocks requests without a descriptive User-Agent.

Output: edgar_firm_quarter_panel.csv
    Columns: cik, gvkey, fy, fp, end_date, assets, current_assets,
             current_liabilities, ppe_net, capex_ytd, net_income,
             depreciation, bind
"""

import requests
import time
import pandas as pd
import numpy as np  # noqa: F401 (used for NaN fallback in build_firm_panel)

USER_AGENT = "YourName YourEmail@university.edu"  # <-- REPLACE ME
HEADERS = {"User-Agent": USER_AGENT}

VIOLATIONS_CSV = "fulltext_violations_events_FINAL.csv"  # full-text-derived events, 2010-2019, 54 companies (rename fulltext_violations_events_FINAL2.csv to this exact filename)
OUTPUT_CSV = "edgar_firm_quarter_panel.csv"

# Financial-firm CIKs to exclude, matching Chava & Roberts' SIC 6000-6999
# exclusion (bank holding companies work fundamentally differently).
EXCLUDED_CIKS = {
    712534,  # First Merchants Corp (bank holding company)
}

# Candidate XBRL tags per concept -- firms use different ones, so we try
# each in order and take the first that exists.
TAG_CANDIDATES = {
    # "LiabilitiesAndStockholdersEquity" is a fallback via the accounting
    # identity (Assets = Liabilities + Equity, always true by definition).
    # Some filers, especially early XBRL adopters (pre-2011 for smaller
    # reporting companies), tag this total more consistently than "Assets"
    # itself, particularly for prior-period comparative columns.
    "assets": ["Assets", "LiabilitiesAndStockholdersEquity"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "ppe_gross": ["PropertyPlantAndEquipmentGross"],
    "accum_depreciation": ["AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
    "capex_ytd": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireEquipmentOnLease",
        "PaymentsForProceedsFromProductiveAssets",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireMachineryAndEquipment",
        "PaymentsToAcquireBuildings",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ],
    "total_debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "total_debt_st": ["ShortTermBorrowings", "DebtCurrent"],
}


def fetch_company_facts(cik: str):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    return resp.json()


def extract_concept_series(facts: dict, tag_list: list) -> pd.DataFrame:
    """Return a tidy dataframe of (end_date, fy, fp, val) for the first
    tag in tag_list that the firm actually reports, using 10-Q/10-K only,
    USD units."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tag_list:
        if tag not in us_gaap:
            continue
        units = us_gaap[tag].get("units", {})
        rows = units.get("USD", [])
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # Keep only actual quarterly/annual filings, not amendment noise
        df = df[df["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])]
        if df.empty:
            continue
        df = df[["end", "fy", "fp", "val", "form"]].drop_duplicates()
        return df
    return pd.DataFrame()


def build_firm_panel(cik: str, facts: dict) -> pd.DataFrame:
    """Merge all concept series into one wide firm-quarter table.
    fy/fp are taken specifically from the capex series (the only place we
    actually need fiscal-year grouping) and deduplicated to exactly one
    row per end_date -- taking the LAST reported value, which corresponds
    to the filing's own reporting period rather than a comparative figure
    restated inside a later/earlier filing. This avoids the row-duplication
    bug where the same end_date appears under multiple fp labels."""
    merged = None
    capex_fy_fp = None
    for col, tags in TAG_CANDIDATES.items():
        s = extract_concept_series(facts, tags)
        if s.empty:
            continue
        if col == "capex_ytd":
            capex_fy_fp = (
                s[["end", "fy", "fp"]]
                .drop_duplicates(subset="end", keep="last")
            )
        s = s.rename(columns={"val": col})[["end", col]]
        s = s.groupby("end", as_index=False)[col].last()  # dedupe same-date filings
        merged = s if merged is None else merged.merge(s, on="end", how="outer")
    if merged is None:
        return pd.DataFrame()
    if capex_fy_fp is not None:
        merged = merged.merge(capex_fy_fp, on="end", how="left")
    else:
        merged["fy"], merged["fp"] = np.nan, np.nan

    # Fallback via accounting identity: Net PP&E = Gross PP&E - Accumulated
    # Depreciation. Fill this in wherever ppe_net is missing but both
    # components exist -- covers filers who tag the components but not
    # the net figure directly (common in early/comparative-period XBRL).
    if "ppe_gross" in merged.columns and "accum_depreciation" in merged.columns:
        computed = merged["ppe_gross"] - merged["accum_depreciation"]
        if "ppe_net" not in merged.columns:
            merged["ppe_net"] = np.nan
        merged["ppe_net"] = merged["ppe_net"].fillna(computed)

    merged["cik"] = cik
    merged = merged.rename(columns={"end": "end_date"})
    return merged


def main():
    viol = pd.read_csv(VIOLATIONS_CSV, dtype={"cik10": str})
    viol = viol[~viol["cik"].isin(EXCLUDED_CIKS)]
    if "gvkey" not in viol.columns:
        viol["gvkey"] = pd.NA  # not available from full-text-derived events; not needed for the EDGAR pull itself
    firms = viol[["gvkey", "cik", "cik10"]].drop_duplicates()
    print(f"Pulling {len(firms)} firms from EDGAR...")

    all_panels = []
    missing = []

    for i, row in firms.iterrows():
        cik10 = row["cik10"]
        facts = fetch_company_facts(cik10)
        if facts is None:
            missing.append(row["cik"])
            time.sleep(0.15)
            continue
        panel = build_firm_panel(cik10, facts)
        if panel.empty:
            missing.append(row["cik"])
        else:
            panel["gvkey"] = row["gvkey"]
            all_panels.append(panel)
        time.sleep(0.15)  # stay under 10 req/sec
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(firms)} done")

    if not all_panels:
        print("No data retrieved for any firm. Check USER_AGENT and network access.")
        return

    full_panel = pd.concat(all_panels, ignore_index=True)
    full_panel["end_date"] = pd.to_datetime(full_panel["end_date"])
    # Some CIKs appear under more than one gvkey in the violations file
    # (e.g. a ticker/company history quirk), which pulls the same firm
    # twice. Drop exact duplicate firm-quarters, keeping the first.
    before_dedupe = len(full_panel)
    full_panel = full_panel.drop_duplicates(subset=["cik", "end_date"], keep="first")
    if before_dedupe != len(full_panel):
        print(f"Dropped {before_dedupe - len(full_panel)} duplicate firm-quarter rows "
              f"(same CIK pulled under >1 gvkey).")

    # ---- merge Bind ----
    # Bind = 1 if this firm had a violation event with a fiscal-period 'date'
    # matching this quarter's end_date (same quarter).
    viol["date"] = pd.to_datetime(viol["date"])
    viol_flags = viol[["cik", "date"]].drop_duplicates()
    viol_flags["bind"] = 1

    full_panel["cik"] = full_panel["cik"].astype(int)
    merged = full_panel.merge(
        viol_flags, left_on=["cik", "end_date"], right_on=["cik", "date"], how="left"
    )
    merged["bind"] = merged["bind"].fillna(0).astype(int)
    merged = merged.drop(columns=["date"])

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(merged)} firm-quarter rows to {OUTPUT_CSV}")
    print(f"Firms with no usable data: {len(missing)} of {len(firms)}")
    print(f"Bind=1 rows in final panel: {int(merged['bind'].sum())}")
    if missing:
        pd.Series(missing, name="cik").to_csv("edgar_missing_firms.csv", index=False)
        print("List of missing firms saved to edgar_missing_firms.csv")


if __name__ == "__main__":
    main()
