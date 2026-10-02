"""
Combines Sample 1 (Roberts/Sufi disclosed violations, 2009-2011) and
Sample 3 (full-text search, 2010-2019) into ONE larger dataset, since
they are built from genuinely different, mostly non-overlapping firms.

This is an honest way to increase sample size: combining two
independently-sourced real datasets, not tuning a specification.

Each firm's Bind=0 quarters are only trusted within THAT firm's own
verified search window (the sample it originated from), since we only
know a firm is "not violating" for years we actually searched for it.

Run locally:
    pip install pandas numpy
    python combine_samples.py

Inputs (put both in the same folder):
    sample1_edgar_panel.csv   -- rename edgar_firm_quarter_panel-4.csv to this
    sample3_edgar_panel.csv   -- rename edgar_firm_quarter_panel-6.csv to this

Output:
    combined_edgar_panel.csv -- feed this into run_full_pipeline.py
    (set CUTOFF_DATE per-row logic is handled here already, so
    run_full_pipeline.py's single CUTOFF_DATE should be set generously,
    e.g. "2019-12-31" -- the real per-firm trimming already happened here)
"""

import pandas as pd

SAMPLE1_CUTOFF = "2011-12-31"  # Roberts/Sufi violations file coverage ends here
SAMPLE3_CUTOFF = "2019-12-31"  # full-text search window end

s1 = pd.read_csv("sample1_edgar_panel.csv")
s3 = pd.read_csv("sample3_edgar_panel.csv")

s1["end_date"] = pd.to_datetime(s1["end_date"])
s3["end_date"] = pd.to_datetime(s3["end_date"])

# Trim each sample to ITS OWN verified window before combining -- this is
# the critical step. A Sample-1-origin firm's quarters after 2011 are NOT
# verified (we never searched for later violations for that firm via the
# Roberts/Sufi source), so they must be dropped, not silently kept.
s1 = s1[s1["end_date"] <= SAMPLE1_CUTOFF].copy()
s3 = s3[s3["end_date"] <= SAMPLE3_CUTOFF].copy()

s1["source"] = "sample1_disclosed"
s3["source"] = "sample3_fulltext"

print(f"Sample 1 (trimmed to {SAMPLE1_CUTOFF}): {len(s1)} rows, {s1.cik.nunique()} firms")
print(f"Sample 3 (trimmed to {SAMPLE3_CUTOFF}): {len(s3)} rows, {s3.cik.nunique()} firms")

# For the 6 overlapping firms, keep Sample 3's row if both sources have
# data for the same (cik, end_date) -- Sample 3's more recent pull included
# the accounting-identity fallback fixes, so prefer it on conflict.
combined = pd.concat([s1, s3], ignore_index=True)
before = len(combined)
combined = combined.sort_values("source").drop_duplicates(subset=["cik", "end_date"], keep="last")
print(f"\nAfter combining and deduping overlaps: {before} -> {len(combined)} rows")
print(f"Combined unique firms: {combined.cik.nunique()}")
print(f"Combined Bind=1 events: {int(combined['bind'].sum())}")

violators = combined.loc[combined["bind"] == 1, "cik"].unique()
print(f"Firms with >=1 violation in combined sample: {len(violators)}")

combined.to_csv("combined_edgar_panel.csv", index=False)
print("\nSaved to combined_edgar_panel.csv")
print("\nNEXT STEP: rename this to edgar_firm_quarter_panel.csv, set")
print("CUTOFF_DATE = '2019-12-31' in run_full_pipeline.py (the real per-firm")
print("trimming already happened here), and run the full pipeline.")
