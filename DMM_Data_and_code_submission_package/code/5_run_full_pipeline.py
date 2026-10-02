"""
Full empirical pipeline: clean the raw EDGAR pull, build Appendix-A style
variables, and run every specification (main panel regression, RDD, fuzzy
RDD, reduced-form, heterogeneity, threshold sensitivity).

Requires: edgar_firm_quarter_panel.csv (from build_edgar_panel.py, or
combine_samples.py for the combined sample) in the same folder.

Run locally:
    pip install pandas numpy linearmodels statsmodels
    python run_full_pipeline.py

If running in a notebook: paste this ENTIRE file into ONE fresh cell and
run it top to bottom. Don't reuse older cells from previous versions.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# CONFIG -- change ACTIVE_SAMPLE to switch samples. This replaces manually
# editing a bare date string, which was a real source of bugs earlier
# (a cutoff left over from a different sample silently corrupted results).
# ---------------------------------------------------------------------
SAMPLE_CONFIGS = {
    "sample1": {"cutoff": "2011-12-31", "desc": "Disclosed violations (Roberts/Sufi), 2009-2011"},
    "sample2": {"cutoff": "2014-12-31", "desc": "Full-text search, narrow window, 2012-2014"},
    "sample3": {"cutoff": "2019-12-31", "desc": "Full-text search, extended window, 2010-2019"},
    "combined": {"cutoff": "2019-12-31", "desc": "Sample 1 + Sample 3 combined (per-firm cutoffs already applied in combine_samples.py)"},
}
ACTIVE_SAMPLE = "combined"  # <-- set this to match whichever edgar_firm_quarter_panel.csv is loaded

INPUT_CSV = "edgar_firm_quarter_panel.csv"
CUTOFF_DATE = SAMPLE_CONFIGS[ACTIVE_SAMPLE]["cutoff"]
RDD_THRESHOLD = 1.00   # current ratio cutoff used as the constructed RDD proxy
RDD_BANDWIDTH = 0.20   # discontinuity-sample bandwidth, matches paper's convention
THRESHOLD_SENSITIVITY_VALUES = [0.75, 0.90, 1.00, 1.10, 1.25, 1.50]


# =======================================================================
# STEP 1: Clean the raw pull and build Appendix-A style variables
# =======================================================================
def clean_panel() -> pd.DataFrame:
    print(f"[clean] Active sample: '{ACTIVE_SAMPLE}' -- {SAMPLE_CONFIGS[ACTIVE_SAMPLE]['desc']}")

    df = pd.read_csv(INPUT_CSV)
    df["end_date"] = pd.to_datetime(df["end_date"])

    before = len(df)
    df = df[df["end_date"] <= CUTOFF_DATE].copy()
    print(f"[clean] Dropped {before - len(df)} rows after {CUTOFF_DATE} cutoff. {len(df)} rows remain.")

    # Safety check: warn if CUTOFF_DATE looks mismatched with this sample's
    # actual violation data (e.g. ACTIVE_SAMPLE set to the wrong sample).
    max_bind_date = df.loc[df["bind"] == 1, "end_date"].max()
    cutoff_ts = pd.Timestamp(CUTOFF_DATE)
    if pd.notna(max_bind_date) and (cutoff_ts - max_bind_date).days > 400:
        print(f"[WARNING] Last Bind=1 event in this file is {max_bind_date.date()}, but "
              f"CUTOFF_DATE is {CUTOFF_DATE} -- over a year later. Check that ACTIVE_SAMPLE "
              f"matches the violations file that actually built THIS edgar_firm_quarter_panel.csv.")

    df = df.sort_values(["cik", "end_date"]).reset_index(drop=True)

    # Forward-fill balance-sheet items within each firm -- 10-Qs don't
    # always restate every line item every quarter.
    bs_cols = ["assets", "current_assets", "current_liabilities",
               "ppe_net", "total_debt_lt", "total_debt_st"]
    df[bs_cols] = df.groupby("cik")[bs_cols].ffill()

    # De-cumulate capex using the firm's TRUE fiscal year (fy), not a
    # calendar-year approximation -- fixes misgrouping for firms whose
    # fiscal year doesn't end in December.
    group_year = df["fy"].fillna(df["end_date"].dt.year) if "fy" in df.columns and df["fy"].notna().any() \
        else df["end_date"].dt.year
    df["capex_q"] = df.groupby(["cik", group_year])["capex_ytd"].diff()
    is_first = df["capex_q"].isna() & df["capex_ytd"].notna()
    df.loc[is_first, "capex_q"] = df.loc[is_first, "capex_ytd"]
    df.loc[df["capex_q"] < 0, "capex_q"] = np.nan

    df["ppe_net_lag"] = df.groupby("cik")["ppe_net"].shift(1)
    df["assets_lag"] = df.groupby("cik")["assets"].shift(1)
    df["investment"] = df["capex_q"] / df["ppe_net_lag"]              # paper's original scaling
    df["investment_alt"] = df["capex_q"] / df["assets_lag"]           # alt. scaling, robustness check
    df["cash_flow"] = (df["net_income"].fillna(0) + df["depreciation"].fillna(0)) / df["ppe_net_lag"]

    total_debt = df["total_debt_lt"].fillna(0) + df["total_debt_st"].fillna(0)
    df["leverage"] = total_debt / df["assets"]
    df["log_assets"] = np.log(df["assets"].where(df["assets"] > 0))

    df["quarter"] = df["end_date"].dt.to_period("Q").astype(str)
    df["year"] = df["end_date"].dt.year
    df["bind_lag"] = df.groupby("cik")["bind"].shift(1)

    reg_cols = ["investment", "bind_lag", "cash_flow", "log_assets", "leverage"]
    clean = df.dropna(subset=reg_cols).copy()

    # Winsorize at 1%/99% to limit the influence of data errors, standard
    # practice in this literature. Investment and its alt-scaled version
    # are winsorized separately since their scales differ.
    for col in ["investment", "investment_alt"]:
        if clean[col].notna().sum() > 10:
            lo, hi = clean[col].quantile([0.01, 0.99])
            clean[col] = clean[col].clip(lo, hi)

    print(f"[clean] Regression-ready sample: {len(clean)} firm-quarters, {clean.cik.nunique()} firms")
    print(f"[clean] Bind_lag=1 observations: {int(clean['bind_lag'].sum())}, "
          f"firms with >=1 violation: {clean.loc[clean.bind_lag == 1, 'cik'].nunique()}")

    clean.to_csv("regression_ready_panel.csv", index=False)
    print("[clean] Saved to regression_ready_panel.csv")

    _export_stata_panel(clean)
    return clean


def _export_stata_panel(clean: pd.DataFrame) -> None:
    """Stata dislikes long/mixed-case column names and period-string time
    vars, so export a clean numeric quarter id and plain firm id."""
    stata_df = add_rdd_variables(clean, threshold=RDD_THRESHOLD)
    stata_df["qtr_id"] = pd.PeriodIndex(stata_df["quarter"], freq="Q").astype(int)
    stata_df["firm_id"] = stata_df["cik"].astype(int)
    keep_cols = ["firm_id", "qtr_id", "investment", "bind_lag", "cash_flow",
                 "log_assets", "leverage", "treat", "distance"]
    stata_df[keep_cols].to_csv("stata_analysis_panel.csv", index=False)
    print("[clean] Saved Stata-ready panel to stata_analysis_panel.csv\n")


# =======================================================================
# SHARED HELPERS
# =======================================================================
def add_rdd_variables(df: pd.DataFrame, threshold: float = RDD_THRESHOLD) -> pd.DataFrame:
    """Builds the constructed RDD running variable and treatment indicator
    (current ratio vs. `threshold`), plus the leverage-interacted joint-
    stress heterogeneity proxy. Shared by every RDD-related specification
    below so this logic exists in exactly one place."""
    df = df.copy()
    df["current_ratio"] = df["current_assets"] / df["current_liabilities"]
    df = df[np.isfinite(df["current_ratio"])].copy()

    df["distance"] = df["current_ratio"] - threshold
    df["treat"] = (df["distance"] < 0).astype(int)
    df["treat_x_distance"] = df["treat"] * df["distance"]

    median_leverage = df["leverage"].median()
    df["high_leverage"] = (df["leverage"] > median_leverage).astype(int)
    df["joint_stress"] = ((df["treat"] == 1) & (df["high_leverage"] == 1)).astype(int)
    df["distance_x_stress"] = df["distance"] * df["joint_stress"]

    return df


def run_regression(data: pd.DataFrame, x_cols: list, label: str,
                    time_effects: bool = True, y_col: str = "investment"):
    """Fits a firm+time fixed-effects panel regression, clustered by firm.
    Shared estimation routine used by every specification below."""
    from linearmodels.panel import PanelOLS

    if len(data) < 30:
        print(f"[{label}] Sample too small to run reliably ({len(data)} rows). Skipping.\n")
        return None

    data = data.copy()
    # linearmodels needs a real date/period type for the time index, not a
    # plain string -- PeriodIndex alone isn't recognized either, it must be
    # converted to an actual Timestamp.
    data["quarter"] = pd.PeriodIndex(data["quarter"], freq="Q").to_timestamp()
    panel_df = data.set_index(["cik", "quarter"])

    y = panel_df[y_col]
    X = panel_df[x_cols].assign(const=1)

    try:
        model = PanelOLS(y, X, entity_effects=True, time_effects=time_effects, drop_absorbed=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        print(result)
        return result
    except Exception as e:
        print(f"[{label}] Regression failed: {e}\n")
        return None


# =======================================================================
# STEP 2: Main violation-based regression (paper's core specification)
# =======================================================================
def run_main_regression(clean: pd.DataFrame) -> None:
    violators = clean.loc[clean.bind_lag == 1, "cik"].unique()
    sample = clean[clean.cik.isin(violators)]

    run_regression(
        sample,
        x_cols=["bind_lag", "cash_flow", "log_assets", "leverage"],
        label="MAIN REGRESSION (violators only, Bind_lag)",
    )

    # PROOF: run on the ENTIRE sample (violators + never-violators). A firm
    # that never violates has Bind=0 every quarter -- constant within that
    # firm, so the firm fixed effect absorbs it entirely and it contributes
    # nothing to identifying the Bind coefficient. This directly tests
    # whether restricting to violators (as the original paper does) changes
    # the estimate, or just drops uninformative rows.
    run_regression(
        clean,
        x_cols=["bind_lag", "cash_flow", "log_assets", "leverage"],
        label="MAIN REGRESSION -- ENTIRE SAMPLE incl. never-violators (proof check)",
    )

    # Robustness check: year FE instead of quarter FE (fewer time dummies
    # relative to sample size -- a standard trade-off, not a trick).
    yr_dummies = pd.get_dummies(sample["year"], prefix="yr", drop_first=True)
    sample_yr = pd.concat([sample, yr_dummies], axis=1)
    run_regression(
        sample_yr,
        x_cols=["bind_lag", "cash_flow", "log_assets", "leverage"] + list(yr_dummies.columns),
        label="MAIN REGRESSION -- YEAR FE instead of quarter FE (violators only)",
        time_effects=False,
    )

    # Robustness check: alternative dependent variable (capex / lagged assets).
    alt_sample = sample.dropna(subset=["investment_alt"])
    run_regression(
        alt_sample,
        x_cols=["bind_lag", "cash_flow", "log_assets", "leverage"],
        label="MAIN REGRESSION -- Investment = capex / lagged ASSETS (alt. scaling)",
        y_col="investment_alt",
    )


# =======================================================================
# STEP 3: RDD analysis using constructed current-ratio threshold
# =======================================================================
def run_rdd(clean: pd.DataFrame) -> None:
    df = add_rdd_variables(clean)

    overlap = pd.crosstab(df["bind_lag"], df["treat"])
    print("\n[rdd] Cross-tab: disclosed violation (bind_lag) vs constructed threshold (treat)")
    print(overlap)
    print(f"\n[rdd] Full sample: {len(df)} firm-quarters, {df.cik.nunique()} firms, "
          f"treat=1: {int(df['treat'].sum())}")

    x_cols = ["treat", "distance", "treat_x_distance", "cash_flow", "log_assets", "leverage"]
    run_regression(df, x_cols=x_cols, label="RDD - FULL SAMPLE")

    disc = df[df["distance"].abs() < RDD_BANDWIDTH].copy()
    print(f"\n[rdd] Discontinuity sample (|distance| < {RDD_BANDWIDTH}): "
          f"{len(disc)} firm-quarters, {disc.cik.nunique()} firms")
    run_regression(disc, x_cols=x_cols, label=f"RDD - DISCONTINUITY SAMPLE (bandwidth={RDD_BANDWIDTH})")


# =======================================================================
# STEP 4: Fuzzy RDD via 2SLS -- treat instruments for bind_lag
# =======================================================================
def run_fuzzy_rdd(clean: pd.DataFrame) -> None:
    import statsmodels.api as sm
    from linearmodels.iv import IV2SLS

    df = add_rdd_variables(clean)

    # First stage, reported on its own so instrument strength is checkable
    # (rule of thumb: first-stage F > 10 is considered a "strong" instrument).
    fs_X = sm.add_constant(df[["treat", "distance", "cash_flow", "log_assets", "leverage"]])
    fs_model = sm.OLS(df["bind_lag"], fs_X, missing="drop").fit(
        cov_type="cluster", cov_kwds={"groups": df["cik"]}
    )
    print("\n" + "=" * 60)
    print("FIRST STAGE: bind_lag ~ treat + controls")
    print("=" * 60)
    print(fs_model.summary())
    f_stat = fs_model.tvalues["treat"] ** 2
    print(f"\nApprox. first-stage F on 'treat' (t^2): {f_stat:.2f} "
          f"({'looks weak, F<10' if f_stat < 10 else 'looks reasonably strong, F>=10'})")

    # Second stage: 2SLS
    iv_df = df.dropna(subset=["investment", "bind_lag", "treat", "distance",
                               "cash_flow", "log_assets", "leverage"])
    model = IV2SLS(
        iv_df["investment"],
        sm.add_constant(iv_df[["distance", "cash_flow", "log_assets", "leverage"]]),
        iv_df[["bind_lag"]],
        iv_df[["treat"]],
    )
    result = model.fit(cov_type="clustered", clusters=iv_df["cik"])
    print("\n" + "=" * 60)
    print("FUZZY RDD (2SLS): bind_lag instrumented by treat")
    print("=" * 60)
    print(result)


# =======================================================================
# STEP 5: Reduced-form / intent-to-treat regression
# Since the first stage is weak, this directly tests whether crossing the
# constructed threshold shifts investment -- without relying on the
# (unreliable) 2SLS scaling by a weak first stage.
# =======================================================================
def run_reduced_form(clean: pd.DataFrame) -> None:
    df = add_rdd_variables(clean)
    x_cols = ["treat", "distance", "cash_flow", "log_assets", "leverage"]

    run_regression(df, x_cols=x_cols, label="REDUCED-FORM / ITT: investment ~ treat (full sample)")

    disc = df[df["distance"].abs() < RDD_BANDWIDTH].copy()
    run_regression(disc, x_cols=x_cols,
                    label=f"REDUCED-FORM / ITT: investment ~ treat (bandwidth={RDD_BANDWIDTH})")


# =======================================================================
# STEP 6: Continuous headroom + joint stress
# distance = continuous "headroom" (dose-response, not just a 0/1 cutoff).
# joint_stress = below the current-ratio threshold AND above-median
# leverage -- a proxy for more severe distress, echoing the paper's
# Section IV logic that the effect concentrates where agency/information
# problems are worse.
# =======================================================================
def run_joint_stress(clean: pd.DataFrame) -> None:
    df = add_rdd_variables(clean)
    print(f"\n[joint_stress] Firm-quarters with joint_stress=1: {int(df['joint_stress'].sum())} of {len(df)}")

    x_cols = ["distance", "joint_stress", "distance_x_stress", "cash_flow", "log_assets", "leverage"]
    run_regression(df, x_cols=x_cols, label="CONTINUOUS HEADROOM + JOINT STRESS (full sample)")

    disc = df[df["distance"].abs() < RDD_BANDWIDTH].copy()
    run_regression(disc, x_cols=x_cols,
                    label=f"CONTINUOUS HEADROOM + JOINT STRESS (bandwidth={RDD_BANDWIDTH})")


# =======================================================================
# STEP 7: Second RDD proxy -- net worth crossing zero
# Net worth = Assets - Total Debt (a rough book-equity proxy, since we
# only pulled debt, not full liabilities). Mirrors the original paper's
# SECOND covenant type (current ratio was the first). Note this ratio is
# mechanically close to (1 - leverage) given how our variables are
# constructed, so treat it as a related, not fully independent, check.
# =======================================================================
def run_networth_rdd(clean: pd.DataFrame) -> None:
    df = clean.copy()
    total_debt = df["total_debt_lt"].fillna(0) + df["total_debt_st"].fillna(0)
    df["net_worth"] = df["assets"] - total_debt
    df["distance_nw"] = df["net_worth"] / df["assets"]  # already centered at 0
    df = df[np.isfinite(df["distance_nw"])].copy()

    df["treat_nw"] = (df["distance_nw"] < 0).astype(int)
    df["treat_nw_x_distance"] = df["treat_nw"] * df["distance_nw"]

    overlap = pd.crosstab(df["bind_lag"], df["treat_nw"])
    print("\n[networth] Cross-tab: disclosed violation (bind_lag) vs net-worth-below-zero (treat_nw)")
    print(overlap)
    print(f"[networth] Full sample: {len(df)} firm-quarters, {df.cik.nunique()} firms, "
          f"treat_nw=1: {int(df['treat_nw'].sum())}")

    x_cols = ["treat_nw", "distance_nw", "treat_nw_x_distance", "cash_flow", "log_assets", "leverage"]
    run_regression(df, x_cols=x_cols, label="NET WORTH RDD - FULL SAMPLE")

    disc = df[df["distance_nw"].abs() < RDD_BANDWIDTH].copy()
    print(f"\n[networth] Discontinuity sample (|distance_nw| < {RDD_BANDWIDTH}): "
          f"{len(disc)} firm-quarters, {disc.cik.nunique()} firms")
    run_regression(disc, x_cols=x_cols,
                    label=f"NET WORTH RDD - DISCONTINUITY SAMPLE (bandwidth={RDD_BANDWIDTH})")


# =======================================================================
# STEP 8: Two-instrument IV (distance + joint_stress) for bind_lag, with a
# real first-stage joint F-test and an overidentification (Sargan) test.
# =======================================================================
def run_two_instrument_iv(clean: pd.DataFrame) -> None:
    import statsmodels.api as sm
    from linearmodels.iv import IV2SLS

    df = add_rdd_variables(clean)
    df = df.dropna(subset=["investment", "bind_lag", "distance", "joint_stress",
                            "cash_flow", "log_assets", "leverage"])

    print(f"\n[iv2] Sample: {len(df)} firm-quarters, {df.cik.nunique()} firms, "
          f"joint_stress=1: {int(df['joint_stress'].sum())}")

    fs_X = sm.add_constant(df[["distance", "joint_stress", "cash_flow", "log_assets", "leverage"]])
    fs_model = sm.OLS(df["bind_lag"], fs_X).fit(cov_type="cluster", cov_kwds={"groups": df["cik"]})
    print("\n" + "=" * 60)
    print("FIRST STAGE: bind_lag ~ distance + joint_stress + controls")
    print("=" * 60)
    print(fs_model.summary())

    joint_test = fs_model.f_test("distance = 0, joint_stress = 0")
    print(f"\nJoint F-test on (distance, joint_stress) = 0: "
          f"F = {float(joint_test.fvalue):.3f}, p = {float(joint_test.pvalue):.4f}")
    print("(Rule of thumb: F > 10 is considered a strong joint instrument.)")

    model = IV2SLS(
        df["investment"],
        sm.add_constant(df[["cash_flow", "log_assets", "leverage"]]),
        df[["bind_lag"]],
        df[["distance", "joint_stress"]],
    )
    result = model.fit(cov_type="clustered", clusters=df["cik"])
    print("\n" + "=" * 60)
    print("2SLS: bind_lag instrumented by (distance, joint_stress)")
    print("=" * 60)
    print(result)

    try:
        sargan = result.sargan
        print(f"\nSargan overidentification test: stat = {sargan.stat:.3f}, p = {sargan.pval:.4f}")
        print("(A LOW p-value here would suggest the two instruments do not agree --")
        print(" i.e. at least one is likely invalid. This test assumes homoskedasticity,")
        print(" so treat it as indicative alongside, not a replacement for, the")
        print(" clustered-SE results above.)")
    except Exception as e:
        print(f"\nSargan test unavailable: {e}")


# =======================================================================
# STEP 9: Threshold sensitivity -- vary the current-ratio cutoff and
# report EVERY result, not just the best-looking one. Tests whether 1.0
# was a reasonable choice, or whether (in)significance is an artefact of
# that specific number.
# =======================================================================
def run_threshold_sensitivity(clean: pd.DataFrame) -> None:
    results_summary = []

    for thresh in THRESHOLD_SENSITIVITY_VALUES:
        df = add_rdd_variables(clean, threshold=thresh)
        x_cols = ["treat", "distance", "treat_x_distance", "cash_flow", "log_assets", "leverage"]
        result = run_regression(df, x_cols=x_cols, label=f"[threshold sensitivity: {thresh}]")
        n_treat = int(df["treat"].sum())
        if result is not None:
            results_summary.append((thresh, n_treat, result.params["treat"],
                                     result.std_errors["treat"], result.pvalues["treat"]))
        else:
            results_summary.append((thresh, n_treat, None, None, None))

    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY: varying the current-ratio cutoff")
    print("=" * 70)
    print(f"{'Threshold':>10} {'N treat=1':>10} {'Coef (treat)':>14} {'Std Err':>10} {'p-value':>10}")
    for thresh, n_treat, coef, se, pval in results_summary:
        if coef is None:
            print(f"{thresh:>10.2f} {n_treat:>10} {'FAILED':>14} {'--':>10} {'--':>10}")
        else:
            print(f"{thresh:>10.2f} {n_treat:>10} {coef:>14.4f} {se:>10.4f} {pval:>10.4f}")
    print("\nNote: all thresholds are reported here, not just the most favourable one.")


if __name__ == "__main__":
    clean = clean_panel()
    run_main_regression(clean)
    run_rdd(clean)
    run_fuzzy_rdd(clean)
    run_reduced_form(clean)
    run_joint_stress(clean)
    run_networth_rdd(clean)
    run_two_instrument_iv(clean)
    run_threshold_sensitivity(clean)
