* ==============================================================================
* Covenant Violations and Investment -- EDGAR-based replication
* Reads stata_analysis_panel.csv, produced by run_full_pipeline.py (Python)
* ==============================================================================

clear all
set more off
log using "stata_output.log", replace text

import delimited "stata_analysis_panel.csv", clear

xtset firm_id qtr_id

* ==============================================================================
* 1. MAIN REGRESSION -- full controls (paper's core specification, violators only)
* ==============================================================================
di "======================================================================"
di "1. MAIN REGRESSION (full controls, violators-only sample)"
di "======================================================================"

* Restrict to firms with at least one violation, per the paper's identification
* logic under firm fixed effects.
egen ever_bind = max(bind_lag), by(firm_id)
preserve
keep if ever_bind == 1
xtreg investment bind_lag cash_flow log_assets leverage i.qtr_id, fe vce(cluster firm_id)
restore

* ==============================================================================
* 2. PARSIMONIOUS SPECIFICATION -- fewer controls, to check if N/power improves
* (mirrors the original paper's approach of reporting multiple specifications
*  with varying controls, e.g. their Table V columns 1-7)
* ==============================================================================
di "======================================================================"
di "2. PARSIMONIOUS SPECIFICATION (cash_flow only, violators-only sample)"
di "======================================================================"
preserve
keep if ever_bind == 1
xtreg investment bind_lag cash_flow i.qtr_id, fe vce(cluster firm_id)
restore

* ==============================================================================
* 3. ENTIRE SAMPLE (not just violators) -- paper notes this is "qualitatively similar"
* ==============================================================================
di "======================================================================"
di "3. ENTIRE SAMPLE (all pulled firms, not just violators)"
di "======================================================================"
xtreg investment bind_lag cash_flow log_assets leverage i.qtr_id, fe vce(cluster firm_id)

* ==============================================================================
* 4. RDD -- FULL SAMPLE, interacted specification
* ==============================================================================
di "======================================================================"
di "4. RDD - FULL SAMPLE (treat x distance interaction)"
di "======================================================================"
gen treat_x_distance = treat * distance
xtreg investment treat distance treat_x_distance cash_flow log_assets leverage i.qtr_id, fe vce(cluster firm_id)

* ==============================================================================
* 5. REDUCED-FORM / ITT ACROSS MULTIPLE BANDWIDTHS
* Report ALL bandwidths -- this is a transparency check, not bandwidth-shopping.
* ==============================================================================
di "======================================================================"
di "5. REDUCED-FORM / ITT ACROSS BANDWIDTHS"
di "======================================================================"

foreach h in 0.10 0.15 0.20 0.25 {
    di "--- Bandwidth h = `h' ---"
    xtreg investment treat distance cash_flow log_assets leverage i.qtr_id ///
        if abs(distance) <= `h', fe vce(cluster firm_id)
}

* ==============================================================================
* 6. FUZZY RDD (2SLS) -- treat instruments for bind_lag
* Note: estat ffirst does not work under vce(cluster), so the first-stage
* cluster-robust F is obtained manually via regress + test.
* ==============================================================================
di "======================================================================"
di "6. FUZZY RDD (2SLS) - FIRST STAGE"
di "======================================================================"
regress bind_lag treat distance cash_flow log_assets leverage i.qtr_id, vce(cluster firm_id)
test treat
di "First-stage cluster-robust F on 'treat' reported above (test treat)."
di "Rule of thumb: F > 10 is considered a strong instrument (Stock & Yogo)."

di "======================================================================"
di "6b. FUZZY RDD (2SLS) - SECOND STAGE"
di "======================================================================"
ivregress 2sls investment distance cash_flow log_assets leverage i.qtr_id (bind_lag = treat), vce(cluster firm_id)

* ==============================================================================
* 7. CONTINUOUS HEADROOM + JOINT STRESS
* distance = continuous headroom (dose-response) rather than a 0/1 cutoff.
* joint_stress = firm is both below the current-ratio threshold AND above
* median leverage -- proxy for more severe distress (echoes paper's
* Section IV logic that the effect concentrates where agency/information
* problems are worse).
* ==============================================================================
di "======================================================================"
di "7. CONTINUOUS HEADROOM + JOINT STRESS"
di "======================================================================"

summarize leverage, detail
local med_lev = r(p50)
di "Median leverage: `med_lev'"

gen high_leverage = (leverage > `med_lev')
gen joint_stress = (treat == 1 & high_leverage == 1)
gen distance_x_stress = distance * joint_stress

count if joint_stress == 1
di "Firm-quarters with joint_stress=1: " r(N)

di "--- Full sample ---"
xtreg investment distance joint_stress distance_x_stress cash_flow log_assets leverage i.qtr_id, fe vce(cluster firm_id)

di "--- Discontinuity sample (bandwidth = 0.20) ---"
xtreg investment distance joint_stress distance_x_stress cash_flow log_assets leverage i.qtr_id ///
    if abs(distance) <= 0.20, fe vce(cluster firm_id)

log close
