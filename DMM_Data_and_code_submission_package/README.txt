EMPIRICAL REPLICATION PROJECT -- README
Chava & Roberts (2008), "How Does Financing Impact Investment? The Role of Debt Covenants"

SOFTWARE USED:

- Python 3.x
  Packages: pandas, numpy, requests, linearmodels, statsmodels, python-dateutil
  Install with: pip install pandas numpy requests linearmodels statsmodels python-dateutil

- Stata (for the final regression specifications, per course preference)
  Script tested on a standard Stata installation; no special packages required.

The csv files required for the stata run are included in the combined sample folder. 

FOLDER STRUCTURE

code/
  1_search_violations_fulltext.py    Stage 1: searches SEC EDGAR full-text
                                      search for candidate covenant-violation
                                      disclosures (used to build Samples 2 & 3)
  2_verify_violation_candidates.py   Stage 2: fetches each candidate filing's
                                      actual text and flags negated statements
                                      (e.g. "we were NOT in violation...")
  3_build_fulltext_bind_events.py    Stage 3: extracts the violation period
                                      date from each verified snippet, builds
                                      the final events file
  4_build_edgar_panel.py             Pulls firm-level financials (capex, PP&E,
                                      assets, etc.) from SEC EDGAR's XBRL
                                      Company Facts API for a given violations
                                      events file
  5_run_full_pipeline.py             Cleans the raw EDGAR pull, builds all
                                      regression variables, and runs every
                                      specification (main panel regression,
                                      RDD, fuzzy RDD, reduced-form,
                                      heterogeneity, threshold sensitivity)
  6_analysis_stata.do                Stata equivalent of the main
                                      specifications, reading the
                                      Stata-ready panel exported by script 5
  7_combine_samples.py               Merges Sample 1 and Sample 3's EDGAR
                                      pulls into the combined sample, with
                                      per-firm cutoff trimming

data/
  sample1_disclosed_2009_2011/       Roberts & Sufi (2009) disclosed
                                      violations, restricted to 2009-2011
  sample2_fulltext_2012_2014/        Our own full-text search, narrow window
  sample3_fulltext_2010_2019/        Our own full-text search, extended window
  combined_sample/                   Combined Sample 1 and 3. 

EXECUTION ORDER (to reproduce from scratch)

For Sample 1 (disclosed violations):
  1. Start from data/sample1_disclosed_2009_2011/raw_source_roberts_sufi_violations.xlsx
  2. Filter to 2009-2011 (see violations_filtered_2009_2011.csv for the
     already-filtered result)
  3. Run 4_build_edgar_panel.py (VIOLATIONS_CSV pointed at the filtered file)
  4. Run 5_run_full_pipeline.py (set ACTIVE_SAMPLE = "sample1")

For Samples 2 and 3 (full-text search):
  1. Run 1_search_violations_fulltext.py (set START_DATE/END_DATE for the
     desired window: 2012-2014 for Sample 2, 2012-2019 for Sample 3)
  2. Run 2_verify_violation_candidates.py
  3. Run 3_build_fulltext_bind_events.py
  4. Manually review the resulting events file for non-standard
     (non-quarter-end) dates and drop/verify them (see stage3_final_events.csv
     for the already-cleaned result)
  5. Run 4_build_edgar_panel.py (VIOLATIONS_CSV pointed at the cleaned events file)
  6. Run 5_run_full_pipeline.py (set ACTIVE_SAMPLE = "sample2" or "sample3")

For the combined sample:
  1. Run 7_combine_samples.py, pointing it at Sample 1's and Sample 3's
     already-pulled edgar_financials_pulled.csv files
  2. Run 5_run_full_pipeline.py (set ACTIVE_SAMPLE = "combined")

For the Stata analysis:
  1. After running 5_run_full_pipeline.py, it exports stata_analysis_panel.csv
  2. Run 6_analysis_stata.do in Stata, with stata_analysis_panel.csv in the
     same folder
  3. The csv files required for the stata run are located in the combined sample folder. 


IMPORTANT NOTES / KNOWN GAPS IN THIS PACKAGE

1. Sample 3's Stage 1 raw candidate file (546 candidates, before
   verification) is not included here -- only its Stage 2 (verified)
   output is. The raw Stage 1 file for Sample 2 (212 candidates) IS
   included, for reference on what that stage's output looks like.

2. The combined sample's actual pulled financial panel
   (combined_edgar_panel.csv), regenerated regression_ready_panel.csv, and
   stata_analysis_panel.csv are NOT included in this package, because they
   were generated locally on the student's own machine. 
   To regenerate them: run 7_combine_samples.py
   using Sample 1's and Sample 3's edgar_financials_pulled.csv files
   (both included here), then run 5_run_full_pipeline.py.

3. regression_ready_panel.csv and stata_analysis_panel.csv for Samples 1,
   2, and 3 individually are included, in the combined sample folder.Running
   5_run_full_pipeline.py regenerates them fresh each time (same filename
   is reused per run). The complete, unedited console output from each
   run is provided separately as the required "Complete Software Output"
   submission item (see assignment instructions) -- those logs show the
   actual numbers produced by each run.

4. For a full discussion of data sources, methodology, scope limitations,
   and results, see the main report.
