"""
Turn the verified full-text violation candidates into a clean events file:
one row per (cik, violation period), matching the same shape as the
original Roberts/Sufi violations file so it drops straight into
build_edgar_panel.py.

Run locally:
    pip install pandas python-dateutil
    python build_fulltext_bind_events.py
"""

import re
import html
import pandas as pd
from dateutil import parser as dateparser

INPUT_CSV = "violation_candidates_verified.csv"
OUTPUT_CSV = "fulltext_violations_events.csv"

MONTHS = (r"Jan(?:uary|\.)?|Feb(?:ruary|\.)?|Mar(?:ch|\.)?|Apr(?:il|\.)?|May|"
          r"Jun(?:e|\.)?|Jul(?:y|\.)?|Aug(?:ust|\.)?|Sep(?:t|tember|\.)?|"
          r"Oct(?:ober|\.)?|Nov(?:ember|\.)?|Dec(?:ember|\.)?")

# Three date shapes, in priority order:
# 1. "as of <date>" / "ended <date>" -- strongest signal, this IS the period
CONTEXTUAL_DATE = re.compile(
    rf"(?:as of|ended|period ended|quarter ended|fiscal year ended)\s+"
    rf"(?:(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})",
    re.IGNORECASE,
)
# 2. Plain written month/day/year anywhere in the snippet
WRITTEN_DATE = re.compile(rf"(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}}", re.IGNORECASE)
# 3. Numeric MM/DD/YYYY or M/D/YY anywhere in the snippet
NUMERIC_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _try_parse(text: str):
    try:
        return dateparser.parse(text, fuzzy=True)
    except Exception:
        return None


def extract_dates_near_phrase(snippet: str, phrase: str) -> list:
    """Find date-like strings, preferring an explicit 'as of'/'ended <date>'
    match (which almost always IS the violation period) before falling back
    to any date in the snippet, closest to the matched phrase first."""
    # Raw filing text often has literal HTML entities (&nbsp;, &#160;, etc.)
    # sitting where a real space should be (e.g. "June&nbsp;30, 2010"), which
    # silently breaks \s-based date regexes. Decode them first.
    snippet = html.unescape(snippet)
    idx = snippet.lower().find(phrase.lower())

    # Priority 1: contextual "as of / ended <date>" phrasing
    ctx_matches = list(CONTEXTUAL_DATE.finditer(snippet))
    if ctx_matches:
        ctx_matches.sort(key=lambda m: abs(m.start() - idx))
        dates = [_try_parse(m.group()) for m in ctx_matches]
        dates = [d for d in dates if d is not None]
        if dates:
            return dates

    # Priority 2/3: any written or numeric date, closest to the phrase
    all_matches = list(WRITTEN_DATE.finditer(snippet)) + list(NUMERIC_DATE.finditer(snippet))
    if not all_matches:
        return []
    all_matches.sort(key=lambda m: abs(m.start() - idx))
    dates = [_try_parse(m.group()) for m in all_matches]
    return [d for d in dates if d is not None]


# Leasing/investment funds (ATEL, ICON, etc.) disclose covenant violations
# by their LESSEES/portfolio companies, not themselves as the borrower --
# a fundamental mismatch with this study's design (the firm disclosing the
# violation must be the one whose own investment behavior we're testing).
# Manually confirmed via a real snippet: an ATEL fund's "breach of a
# financial covenant" mention was actually about a lessee (Blackhawk
# Mining), not the fund itself.
FUND_NAME_PATTERN = (
    r"(?i)\b(ATEL|ICON\s|LEASING\s?FUND|EQUIPMENT\s?FUND|CAPITAL\s?FUND|"
    r"\bFUND\s?(X|XI|XII|XIII|XIV|XV|XVI|\d+)\b|BDC\b|BUSINESS\s?DEVELOPMENT\s?COMPANY)"
)


def main():
    df = pd.read_csv(INPUT_CSV)
    clean = df[(df["status"] == "ok") & (df["likely_negated"] == False)].copy()

    before_fund_filter = clean.cik.nunique()
    clean = clean[~clean["company"].str.contains(FUND_NAME_PATTERN, regex=True, na=False)]
    print(f"Excluded leasing/investment fund entities: "
          f"{before_fund_filter} -> {clean.cik.nunique()} unique companies remain")

    events = []
    for _, row in clean.iterrows():
        dates = extract_dates_near_phrase(str(row["snippet"]), row["phrase"])
        if not dates:
            continue
        # take the closest date to the matched phrase as the violation period
        best_date = dates[0]
        events.append({
            "cik": int(row["cik"]),
            "company": row.get("company"),
            "date": best_date.strftime("%Y-%m-%d"),
            "phrase": row["phrase"],
            "adsh": row["adsh"],
        })

    events_df = pd.DataFrame(events)
    before = len(events_df)
    events_df = events_df.drop_duplicates(subset=["cik", "date"])
    print(f"Extracted {before} date-tagged events, {len(events_df)} unique (cik, date) pairs")
    print(f"Unique companies: {events_df['cik'].nunique()}")

    events_df["cik10"] = events_df["cik"].astype(str).str.zfill(10)
    events_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print("\nSample:")
    print(events_df[["cik", "company", "date"]].head(15).to_string())
    print("\nNOTE: dates are extracted heuristically from nearby text. Spot-check")
    print("a handful against the original 'snippet' column before fully trusting them.")


if __name__ == "__main__":
    main()
