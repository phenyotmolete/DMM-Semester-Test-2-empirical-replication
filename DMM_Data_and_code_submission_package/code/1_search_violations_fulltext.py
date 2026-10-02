"""
STAGE 1: Search SEC EDGAR full-text search for candidate covenant
violation disclosures, restricted to a date window where XBRL coverage
is clean/universal (2012+, well past the June 2011 mandate for all filers).

This does NOT try to filter false positives yet (e.g. "we were NOT in
violation of..." will still show up as a hit) -- that's Stage 2, done only
after we see whether this approach turns up enough real candidates to be
worth the extra work.

Run locally:
    pip install requests pandas
    python search_violations_fulltext.py
"""

import requests
import time
import pandas as pd

USER_AGENT = "YourName YourEmail@university.edu"  # <-- REPLACE ME
HEADERS = {"User-Agent": USER_AGENT}

BASE_URL = "https://efts.sec.gov/LATEST/search-index"

# Candidate phrases used to find covenant violation disclosures.
# Each is searched as an exact phrase (quoted). We deliberately cast a
# slightly wide net here -- narrowing/false-positive filtering is Stage 2.
SEARCH_PHRASES = [
    "in violation of certain financial covenants",
    "was not in compliance with certain financial covenants",
    "breach of a financial covenant",
    "waived the Company's noncompliance",
    "technical default under",
]

FORMS = "10-Q,10-K"
START_DATE = "2012-01-01"
END_DATE = "2019-12-31"  # extended window, but stops before COVID-era distortions
PAGE_SIZE = 10  # fixed by the API
MAX_HITS_PER_PHRASE = 1000  # raised from 200 since the wider window will exceed that


def search_phrase(phrase: str) -> list[dict]:
    """Page through all results for one phrase, return raw hit dicts."""
    hits = []
    offset = 0
    while True:
        params = {
            "q": f'"{phrase}"',
            "forms": FORMS,
            "startdt": START_DATE,
            "enddt": END_DATE,
            "from": offset,
        }
        resp = requests.get(BASE_URL, params=params, headers=HEADERS)
        if resp.status_code != 200:
            print(f"  [warn] request failed ({resp.status_code}) at offset {offset}")
            break
        data = resp.json()
        batch = data.get("hits", {}).get("hits", [])
        if not batch:
            break
        for h in batch:
            src = h.get("_source", {})
            hits.append({
                "phrase": phrase,
                "cik": src.get("ciks", [None])[0],
                "company": src.get("display_names", [None])[0],
                "form": src.get("forms", [None])[0] if isinstance(src.get("forms"), list) else src.get("form"),
                "file_date": src.get("file_date"),
                "adsh": src.get("adsh"),  # accession number
            })
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        offset += PAGE_SIZE
        time.sleep(0.15)
        if offset >= total or offset >= MAX_HITS_PER_PHRASE:
            break
    return hits


def main():
    all_hits = []
    for phrase in SEARCH_PHRASES:
        print(f"Searching: \"{phrase}\" ...")
        hits = search_phrase(phrase)
        print(f"  -> {len(hits)} hits")
        all_hits.extend(hits)

    df = pd.DataFrame(all_hits)
    if df.empty:
        print("\nNo hits at all -- check USER_AGENT, date range, or phrasing.")
        return

    df = df.drop_duplicates(subset=["cik", "adsh"])
    print(f"\nTotal unique candidate filings: {len(df)}")
    print(f"Unique companies (by CIK): {df['cik'].nunique()}")
    print("\nBreakdown by phrase:")
    print(df["phrase"].value_counts().to_string())
    print("\nBreakdown by year:")
    df["year"] = df["file_date"].str[:4]
    print(df["year"].value_counts().sort_index().to_string())

    df.to_csv("fulltext_violation_candidates.csv", index=False)
    print("\nSaved candidate list to fulltext_violation_candidates.csv")
    print("NEXT STEP: manually spot-check a sample of these for false positives")
    print("(e.g. sentences containing 'not in violation') before treating this")
    print("as a real violation-events list.")


if __name__ == "__main__":
    main()
