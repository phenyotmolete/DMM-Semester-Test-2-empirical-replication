"""
STAGE 2: For each candidate from Stage 1, fetch the actual filing text and
pull out the sentence around the matched phrase, so we can tell real
violations apart from false positives like "we were NOT in violation of...".

This does a simple automatic negation check (fast, catches the obvious
cases) AND saves the actual context text so you can eyeball the harder
ones yourself -- don't fully trust the automatic flag alone.

Run locally:
    pip install requests pandas
    python verify_violation_candidates.py
"""

import re
import time
import requests
import pandas as pd

USER_AGENT = "YourName YourEmail@university.edu"  # <-- REPLACE ME
HEADERS = {"User-Agent": USER_AGENT}

INPUT_CSV = "fulltext_violation_candidates.csv"
OUTPUT_CSV = "violation_candidates_verified.csv"

# Words appearing shortly before the matched phrase that suggest it's
# actually a NEGATED statement ("we were NOT in violation...").
NEGATION_WORDS = ["not ", "no ", "n't ", "without ", "never "]

CONTEXT_CHARS = 300  # characters of context to pull around each match


def fetch_filing_text(cik: str, adsh: str) -> str | None:
    """Fetch the full submission text file for one filing."""
    adsh_nodash = adsh.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh}.txt"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None
    return resp.text


def extract_context(text: str, phrase: str) -> str | None:
    """Find the phrase (case-insensitive) and return surrounding context."""
    idx = text.lower().find(phrase.lower())
    if idx == -1:
        return None
    start = max(0, idx - CONTEXT_CHARS)
    end = min(len(text), idx + len(phrase) + CONTEXT_CHARS)
    snippet = text[start:end]
    # collapse whitespace/HTML tags roughly for readability
    snippet = re.sub(r"<[^>]+>", " ", snippet)
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet


def looks_negated(snippet: str, phrase: str) -> bool:
    """Very simple heuristic: check for a negation word in the ~40 chars
    immediately before the matched phrase. Not perfect -- always eyeball
    the actual snippet for anything that matters for your final sample."""
    idx = snippet.lower().find(phrase.lower())
    if idx == -1:
        return False
    window = snippet[max(0, idx - 40):idx].lower()
    return any(neg in window for neg in NEGATION_WORDS)


def main():
    df = pd.read_csv(INPUT_CSV)
    results = []

    for i, row in df.iterrows():
        cik, adsh, phrase = row["cik"], row["adsh"], row["phrase"]
        text = fetch_filing_text(cik, adsh)
        time.sleep(0.15)

        if text is None:
            results.append({**row, "snippet": None, "likely_negated": None, "status": "fetch_failed"})
            continue

        snippet = extract_context(text, phrase)
        if snippet is None:
            results.append({**row, "snippet": None, "likely_negated": None, "status": "phrase_not_found_in_doc"})
            continue

        negated = looks_negated(snippet, phrase)
        results.append({
            **row,
            "snippet": snippet,
            "likely_negated": negated,
            "status": "ok",
        })

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(df)} processed")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)

    n_ok = (out["status"] == "ok").sum()
    n_negated = (out["likely_negated"] == True).sum()
    n_clean = ((out["status"] == "ok") & (out["likely_negated"] == False)).sum()

    print(f"\nProcessed: {len(out)}")
    print(f"Successfully fetched & found phrase: {n_ok}")
    print(f"Flagged as likely NEGATED (false positive): {n_negated}")
    print(f"Likely genuine violation mentions: {n_clean}")
    print(f"\nSaved to {OUTPUT_CSV} -- open it and read the 'snippet' column")
    print("for anything ambiguous before treating a row as a real violation.")


if __name__ == "__main__":
    main()
