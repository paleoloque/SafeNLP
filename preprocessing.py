"""
preprocessing.py

Utility functions for cleaning and normalizing prompt/response datasets
and for unifying harm categories across different annotation schemes,
adapted and specified to BeaverTails and SafeRLHF local challenging points.

Intended to be used with Hugging Face `datasets.Dataset` / `DatasetDict`.
"""
import re, ast, unicodedata
import pandas as pd
import numpy as np

# 
# Regular expressions for text cleaning
# 

# Generic "end-of-message" markers (e.g., "eom", "end of response")
EOM_GENERIC_RE = re.compile(r"""
    ^\s* [^\w\s]{0,6}\s* (?:eom|end)\b (?:\s+of\s+(?:output|response))? [^\n]*$
""", re.IGNORECASE | re.MULTILINE | re.VERBOSE)

# Triple quotes (often indicate truncation)
TRIPLEQ_RE  = re.compile(r'"""')

# "Instruction:" headings with optional numbering prefix
INSTR_RE    = re.compile(r"\b\d+\s*\.?\s*Instruction\s*:", re.IGNORECASE)

# Emoji ranges
EMOJI_RE    = re.compile("["
    "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF" "\U00002700-\U000027BF" "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF" "]+", re.UNICODE)

# PII and URL detectors
EMAIL_RE    = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE    = re.compile(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b")
URL_PROTO_RE  = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
URL_WWW_RE    = re.compile(r"(?<!@)\bwww\.[^\s)>\]]+", re.IGNORECASE)
URL_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?-i:[a-z]{2,})(?:/[^\s)>\]]*)?", re.IGNORECASE)

# Zero-width characters
_ZW_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")

# 
# Category unification maps
# Keys = unified categories; values = source-category names to OR together
# 
BT_UNIFIED_MAP = {
    "Violence and Threats": [
        "violence,aiding_and_abetting,incitement","child_abuse","animal_abuse","self_harm",
    ],
    "Crime and Illegal Activities": [
        "financial_crime,property_crime,theft","drug_abuse,weapons,banned_substance","terrorism,organized_crime",
    ],
    "Hate Speech and Discrimination": [
        "hate_speech,offensive_language","discrimination,stereotype,injustice",
    ],
    "Sexual Content": ["sexually_explicit,adult_content"],
    "Privacy Violations": ["privacy_violation"],
    "Misinformation and Manipulation": [
        "misinformation_regarding_ethics,laws_and_safety","non_violent_unethical_behavior","controversial_topics,politics",
    ],
}

RLHF_UNIFIED_MAP = {
    "Violence and Threats": ["Endangering National Security","Violence","Human Trafficking","Physical Harm","Animal Abuse"],
    "Crime and Illegal Activities": ["Drugs","Economic Crime","Cybercrime","White-Collar Crime","Disrupting Public Order","Environmental Damage"],
    "Hate Speech and Discrimination": ["Insulting Behavior","Discriminatory Behavior"],
    "Sexual Content": ["Sexual Content"],
    "Privacy Violations": ["Privacy Violation"],
    "Misinformation and Manipulation": ["Mental Manipulation","Copyright Issues","Psychological Harm","Endangering Public Health"],
}

# 
# Helpers
# 
def _norm_key(s):
    """
    Normalize text for duplicate detection:
    - NFKC normalize, replace NBSP
    - remove zero-width chars
    - collapse whitespace, lowercase, strip
    """
    if s is None: return ""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00A0", " ")
    s = _ZW_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _strip_trailing_non_word(t):
    """
    Remove trailing non-word punctuation; keep a single terminal dot if present.
    """
    if re.search(r"\.[^\w]*$", t):
        return re.sub(r"[^\w]*$", "", t)
    return re.sub(r"\W+$", "", t)

def _clean_once(t):
    """
    One-pass cleaner for a response:
    - Cut at generic EOM markers or triple quotes if present
    - Remove emojis
    - Cut off at an 'Instruction:' heading if present
    - Mask emails/phones
    - Replace URLs with [LINK]
    - Ensure a space after periods before capital letters (e.g., 'a.B' -> 'a. B')
    - Trim trailing punctuation and collapse whitespace
    """
    if not t: return ""
    t = str(t)

    # Truncate at EOM markers or triple quotes (choose earliest cut)
    m1 = EOM_GENERIC_RE.search(t)
    m2 = TRIPLEQ_RE.search(t)
    cut_pos = min([m.start() for m in [m1, m2] if m] or [None]) if (m1 or m2) else None
    if cut_pos is not None:
        t = t[:cut_pos].rstrip()

    # Strip emojis
    t = EMOJI_RE.sub("", t)

    # Truncate at "Instruction:"
    m = INSTR_RE.search(t)
    if m: t = t[:m.start()].rstrip()

    # Mask PII
    t = EMAIL_RE.sub(" [EMAIL] ", t)
    t = PHONE_RE.sub(" [PHONE] ", t)

    # Replace any URL with [LINK]
    for rx in (URL_PROTO_RE, URL_WWW_RE, URL_DOMAIN_RE):
        t = rx.sub(" [LINK] ", t)

    # Insert a space after a period before a capital letter (sentence boundary fix)
    t = re.sub(r"([a-z])(\.)([A-Z])", r"\1. \3", t)

    # Tidy trailing punctuation and spaces
    t = _strip_trailing_non_word(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _parse_dict_cell(v):
    """
    Parse a cell that should hold a dict (e.g., category labels):
    - Pass dicts through
    - Try literal_eval on strings
    - Fallback to empty dict
    """
    if isinstance(v, dict): return v
    if pd.isna(v): return {}
    s = str(v)
    try: return ast.literal_eval(s)
    except Exception: return {}

def _unify_row(cat, is_safe_val, mapping):
    """
    Build a unified category dict for a single row:
    - Initialize all unified categories to False
    - If the instance is unsafe (is_safe==False), set a unified category True
      if any of its source categories are True.
    """
    out = {k: False for k in mapping.keys()}
    if not bool(is_safe_val):
        for uni, src_list in mapping.items():
            out[uni] = any(bool(cat.get(src, False)) for src in src_list)
    return out

# 
# Main preprocessing pipeline
# 
def preprocess_df(
    df: pd.DataFrame,
    *,
    mapping_scheme: str,   # "bt" | "rlhf"
    drop_duplicates: bool = True,
    min_len: int = 70,
    max_len: int = 2000,
):
    """
    Return a normalized DataFrame with columns:
      ['id','prompt','response','is_safe','harm_category','response_length']

    Steps:
      1) Clean 'response' text once.
      2) Filter by response length [min_len, max_len].
      3) Unify harm categories:
         - mapping_scheme='bt' expects source column 'category'
         - mapping_scheme='rlhf' expects source column 'harm_category'
         - For unsafe rows, each unified category is the OR over its mapped source categories.
      4) Optional deduplication by normalized 'response' (after cleaning here; see note).
      5) Add sequential 'id' (0..N-1).
      6) Reorder/ensure output columns; fill missing with NaN or {} for 'harm_category'.

    Notes:
      - Requires columns: 'response', 'is_safe', and the appropriate source category column.
      - Dedup uses a normalization key focusing on textual equivalence.
    """
    assert "response" in df.columns, "Column 'response' is required"
    out = df.copy()

    # cleaning
    out["response"] = out["response"].map(_clean_once)

    # length filter
    out["response_length"] = out["response"].fillna("").astype(str).str.len()
    out = out[(out["response_length"] >= min_len) & (out["response_length"] <= max_len)].reset_index(drop=True)

    # category unification
    if mapping_scheme == "bt":
        mapping = BT_UNIFIED_MAP
        src_col = "category"
    elif mapping_scheme == "rlhf":
        mapping = RLHF_UNIFIED_MAP
        src_col = "harm_category"
    else:
        raise ValueError("mapping_scheme must be 'bt' or 'rlhf'")

    # ensure is_safe is present and boolean
    if "is_safe" not in out.columns:
        raise ValueError("Column 'is_safe' is required")
    isb = out["is_safe"].astype("boolean")

    # parse source categories and create unified dict per row
    cats = out[src_col].apply(_parse_dict_cell)
    out["harm_category"] = [_unify_row(cats.iloc[i], bool(isb.iloc[i]), mapping) for i in range(len(out))]

    # deduplicate by normalized response (after cleaning in this pipeline)
    if drop_duplicates:
        key = out["response"].map(_norm_key)
        keep_idx = (~key.duplicated(keep="first"))
        out = out.loc[keep_idx].reset_index(drop=True)

    # sequential id 0..N-1
    out.insert(0, "id", np.arange(len(out)))

    # keep only required columns in fixed order; fill if missing
    keep = ["id", "prompt", "response", "is_safe", "harm_category", "response_length"]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan if c != "harm_category" else [{} for _ in range(len(out))]
    out = out[keep]

    return out

