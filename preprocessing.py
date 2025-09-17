"""
preprocessing.py

Utility functions for cleaning and normalizing prompt/response datasets
and for unifying harm categories across different annotation schemes,
adapted and specified to BeaverTails and SafeRLHF local challenging points.

Intended to be used with Hugging Face `datasets.Dataset` / `DatasetDict`.
"""

import re
import pandas as pd
from datasets import Dataset, DatasetDict

# ----------------------------------------------------------------------
# Deduplication (exact duplicates by `response`)
# ----------------------------------------------------------------------

def dedup_by_response(ds: Dataset, col: str = "response", keep: str = "first") -> Dataset:
    """
    Drop exact duplicate rows by `col` (default: 'response'), keeping the first.
    No extra checks. Exact match (case/spacing preserved).
    """
    df = ds.to_pandas()
    df = df.drop_duplicates(subset=col, keep=keep).reset_index(drop=True)
    return Dataset.from_pandas(df, preserve_index=False)

def dedup(ds_or_dd, col: str = "response", keep: str = "first"):
    """
    Dedup for Dataset or DatasetDict (per split).
    """
    if isinstance(ds_or_dd, DatasetDict):
        return DatasetDict({name: dedup_by_response(split, col=col, keep=keep)
                            for name, split in ds_or_dd.items()})
    else:
        return dedup_by_response(ds_or_dd, col=col, keep=keep)

# ----------------------------------------------------------------------
# Regular Expressions for Cleaning
# ----------------------------------------------------------------------

# Detects generic "end of output/response" markers (e.g. "EOM", "End of Response")
EOM_GENERIC_RE = re.compile(
    r"""
    ^\s*                       
    [^\w\s]{0,6}\s*            
    (?:eom|end)\b              
    (?:\s+of\s+(?:output|response))? 
    [^\n]*$                   
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

# Detects triple quotes (possible truncation markers)
TRIPLEQ_RE = re.compile(r'"""')

# Detects "Instruction:" headings with optional numbering
INSTR_RE = re.compile(r"\b\d+\s*\.?\s*Instruction\s*:", re.IGNORECASE)

# Emoji ranges (various Unicode blocks)
EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F" 
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F1E0-\U0001F1FF" 
    "\U00002700-\U000027BF"  
    "\U0001F900-\U0001F9FF"  
    "\U00002600-\U000026FF"  
    "]+",
    re.UNICODE,
)


# Personal information masking
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b")

# URL detectors
URL_PROTO_RE  = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
URL_WWW_RE    = re.compile(r"(?<!@)\bwww\.[^\s)>\]]+", re.IGNORECASE)
URL_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?-i:[a-z]{2,})(?:/[^\s)>\]]*)?", re.IGNORECASE)

# ----------------------------------------------------------------------
# Category Mapping Dictionaries (for both BeaverTails and SafeRLHF)
# ----------------------------------------------------------------------

BT_UNIFIED_MAP = {
    "Violence and Threats": [
        "violence,aiding_and_abetting,incitement",
        "child_abuse",
        "animal_abuse",
        "self_harm",
    ],
    "Crime and Illegal Activities": [
        "financial_crime,property_crime,theft",
        "drug_abuse,weapons,banned_substance",
        "terrorism,organized_crime",
    ],
    "Hate Speech and Discrimination": [
        "hate_speech,offensive_language",
        "discrimination,stereotype,injustice",
    ],
    "Sexual Content": ["sexually_explicit,adult_content"],
    "Privacy Violations": ["privacy_violation"],
    "Misinformation and Manipulation": [
        "misinformation_regarding_ethics,laws_and_safety",
        "non_violent_unethical_behavior",
        "controversial_topics,politics",
    ],
}

RLHF_UNIFIED_MAP = {
    "Violence and Threats": [
        "Endangering National Security",
        "Violence",
        "Human Trafficking",
        "Physical Harm",
        "Animal Abuse",
    ],
    "Crime and Illegal Activities": [
        "Drugs",
        "Economic Crime",
        "Cybercrime",
        "White-Collar Crime",
        "Disrupting Public Order",
        "Environmental Damage",
    ],
    "Hate Speech and Discrimination": ["Insulting Behavior", "Discriminatory Behavior"],
    "Sexual Content": ["Sexual Content"],
    "Privacy Violations": ["Privacy Violation"],
    "Misinformation and Manipulation": [
        "Mental Manipulation",
        "Copyright Issues",
        "Psychological Harm",
        "Endangering Public Health",
    ],
}

# ----------------------------------------------------------------------
# Cleaning Utilities
# ----------------------------------------------------------------------

def cut_early(text: str) -> str:
    """Cut off text at the first occurrence of an 'end of output' marker or triple quotes."""
    spots = []
    for rx in (EOM_GENERIC_RE, TRIPLEQ_RE):
        m = rx.search(text)
        if m:
            spots.append(m.start())
    return text[: min(spots)].rstrip() if spots else text


def cut_instruction(text: str) -> str:
    """Remove trailing 'Instruction:' sections if present."""
    m = INSTR_RE.search(text)
    return text[: m.start()].rstrip() if m else text


def strip_trailing_non_word(t: str) -> str:
    """
    Strip trailing non-word characters but preserve a single final period.
    Example:
        'Hello!!!'   -> 'Hello'
        'Hello....'  -> 'Hello.'
        'Hello).'    -> 'Hello.'
    """
    if re.search(r"\.[^\w]*$", t):
        return re.sub(r"[^\w]*$", "", t)
    return re.sub(r"\W+$", "", t)


def clean_once(t: str) -> str:
    """
    Apply one round of cleaning to a single text string:
      - remove end markers, instructions, emojis
      - mask emails, phones, URLs
      - drop illegal chars
      - normalize spacing
      - strip trailing punctuation
    """
    if not t:
        return ""
    t = cut_early(t)
    t = EMOJI_RE.sub("", t)
    t = cut_instruction(t)
    t = EMAIL_RE.sub(" [EMAIL] ", t)
    t = PHONE_RE.sub(" [PHONE] ", t)
    for rx in (URL_PROTO_RE, URL_WWW_RE, URL_DOMAIN_RE):
        t = rx.sub(" [LINK] ", t)
    t = re.sub(r"([a-z])(\.)([A-Z])", r"\1. \3", t)
    t = strip_trailing_non_word(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_only(x: str) -> str:
    """Wrapper: apply clean_once and unwrap tuple if needed (defensive)."""
    y = clean_once(x)
    return y[0] if isinstance(y, tuple) else y


# ----------------------------------------------------------------------
# Dataset-Level Cleaning / Mapping Functions
# ----------------------------------------------------------------------

def map_clean(batch):
    """Map clean_only() across batch['response'], returning cleaned texts + their lengths."""
    cleaned = [clean_only(t) for t in batch["response"]]
    return {
        "response": cleaned,
        "response_length": [len(t) for t in cleaned],
    }


def _apply(ds: Dataset, min_len: int, max_len: int) -> Dataset:
    """Apply cleaning + length filter to a single Dataset split."""
    ds = ds.map(map_clean, batched=True, desc="clean")
    ds = ds.filter(lambda ex: min_len<= ex["response_length"] <= max_len, desc=f"{min_len} <= len <= {max_len}")
    ds = dedup_by_response(ds, col="response", keep="first")
    return ds


def apply_pipeline(ds_or_dd, min_len: int = 70, max_len: int = 2000, dedup: bool = True):
    """
    Apply the full cleaning pipeline to either a Dataset or DatasetDict.
    - Cleans text, filters by length, and (by default) drops exact duplicate `response`s.
    """
    if isinstance(ds_or_dd, DatasetDict):
        return DatasetDict({name: _apply(split, min_len, max_len, dedup) for name, split in ds_or_dd.items()})
    else:
        return _apply(ds_or_dd, min_len, max_len, dedup)

def explode_pairs(batch):
    """
    Explode preference-style records (prompt + 2 responses) into
    individual (prompt, response) rows.
    """
    prompts, responses, is_safe, categories, severity = [], [], [], [], []

    for i in range(len(batch["prompt"])):
        # response_0
        prompts.append(batch["prompt"][i])
        responses.append(batch["response_0"][i])
        is_safe.append(batch["is_response_0_safe"][i])
        categories.append(batch["response_0_harm_category"][i])
        severity.append(batch["response_0_severity_level"][i])

        # response_1
        prompts.append(batch["prompt"][i])
        responses.append(batch["response_1"][i])
        is_safe.append(batch["is_response_1_safe"][i])
        categories.append(batch["response_1_harm_category"][i])
        severity.append(batch["response_1_severity_level"][i])

    return {
        "prompt": prompts,
        "response": responses,
        "is_safe": is_safe,
        "harm_category": categories,
        "severity_level": severity,
    }

def explode_dataset(ds: Dataset) -> Dataset:
    """Explodes a dataset of paired responses into a flat dataset of single responses."""
    return ds.map(explode_pairs, batched=True, remove_columns=ds.column_names)


def unify_bt_batch(batch, mapping_list):
    """
    Unify harm_category dictionaries.
    mapping_list: dict[str, list[str]]
        Maps unified domain names to the list of source labels.
    """
    out = []
    for cats, is_safe in zip(batch["harm_category"], batch["is_safe"]):
        new_dict = {k: False for k in list(mapping_list.keys())}
        if not is_safe:
            for domain, src_list in mapping_list.items():
                new_dict[domain] = any(bool(cats.get(src, False)) for src in src_list)
        out.append(new_dict)
    return {"harm_category": out}


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__ = [
    "clean_once", "clean_only", "apply_pipeline",
    "explode_dataset", "unify_bt_batch", "dedup", "dedup_by_response",
    "BT_UNIFIED_MAP", "RLHF_UNIFIED_MAP",

]
