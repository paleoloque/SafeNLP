"""
feature_extraction.py

Feature extractor for English texts using spaCy, profanity checks, and simple
heuristics. Produces count- and ratio-type features (POS distributions, modal usage,
imperatives, profanity, negation, and NER counts). Can return dense numpy arrays
or CSR sparse matrices. Includes utilities to summarize features by class.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
import time
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy import sparse
import spacy
from spacy.language import Language
from better_profanity import profanity


def _as_text(x):
    """Coerce any value to a safe string; NaNs become empty strings."""
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def _now():
    """Current timestamp (YYYY-MM-DD HH:MM:SS) for progress messages."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_div(a, b):
    """Division with zero protection; returns 0.0 if denominator is 0."""
    return (a / b) if b else 0.0


@dataclass
class PipelineArtifacts:
    """Container for constructed pipeline components."""
    engineered: "UnifiedEngineeredFeatures"


class UnifiedEngineeredFeatures(BaseEstimator, TransformerMixin):
    """
    Extract engineered linguistic features from text with spaCy.

    Features:
      - "pos": POS ratios for Universal POS tags (per-token ratios in [0,1])
      - "modal": Ratio of modal verbs (Penn tag MD) to tokens
      - "ratios": noun_verb_ratio and aux_verb_ratio (bounded in [0,1])
      - "imperative": imperative sentence count + ratio per document
      - "profanity": token-level profanity count + ratio
      - "negation": dependency-based negation count + ratio
      - "ner": per-entity-label counts (optionally filtered by `include_entity_labels`)

    Parameters:
    text_col : str
        Column name with raw text.
    spacy_model : str
        spaCy model to load (e.g., "en_core_web_sm").
    features : list[str] | None
        Feature groups to extract; defaults to all available.
    include_entity_labels : list[str] | None
        If set, only these NER labels are counted.
    n_process : int
        spaCy pipe n_process.
    batch_size : int
        spaCy pipe batch size.
    progress : bool
        Show tqdm progress bar.
    return_sparse : bool
        Return csr_matrix instead of dense numpy array.
    """

    def __init__(
        self,
        text_col = "response",
        spacy_model = "en_core_web_sm",
        features = None,
        include_entity_labels = None,
        n_process = 2,
        batch_size = 1024,
        progress = True,
        return_sparse = False
    ):
        self.text_col = text_col
        self.spacy_model = spacy_model
        self.features = features or [
            "pos","modal","ratios","imperative","profanity","negation","ner"
        ]
        self.include_entity_labels = include_entity_labels
        self.n_process = n_process
        self.batch_size = batch_size
        self.progress = progress
        self.return_sparse = return_sparse

        # Universal POS tags (used to allocate stable output schema)
        self._upos = [
            "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM",
            "PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"
        ]
        self._count_cols = []
        self._ratio_cols = []

    def fit(self, X, y=None):
        """Load spaCy pipeline, prepare profanity list, and freeze feature schema."""
        self._nlp: Language = spacy.load(self.spacy_model)
        if "profanity" in self.features:
            profanity.load_censor_words()

        # Discover NER labels from the model (optionally filter)
        if "ner" in self.features and "ner" in self._nlp.pipe_names:
            ner_labels = list(self._nlp.get_pipe("ner").labels)
        else:
            ner_labels = []
        if self.include_entity_labels is not None:
            allow = set(self.include_entity_labels)
            ner_labels = [l for l in ner_labels if l in allow]
        self.entity_labels_ = sorted(ner_labels)

        names = []
        count_cols = []
        ratio_cols = []

        if "pos" in self.features:
            pos_names = [f"pos_ratio_{p}" for p in self._upos]
            names += pos_names
            ratio_cols += pos_names  # POS features are given as ratios (per sentence)

        if "modal" in self.features:
            names += ["modal_ratio"]
            ratio_cols += ["modal_ratio"]

        if "ratios" in self.features:
            names += ["noun_verb_ratio","aux_verb_ratio"]
            ratio_cols += ["noun_verb_ratio","aux_verb_ratio"]

        if "imperative" in self.features:
            names += ["imperative_count","imperative_ratio"]
            count_cols += ["imperative_count"]
            ratio_cols += ["imperative_ratio"]

        if "profanity" in self.features:
            names += ["profanity_count","profanity_ratio"]
            count_cols += ["profanity_count"]
            ratio_cols += ["profanity_ratio"]

        if "negation" in self.features:
            names += ["neg_count","neg_ratio"]
            count_cols += ["neg_count"]
            ratio_cols += ["neg_ratio"]

        if "ner" in self.features:
            ent_names = [f"ent_{lbl}_count" for lbl in self.entity_labels_]
            names += ent_names
            count_cols += ent_names

        self.feature_names_ = np.array(names, dtype=object)
        self._count_cols = count_cols
        self._ratio_cols = ratio_cols
        return self

    def transform(self, X):
        """Compute features for each text and return an ndarray or CSR matrix."""
        texts = X[self.text_col].map(_as_text).tolist()
        n = len(texts)

        nlp = self._nlp
        iterator = nlp.pipe(
            texts,
            n_process=self.n_process,
            batch_size=self.batch_size,
            disable=[],
        )
        if self.progress:
            iterator = tqdm(iterator, total=n, desc=f"{_now()} Getting the features...", unit="doc")

        rows = []
        for doc in iterator:
            feats = {}
            tokens = [t for t in doc if not t.is_space]
            n_tokens = len(tokens)

            # POS ratios per Universal POS tag
            if "pos" in self.features:
                pos_counts = {p: 0 for p in self._upos}
                for t in tokens:
                    pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1
                for p in self._upos:
                    feats[f"pos_ratio_{p}"] = _safe_div(pos_counts[p], n_tokens)  # [0,1]

            # Modal ratio
            if "modal" in self.features:
                modal_count = sum(1 for t in tokens if t.tag_ == "MD")
                feats["modal_ratio"] = _safe_div(modal_count, n_tokens)  # [0,1]

            # Ratios: share of NOUN among (NOUN+VERB); share of AUX among (AUX+VERB)
            if "ratios" in self.features:
                verb_cnt = sum(1 for t in tokens if t.pos_ == "VERB")
                noun_cnt = sum(1 for t in tokens if t.pos_ == "NOUN")
                aux_cnt  = sum(1 for t in tokens if t.pos_ == "AUX")
                feats["noun_verb_ratio"] = _safe_div(noun_cnt, (noun_cnt + verb_cnt))  # [0,1]
                feats["aux_verb_ratio"]  = _safe_div(aux_cnt, (aux_cnt + verb_cnt))    # [0,1]

            # Imperative: heuristics via ROOT tag VB without an explicit subject
            if "imperative" in self.features:
                imp_count = 0
                sents = list(doc.sents)
                n_sents = max(1, len(sents))
                SUBJECT_DEPS = {"nsubj","nsubjpass","csubj","csubjpass","expl"}
                for sent in sents:
                    roots = [t for t in sent if t.dep_ == "ROOT"]
                    if not roots:
                        continue
                    root = roots[0]
                    has_subject = any(ch.dep_ in SUBJECT_DEPS for ch in root.children)
                    if (root.pos_ in {"VERB","AUX"}) and (root.tag_ == "VB") and (not has_subject):
                        imp_count += 1
                feats["imperative_count"] = float(imp_count)              
                feats["imperative_ratio"] = _safe_div(imp_count, n_sents)  # [0,1]

            # Profanity: per-token check (simple word-level detection)
            if "profanity" in self.features:
                tokens_text = [t.text for t in tokens]
                c = sum(1 for w in tokens_text if profanity.contains_profanity(w))
                feats["profanity_count"] = float(c)                        
                feats["profanity_ratio"] = _safe_div(c, len(tokens_text))  # [0,1]

            # Negation: dependency-based 'neg' occurrences
            if "negation" in self.features:
                negs = sum(1 for t in tokens if t.dep_ == "neg")
                feats["neg_count"] = float(negs)                           
                feats["neg_ratio"] = _safe_div(negs, n_tokens)             # [0,1]

            # NER counts per selected label (kept as counts; may binarize below)
            if "ner" in self.features and self.entity_labels_:
                ent_counts = {f"ent_{lbl}_count": 0 for lbl in self.entity_labels_}
                for ent in doc.ents:
                    key = f"ent_{ent.label_}_count"
                    if key in ent_counts:
                        ent_counts[key] += 1
                feats.update({k: float(v) for k, v in ent_counts.items()})
            rows.append(feats)

        df = pd.DataFrame(rows, columns=self.get_feature_names_out())

        # Final safety: replace inf/-inf with NaN, then fill NaNs with 0.0
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return sparse.csr_matrix(df.values) if self.return_sparse else df.values

    def get_feature_names_out(self) -> np.ndarray:
        """Return the frozen list of output feature names."""
        return getattr(self, "feature_names_", np.array([], dtype=object))
#
# Running a pipeline
#
def build_pipeline(
    text_col = "response",
    spacy_model = "en_core_web_sm",
    features = None,
    include_entity_labels = None,
    n_process = 2,
    batch_size = 256,
    return_sparse_engineered = False,
    progress = True,
):
    """Factory: build and wrap the engineered-features extractor."""
    engineered = UnifiedEngineeredFeatures(
        text_col=text_col,
        spacy_model=spacy_model,
        features=features,
        include_entity_labels=include_entity_labels,
        n_process=n_process,
        batch_size=batch_size,
        progress=progress,
        return_sparse=return_sparse_engineered,
    )
    return PipelineArtifacts(engineered=engineered)

#
# Building a summary
#
def nonzero_share(s):
    """Share of non-zero values among non-missing entries."""
    m = s.notna()
    return (s.ne(0) & m).mean()

nonzero_share.__name__  = "nonzero_share"


def summarize_engineered_by_class(
    arts,
    df,
    text_col,
    label_col,
    precomputed,
):
    """
    Compute per-class summary statistics (mean/non_zero_share) for engineered features.

    If `precomputed` (feature matrix) is provided, spaCy is not re-run.
    Otherwise, the function fits and transforms using `arts.engineered`.

    """
    if precomputed is None:
        arts.engineered.fit(df)
        mat = arts.engineered.transform(df)
        cols = arts.engineered.get_feature_names_out()
    else:
        mat = precomputed
        cols = arts.engineered.get_feature_names_out()
        if len(cols) == 0:
            cols = [f"f{i}" for i in range(mat.shape[1])]

    eng_df = pd.DataFrame(mat, columns=cols, index=df.index)
    eng_df[label_col] = df[label_col].values

    keep = [c for c in eng_df.columns if c != label_col]
    summary = (
        eng_df.groupby(label_col)[keep]
              .agg(["mean", nonzero_share])
              .copy()
    )
    summary.columns = [f"{feat}__{stat}" for feat, stat in summary.columns]
    return summary.sort_index(axis=1)
