"""
model_pipeline.py

End-to-end training and inference pipeline for text classification with LightGBM.
Combines Word2Vec embeddings with engineered linguistic features (POS, modal verbs,
ratios, imperatives, profanity, negation, NER). Supports model training, evaluation,
artifact saving, and single/multiple text prediction.
"""

import re, os, numpy as np, pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
from scipy import sparse
import lightgbm as lgb
import joblib
from feature_extraction import UnifiedEngineeredFeatures
from typing import Union

#
# General Settings
#
TEXT_COL = "response"   # text column
LABEL_COL = "is_safe"   # label column (0/1)
W2V_DIM  = 300
W2V_EPOCHS = 5

#
# Tokenization
#
def simple_tok(s: str):
    s = (s or "").lower()
    return [t for t in re.split(r"[^a-z0-9_]+", s) if t]

def df_to_tokens(df, text_col):
    return [simple_tok(x) for x in df[text_col].astype(str).tolist()]

#
# Document embeddings
#
def doc_pool(tokens, wv, mode="mean"):
    if not tokens: return np.zeros(wv.vector_size, dtype=np.float32)
    vecs = [wv[t] for t in tokens if t in wv.key_to_index]
    if not vecs: return np.zeros(wv.vector_size, dtype=np.float32)
    M = np.vstack(vecs)
    if mode == "mean": return M.mean(axis=0)
    if mode == "max":  return M.max(axis=0)
    if mode == "min":  return M.min(axis=0)
    if mode == "std":  return M.std(axis=0)
    return M.mean(axis=0)

def docs_to_matrix(tokens_list, wv, modes=("mean","max","min","std")):
    parts = []
    for mode in modes:
        arr = np.vstack([doc_pool(toks, wv, mode) for toks in tokens_list]).astype(np.float32)
        parts.append(arr)
    return np.hstack(parts).astype(np.float32)

#
# Loading features from files (if available)
#
def _load_engineered(eng_src: Union[None, str, Path, pd.DataFrame, np.ndarray, sparse.spmatrix],
                     ref_df: pd.DataFrame,
                     expected_n: int) -> sparse.csr_matrix:
    """Load engineered features (from file, DF, ndarray, sparse) and align with reference df."""
    if eng_src is None:
        return sparse.csr_matrix((expected_n, 0), dtype=np.float32)

    if sparse.issparse(eng_src):
        return eng_src.tocsr()

    if isinstance(eng_src, np.ndarray):
        if eng_src.shape[0] != expected_n:
            raise ValueError(f"Rows mismatch: engineered={eng_src.shape[0]} vs expected={expected_n}. "
                             "Add 'id' to both datasets for reliable alignment.")
        return sparse.csr_matrix(eng_src.astype(np.float32, copy=False))

    if isinstance(eng_src, (str, Path)):
        p = Path(eng_src)
        if p.suffix.lower() in {".csv", ".tsv"}:
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported features file: {p.suffix}")
    elif isinstance(eng_src, pd.DataFrame):
        df = eng_src.copy()
    else:
        raise TypeError("Unsupported eng_src type")

    id_col = next((c for c in df.columns if c.lower() in {"id","row_id","sample_id"}), None)
    ref_id_col = next((c for c in ref_df.columns if c.lower() in {"id","row_id","sample_id"}), None)

    if id_col and ref_id_col:
        df = df.set_index(id_col)
        aligned = df.reindex(ref_df[ref_id_col])
        if aligned.isnull().any().any():
            missing = int(aligned.isnull().any(axis=1).sum())
            raise ValueError(f"Engineered features missing for {missing} ids (id column='{ref_id_col}').")
        feat_df = aligned
    else:
        if id_col:
            df = df.drop(columns=[id_col])
        if len(df) != expected_n:
            raise ValueError(f"Rows mismatch: engineered={len(df)} vs expected={expected_n}. "
                             "Add 'id' column to both datasets for reliable alignment.")
        feat_df = df

    feat_df = feat_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sparse.csr_matrix(feat_df.to_numpy(dtype=np.float32, copy=False))

#
# Finding best threshold
#
def best_f1_threshold(y_true, proba, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 81)
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

#
# LightGBM training
#
def train_lightgbm(X_tr, y_tr, X_va, y_va):
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)

    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.03,
        num_leaves=255,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        seed=42,
        n_jobs=-1,
        verbose=-1,
    )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train","valid"]
    )

    proba_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    auc = roc_auc_score(y_va, proba_va)
    thr, f1 = best_f1_threshold(y_va, proba_va)
    pred_va = (proba_va >= thr).astype(int)
    acc = accuracy_score(y_va, pred_va)

    print(f"\nBest iter: {booster.best_iteration}")
    print(f"AUC:      {auc:.4f}")
    print(f"F1@best:  {f1:.4f}  (thr={thr:.3f})")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_va, pred_va, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_va, pred_va))
    return booster, thr

#
# Main pipeline
#
def run_pipeline(train_df, val_df, X_eng_tr=None, X_eng_va=None, artifacts_dir="artifacts"):
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    X_eng_tr = _load_engineered(X_eng_tr, train_df, len(train_df))
    X_eng_va = _load_engineered(X_eng_va, val_df,   len(val_df))

    tokens_tr = df_to_tokens(train_df, TEXT_COL)
    tokens_va = df_to_tokens(val_df, TEXT_COL)
    print("Training Word2Vec...")
    model = Word2Vec(sentences=tokens_tr, vector_size=W2V_DIM, window=5,
                     min_count=3, sg=1, epochs=W2V_EPOCHS, workers=os.cpu_count())
    wv = model.wv
    print(f"Done. vocab={len(wv)}, dim={wv.vector_size}")

    W2V_tr = docs_to_matrix(tokens_tr, wv)
    W2V_va = docs_to_matrix(tokens_va, wv)
    print("W2V shapes:", W2V_tr.shape, W2V_va.shape)

    X_tr = sparse.hstack([X_eng_tr, sparse.csr_matrix(W2V_tr)], format="csr")
    X_va = sparse.hstack([X_eng_va, sparse.csr_matrix(W2V_va)], format="csr")

    y_tr = train_df[LABEL_COL].astype(int).values
    y_va = val_df[LABEL_COL].astype(int).values

    booster, thr = train_lightgbm(X_tr, y_tr, X_va, y_va)

    print("Saving artifacts...")
    model.save(os.path.join(artifacts_dir, "w2v.model"))
    joblib.dump(booster, os.path.join(artifacts_dir, "lgbm_model.joblib"))
    with open(os.path.join(artifacts_dir, "threshold.txt"), "w") as f:
        f.write(str(thr))
    return booster, thr, wv
#
# Inference testing (single/multiple predictions)
#
POOL_MODES = ("mean","max","min","std")

eng = UnifiedEngineeredFeatures(
        text_col=TEXT_COL,
        spacy_model="en_core_web_sm",
        features=["pos","modal","ratios","imperative","profanity","negation","ner"],
        n_process=2,
        batch_size=1024,
        return_sparse=False,
        progress=False
    )
eng.fit(pd.DataFrame({TEXT_COL: [""]}))

def predict_one(text, booster, wv, eng=eng, thr = 0.5):
    """
    Predict class for a single text.
    """
    tokens = [simple_tok(text)]
    W2V = docs_to_matrix(tokens, wv, modes=POOL_MODES)
    df_tmp = pd.DataFrame({TEXT_COL: [text]})
    X_eng = eng.transform(df_tmp)
    X = sparse.hstack([X_eng, sparse.csr_matrix(W2V)], format="csr")

    proba = float(booster.predict(X, num_iteration=booster.best_iteration)[0])
    pred_bool = proba >= thr
    pred_int  = int(pred_bool)
    label     = "Is safe: True" if pred_bool else "Is safe: False"
    return pred_bool, pred_int, label, proba

def predict_many(texts, booster, wv, eng=eng, thr=0.5):
    """
    Predict classes and probabilities for multiple texts.
    """
    tokens = [simple_tok(t) for t in texts]
    W2V = docs_to_matrix(tokens, wv, modes=POOL_MODES)
    if eng is not None:
        df_tmp = pd.DataFrame({TEXT_COL: texts})
        X_eng = eng.transform(df_tmp)
    else:
        X_eng = sparse.csr_matrix((len(texts), 0), dtype=np.float32)

    X = sparse.hstack([X_eng, sparse.csr_matrix(W2V)], format="csr")
    probas = booster.predict(X, num_iteration=getattr(booster, "best_iteration", None))
    pred_bools = (probas >= thr)
    pred_ints = pred_bools.astype(int)
    labels = ["Is safe: True" if b else "Is safe: False" for b in pred_bools]

    return list(map(bool, pred_bools)), pred_ints, labels, probas
