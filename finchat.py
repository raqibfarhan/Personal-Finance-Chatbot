# finchat.py
# Optimized backend for Finance CSV Chatbot (Groq + FAISS + Streamlit)
# Supports batch embeddings, caching, and large CSVs (~100k+ rows)

from __future__ import annotations
import os
import json
import re
import pickle
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Groq LLM (optional)
try:
    from groq import Groq
except Exception:
    Groq = None

# ---------------------- Data Loading ----------------------

NUMERIC_GUESS_COLS = {"amount", "value", "price", "debit", "credit", "balance"}
DATE_GUESS_COLS = {"date", "datetime", "timestamp", "time"}

def load_csv(file_like) -> pd.DataFrame:
    if isinstance(file_like, (str, os.PathLike)):
        df = pd.read_csv(file_like)
    else:
        df = pd.read_csv(file_like)

    df.columns = [c.strip().replace("\n", " ") for c in df.columns]

    for col in df.columns:
        low = col.lower()
        if any(k in low for k in DATE_GUESS_COLS):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if any(k in low for k in NUMERIC_GUESS_COLS):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df

def ensure_datetime(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if col.lower() in DATE_GUESS_COLS:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
                    return col
            except Exception:
                pass
    return None

# ---------------------- Text Corpus ----------------------

def build_row_text(row: pd.Series, text_cols: List[str]) -> str:
    parts = []
    for c in text_cols:
        v = row.get(c, "")
        if pd.isna(v):
            continue
        parts.append(f"{c}: {v}")
    return " | ".join(parts)

def make_corpus(df: pd.DataFrame, text_cols: Optional[List[str]] = None, max_chars: int = 500) -> List[str]:
    if not text_cols:
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        if not text_cols:
            text_cols = list(df.columns)
    corpus = []
    for _, row in df.iterrows():
        txt = build_row_text(row, text_cols)
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "…"
        corpus.append(txt)
    return corpus

# ---------------------- Embeddings + FAISS ----------------------

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        """
        Encode texts in batches for faster processing with large CSVs.
        """
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = self.model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            embeddings_list.append(batch_emb.astype("float32"))
        return np.vstack(embeddings_list)


def build_vector_store(df: pd.DataFrame, embedder: Embedder, text_cols: Optional[List[str]] = None, cache_path: str = "embed_cache.pkl"):
    """
    Build FAISS index from dataframe with batch embeddings and optional caching.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            index, embeddings = pickle.load(f)
        print("Loaded cached embeddings.")
    else:
        corpus = make_corpus(df, text_cols=text_cols)
        embeddings = embedder.encode(corpus)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        with open(cache_path, "wb") as f:
            pickle.dump((index, embeddings), f)
        print("Embeddings built and cached.")

    return index, embeddings, df

def query_index(index: faiss.IndexFlatIP, embeddings: np.ndarray, df: pd.DataFrame, query: str, embedder: Embedder, k: int = 6) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    qvec = embedder.encode([query])
    scores, idxs = index.search(qvec, k)
    idxs = idxs[0]
    scores = scores[0]
    rows = df.iloc[idxs].copy()
    texts = [make_row_text(df.iloc[i]) for i in idxs]
    return rows, texts, scores

def make_row_text(row: pd.Series) -> str:
    return " | ".join(f"{c}: {row[c]}" for c in row.index if pd.notna(row[c]))

# ---------------------- Intent Detection ----------------------

INTENT_RULES: Dict[str, List[str]] = {
    "plot": ["plot", "graph", "chart", "visualize", "trend", "time series"],
    "calc": ["sum", "total", "average", "avg", "mean", "min", "max", "count", "median"],
    "table": ["show rows", "list", "table", "records", "top", "filter", "where"],
}

def detect_intent(query: str) -> str:
    q = query.lower()
    for intent, keys in INTENT_RULES.items():
        if any(k in q for k in keys):
            return intent
    return "qa"

# ---------------------- Calculations ----------------------

AGG_ALIASES = {
    "sum": "sum", "total": "sum",
    "average": "mean", "avg": "mean", "mean": "mean",
    "min": "min", "max": "max", "median": "median", "count": "count"
}

def find_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def guess_amount_column(df: pd.DataFrame) -> Optional[str]:
    for name in ["Amount", "amount", "Value", "Debit", "Credit"]:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    nums = find_numeric_cols(df)
    return nums[0] if nums else None

def build_timeseries(df: pd.DataFrame, amount_col: Optional[str] = None, date_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    date_col = date_col or ensure_datetime(df)
    if not date_col:
        return None
    amount_col = amount_col or guess_amount_column(df)
    if not amount_col:
        return None
    ts = (
        df.dropna(subset=[date_col])
          .groupby(pd.Grouper(key=date_col, freq="M"))[amount_col]
          .sum()
          .reset_index()
          .rename(columns={amount_col: "Amount"})
    )
    return ts

# ---------------------- Groq LLM ----------------------

GROQ_DEFAULT_MODEL = "llama-3.1-8b-instant"

def groq_client() -> Optional[Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return None
    return Groq(api_key=api_key)

SYS_PROMPT = """You are a finance CSV assistant.
Only answer using the supplied context snippets. If the user asks outside the CSV, say you don't have that data.
Be concise. Include short reasoning when relevant.
"""

def answer_with_llm(question: str, context_chunks: List[str], model: str = GROQ_DEFAULT_MODEL, temperature: float = 0.1) -> str:
    client = groq_client()
    if client is None:
        return "LLM is not configured. Please set GROQ_API_KEY."
    context = "\n\n---\n".join(context_chunks)
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# --------------------------- Calculation Helpers ---------------------------

def parse_simple_calc(query: str, df):
    query = query.lower()
    recipe = {"op": None, "column": "Amount", "filter": None}

    if "total" in query or "sum" in query:
        recipe["op"] = "sum"
    elif "average" in query or "mean" in query or "avg" in query:
        recipe["op"] = "mean"
    elif "max" in query or "maximum" in query:
        recipe["op"] = "max"
    elif "min" in query or "minimum" in query:
        recipe["op"] = "min"
    elif "count" in query or "how many" in query:
        recipe["op"] = "count"

    # Try to detect vendor/merchant filter (like 'amazon', 'walmart')
    if "Description" in df.columns:
        for merchant in df["Description"].dropna().unique():
            if str(merchant).lower() in query:
                recipe["filter"] = merchant
                break

    return recipe

def run_calc(df, recipe):
    if not recipe or "op" not in recipe:
        return {"error": "Invalid recipe"}

    op = recipe["op"]
    column = recipe.get("column", "Amount")
    filt = recipe.get("filter", None)

    if filt:
        try:
            df = df[df.astype(str).apply(lambda row: row.str.contains(str(filt), case=False, na=False)).any(axis=1)]
        except Exception as e:
            return {"error": f"Filter failed: {e}"}

    if op == "sum":
        value = df[column].sum()
    elif op == "mean":
        value = df[column].mean()
    elif op == "count":
        value = df[column].count()
    elif op == "max":
        value = df[column].max()
    elif op == "min":
        value = df[column].min()
    else:
        return {"error": f"Unknown operation: {op}"}

    return {"ok": True, "agg": op, "target": column, "n_rows": len(df), "value": float(value)}

# ---------------------- Query Orchestrator ----------------------

def handle_query(
    query: str,
    df: pd.DataFrame,
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    embedder: Embedder,
    k: int = 6,
    llm_model: str = GROQ_DEFAULT_MODEL,
    use_llm: bool = True
) -> dict:

    intent = detect_intent(query)
    result: Dict[str, Any] = {"intent": intent}

    # Retrieve top-K rows for context
    rows, texts, scores = query_index(index, embeddings, df, query, embedder, k=k)
    result["retrieved_rows"] = rows
    result["retrieved_texts"] = texts
    result["retrieved_scores"] = scores

    # Calculation intent
    if intent == "calc":
        recipe = parse_simple_calc(query, df)
        calc = run_calc(df, recipe)   # ✅ run on FULL dataframe
        result["calc"] = calc

        if use_llm:
            ctx = texts[:3]  # still only pass top rows for context
            calc_text = json.dumps(calc, indent=2)
            llm_q = f"User asked: {query}\nComputed result JSON: {calc_text}\nExplain briefly."
            result["llm_answer"] = answer_with_llm(llm_q, ctx, model=llm_model)
        return result

    # Plot intent
    if intent == "plot":
        ts = build_timeseries(df)
        result["timeseries"] = ts
        if use_llm:
            result["llm_answer"] = answer_with_llm(
                f"User asked to visualize: {query}\nDescribe trends in the time series.",
                texts[:4],
                model=llm_model
            )
        return result

    # Table intent
    if intent == "table":
        result["table"] = rows
        if use_llm:
            result["llm_answer"] = answer_with_llm(
                f"User asked: {query}\nSummarize the relevant rows.",
                texts[:6],
                model=llm_model
            )
        return result

    # Default: QA
    if use_llm:
        result["llm_answer"] = answer_with_llm(query, texts[:8], model=llm_model)
    else:
        result["llm_answer"] = "LLM disabled."
    return result
