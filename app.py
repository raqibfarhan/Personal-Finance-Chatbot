# ap# app.py
import os
import pandas as pd
import streamlit as st
import plotly.express as px

from finchat import (
    load_csv, ensure_datetime,
    Embedder, build_vector_store, query_index, handle_query,
    guess_amount_column, build_timeseries,
    GROQ_DEFAULT_MODEL
)

st.set_page_config(page_title="Finance Chatbot", page_icon="💬", layout="wide")
st.title("💬 Finance Chatbot ")
st.caption("Upload a CSV, ask questions, get calculations and charts. Uses FAISS for retrieval and Groq LLM for answers.")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input("GROQ_API_KEY (optional)", type="password", help="Set to enable LLM answers.")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    embed_model_name = st.selectbox(
        "Embedding model",
        ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2"],
        index=0
    )
    k_retrieval = st.slider("Top-K retrieval", 3, 20, 8)
    use_llm = st.checkbox("Use Groq LLM", value=True)
    llm_model = st.text_input("Groq Chat Model", value=GROQ_DEFAULT_MODEL)
    st.markdown("---")
    st.caption("Tip: Keep numeric column named `Amount` if possible for best auto-calculation.")

# --- File Upload ---
file = st.file_uploader("Upload your finance CSV", type=["csv"])
if file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# --- Load CSV & detect date column ---
with st.spinner("Loading CSV…"):
    df = load_csv(file)
    date_col = ensure_datetime(df)

st.success(f"Loaded {len(df):,} rows. Date column: `{date_col}`" if date_col else f"Loaded {len(df):,} rows.")

with st.expander("Preview Data", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

# --- Build embeddings & FAISS index (cached) ---
@st.cache_resource(show_spinner=False)
def load_index(dataframe, model_name):
    embedder = Embedder(model_name=model_name)
    index, embeddings, df_processed = build_vector_store(dataframe, embedder)
    return embedder, index, embeddings, df_processed

# Non-blocking spinner + input
with st.spinner("Indexing for similarity search… this may take a moment"):
    embedder, index, embeddings, df = load_index(df, embed_model_name)


# --- Query Input ---
st.subheader("Ask a question")
default_q = "What is the total Amount for groceries in 2024?"
query = st.text_input("Type your question", value=default_q, placeholder="e.g., plot monthly expenses, total spend at Amazon…")
go = st.button("Ask", type="primary")
if not go and query.strip() == "":
    st.stop()

if go:
    with st.spinner("🤖 Thinking…"):
        result = handle_query(
            query=query,
            df=df,
            index=index,
            embeddings=embeddings,
            embedder=embedder,
            k=k_retrieval,
            llm_model=llm_model,
            use_llm=use_llm
        )

    # --- Results ---
    colA, colB = st.columns([2, 1])

    # Column A: LLM, Calc, Table, Plot
    with colA:
        st.markdown(f"**Intent detected:** `{result['intent']}`")
        if "llm_answer" in result and result["llm_answer"]:
            st.markdown("#### 🧠 Answer")
            st.write(result["llm_answer"])

        # Calculations
        if result["intent"] == "calc":
            calc = result.get("calc", {})
            if calc and calc.get("ok"):
                st.markdown("#### 🧮 Calculation")
                st.write(f"**{calc['agg'].upper()}** of **{calc['target']}** on {calc['n_rows']} rows = **{calc['value']:.2f}**")
            elif calc and not calc.get("ok"):
                st.error(calc.get("error", "Could not compute."))

        # Table intent
        if result["intent"] == "table":
            st.markdown("#### 📄 Matching Rows")
            st.dataframe(result["table"], use_container_width=True, height=360)

        # Plot intent
        if result["intent"] == "plot":
            ts = result.get("timeseries")
            if ts is None or ts.empty:
                amt_col = guess_amount_column(df)
                if amt_col:
                    st.warning("No date column found. Showing top totals by Category instead.")
                    if "Category" in df.columns:
                        topcat = df.groupby("Category")[amt_col].sum().reset_index().sort_values(amt_col, ascending=False).head(15)
                        fig = px.bar(topcat, x="Category", y=amt_col, title="Top Categories by Total")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No `Category` column to aggregate.")
                else:
                    st.info("No numeric column to plot.")
            else:
                fig = px.line(ts, x=ts.columns[0], y="Amount", title="Monthly Total Amount")
                st.plotly_chart(fig, use_container_width=True)

    # Column B: Retrieved Context
    with colB:
        st.markdown("#### 🔎 Retrieved Context")
        top_k = len(result["retrieved_texts"])
        for i in range(top_k):
            score = float(result["retrieved_scores"][i])
            with st.expander(f"Row {i+1} (score {score:.3f})"):
                st.code(result["retrieved_texts"][i])
                st.dataframe(result["retrieved_rows"].iloc[[i]], use_container_width=True)

    st.markdown("---")
    st.caption("This app uses FAISS for similarity search over row text, a SentenceTransformer for embeddings, "
               "and Groq for LLM answers constrained to retrieved context.")
