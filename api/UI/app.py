# api/UI/app.py
import os
import time
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8010")

st.set_page_config(page_title="Company Assistant", page_icon="🤖")
st.title("🤖 Company Assistant")

# ========== Sidebar: server status & controls ==========
with st.sidebar:
    st.subheader("Server status")

    # Health
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        st.write("Health:", "✅" if r.ok else "❌")
    except Exception as e:
        st.write("Health:", f"❌ ({e})")

    # Qdrant
    try:
        rq = requests.get(f"{API_BASE}/ping/qdrant", timeout=5)
        ok_q = rq.ok and rq.json().get("ok")
        st.write("Qdrant:", "✅" if ok_q else "❌")
    except Exception as e:
        st.write("Qdrant:", f"❌ ({e})")

    # Ollama (fast model-list ping; allow a longer timeout for first run)
    try:
        ro = requests.get(f"{API_BASE}/ping/ollama", timeout=10)
        ok_o = ro.ok and ro.json().get("ok")
        st.write("Ollama:", "✅" if ok_o else "❌")
        if ok_o:
            payload = ro.json()
            models = payload.get("models") or []
            if models:
                st.caption(" · ".join(models[:3]))
            elif payload.get("reply"):
                st.caption(payload.get("reply"))
    except Exception as e:
        st.write("Ollama:", f"❌ ({e})")

    st.divider()
    top_k = st.slider("Top-k passages", 1, 12, 4, 1)
    score_thr = st.slider("Min cosine score", 0.0, 1.0, 0.50, 0.01)
    exact_search = st.toggle("Exact search (exhaustive)", value=False)
    max_per_doc = st.slider("Max chunks per doc", 1, 5, 2, 1)

    if "latency_samples" not in st.session_state:
        st.session_state.latency_samples = []

    if st.session_state.latency_samples:
        avg = sum(st.session_state.latency_samples) / len(st.session_state.latency_samples)
        st.caption(f"Avg client RTT: **{avg:.1f} ms** (last {len(st.session_state.latency_samples)})")

    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.latency_samples = []

# ===== Optional: show server config =====
with st.expander("Server config", expanded=False):
    try:
        cfg = requests.get(f"{API_BASE}/config", timeout=5).json()
        st.json(cfg)
    except Exception as e:
        st.write(f"Couldn’t load config: {e}")

# ========== Session state ==========
if "history" not in st.session_state:
    st.session_state.history = []

# ========== Helper to call API (measures client RTT) ==========
def call_api(query: str, k: int, thr: float, exact: bool, per_doc: int) -> dict:
    try:
        c0 = time.perf_counter()
        resp = requests.post(
            f"{API_BASE}/ask",
            json={
                "query": query,
                "top_k": int(k),
                "score_threshold": float(thr),
                "exact_search": bool(exact),
                "max_per_doc": int(per_doc),
            },
            timeout=120,  # should align with server's request_timeout to Ollama
        )
        c1 = time.perf_counter()
        client_rtt_ms = round((c1 - c0) * 1000, 2)

        # Parse body
        if resp.headers.get("content-type", "").startswith("application/json"):
            data = resp.json()
        else:
            data = {"ok": False, "error": f"Non-JSON response ({resp.status_code}): {resp.text[:300]}"}

        # Attach client timing
        data.setdefault("metrics", {})
        data["metrics"]["client_rtt_ms"] = client_rtt_ms

        # Keep a rolling average in sidebar
        st.session_state.latency_samples = (st.session_state.latency_samples + [client_rtt_ms])[-50:]

        return data
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ========== Chat input & flow ==========
prompt = st.chat_input("Ask anything about our services, policies, or platform…")

if prompt:
    st.session_state.history.append({"role": "user", "text": prompt})
    with st.spinner("Thinking…"):
        data = call_api(prompt, top_k, score_thr, exact_search, max_per_doc)
    if not data.get("ok"):
        st.session_state.history.append(
            {"role": "assistant", "text": f"⚠️ {data.get('error','Unknown error')}"}
        )
    else:
        st.session_state.history.append({
            "role": "assistant",
            "text": (data.get("answer") or "").strip(),
            "sources": data.get("sources", []),
            "metrics": data.get("metrics", {}),
            "retrieved": data.get("retrieved", []),
        })

# ========== Render conversation ==========
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.get("text") or "—")

            # Sources
            if msg.get("sources"):
                st.caption("Sources")
                for s in msg["sources"]:
                    doc = s.get("doc_id", "unknown")
                    chunk = s.get("chunk", "—")
                    score = s.get("score")
                    score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
                    st.write(f"- `{doc}` (chunk {chunk} • score {score_txt})")

            # Context the LLM saw
            if msg.get("retrieved"):
                with st.expander("Context used"):
                    for i, r in enumerate(msg["retrieved"], 1):
                        doc = r.get("doc_id", "unknown")
                        chk = r.get("chunk", "—")
                        sc = r.get("score")
                        sc_txt = f"{sc:.3f}" if isinstance(sc, (int, float)) else "—"
                        st.markdown(f"**{i}. {doc}#{chk} • score {sc_txt}**")
                        st.write(r.get("text", ""))
                        st.markdown("---")

            # Metrics (includes timings)
            if msg.get("metrics"):
                with st.expander("Metrics"):
                    st.json(msg["metrics"])
                    t = msg["metrics"]
                    # Show flattened timing bullets if present
                    tm = t.get("timings_ms", {})
                    bullets = []
                    if "server_total_ms" in tm:
                        bullets.append(f"- **server_total_ms**: {tm['server_total_ms']} ms")
                    if "retrieval_ms" in tm:
                        bullets.append(f"- **retrieval_ms**: {tm['retrieval_ms']} ms")
                    if "generation_ms" in tm:
                        bullets.append(f"- **generation_ms**: {tm['generation_ms']} ms")
                    if "client_rtt_ms" in t:
                        bullets.append(f"- **client_rtt_ms**: {t['client_rtt_ms']} ms")
                    if bullets:
                        st.markdown("\n".join(bullets))
