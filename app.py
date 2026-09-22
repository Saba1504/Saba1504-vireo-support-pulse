
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Vireo Support Pulse", layout="wide")
st.title("Vireo Audio — Support Pulse")
st.caption("AI-assisted weekly support digest + Tier 1 closure leaderboard")

DATA = Path(__file__).parent / "data"
tickets = pd.read_csv(DATA / "tickets.csv")
agents = pd.read_csv(DATA / "agents.csv")
products = pd.read_csv(DATA / "products.csv")

for c in ["created_at","first_response_at","resolved_at"]:
    tickets[c] = pd.to_datetime(tickets[c], errors="coerce")

tickets = tickets.drop_duplicates("ticket_id").copy()
tickets["week"] = tickets["created_at"].dt.to_period("W-MON").apply(lambda p: p.start_time.date())

def themes(df, n=6):
    texts = df["customer_message"].fillna("").astype(str)
    texts = texts[texts.str.len() > 10]
    if len(texts) < 10:
        return pd.DataFrame()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3, max_features=1000)
    X = vec.fit_transform(texts)
    k = min(n, max(2, len(texts)//50))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        centroid = km.cluster_centers_[i]
        top = terms[centroid.argsort()[-5:][::-1]]
        rows.append({"theme": " / ".join(top[:3]), "tickets": len(idx)})
    return pd.DataFrame(rows).sort_values("tickets", ascending=False)

weeks = sorted(tickets["week"].dropna().unique())
selected = st.sidebar.selectbox("Week", weeks, index=len(weeks)-1)
w = tickets[tickets.week == selected].copy()

st.subheader(f"Weekly digest — week of {selected}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Tickets", len(w))
c2.metric("High priority", int((w.priority=="High").sum()))
c3.metric("Transfers", int(w.transfers.sum()))
c4.metric("CSAT", f"{w.csat_score.replace(0,np.nan).dropna().mean():.2f}" if w.csat_score.replace(0,np.nan).dropna().size else "—")

st.markdown("### What customers are contacting us about")
cat = w["category"].value_counts().rename_axis("category").reset_index(name="tickets")
st.dataframe(cat, use_container_width=True, hide_index=True)

st.markdown("### AI-assisted theme extraction")
st.caption("TF-IDF + K-means groups the week's customer messages into recurring language themes. Category tags remain the auditable baseline.")
th = themes(w)
if len(th):
    st.dataframe(th, use_container_width=True, hide_index=True)
else:
    st.info("Not enough text for clustering this week.")

st.markdown("### Agent leaderboard — Tier 1 only")
st.caption("Escalations & Warranty / Tier 2 are excluded because the operating policy says Tier 2 is measured in days, not weekly ticket volume.")
resolved = w[w.status.isin(["resolved","closed"])].copy()
lb = resolved.groupby("agent_id").size().rename("tickets_closed").reset_index()
lb["name"] = lb.agent_id.map(agents.groupby("agent_id").name.last())
lb["team"] = lb.agent_id.map(agents.groupby("agent_id").team.last())
lb["tier"] = lb.agent_id.map(agents.groupby("agent_id").tier.last())
lb = lb[(lb.tier==1) & (lb.team!="Escalations & Warranty")].sort_values("tickets_closed", ascending=False)
st.dataframe(lb[["agent_id","name","team","tickets_closed"]], use_container_width=True, hide_index=True)

st.markdown("### Quality / cost signals")
cost = {"chat":210,"email":260,"voice":520,"social":240}
targets = {"chat":15,"voice":120,"social":240,"email":480}
w["fr_min"] = (w.first_response_at-w.created_at).dt.total_seconds()/60
w["breach"] = w.fr_min > w.channel.map(targets)
b1,b2 = st.columns(2)
b1.metric("SLA breaches", int(w.breach.sum()))
b2.metric("SLA credit exposure", f"₹{int(w.breach.sum()*350):,}")
st.caption("SLA breach credit is ₹350 per breached ticket under policy v3.2.")

st.markdown("### Prompt / model audit")
st.code("""Task: summarize the selected week's customer complaints.
Constraints: use only customer_message text; preserve category counts; do not infer causes not present in the data.
Output: top themes, examples, change vs prior week, and a short action list.
Validation: compare theme/category counts to deterministic SQL/Pandas counts; manually review a sample of summaries.""")
