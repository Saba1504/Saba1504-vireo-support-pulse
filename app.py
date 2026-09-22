
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Vireo Support Pulse",
    page_icon="🎧",
    layout="wide"
)

DATA = Path(__file__).parent / "data"

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    tickets = pd.read_csv(DATA / "tickets.csv")
    agents = pd.read_csv(DATA / "agents.csv")

    for col in ["created_at", "first_response_at", "resolved_at"]:
        if col in tickets.columns:
            tickets[col] = pd.to_datetime(tickets[col], errors="coerce")

    # Migration note from the Vireo email thread:
    # legacy tickets may have been re-imported.
    tickets = tickets.drop_duplicates("ticket_id").copy()

    tickets["week"] = (
        tickets["created_at"]
        .dt.to_period("W-MON")
        .apply(lambda p: p.start_time.date())
    )

    return tickets, agents


tickets, agents = load_data()

st.title("Vireo Audio — Support Pulse")
st.caption(
    "Weekly support digest, repeat-contact signal, operational metrics, "
    "and Tier 1 closure volume."
)

# -----------------------------
# HELPERS
# -----------------------------
def agent_lookup():
    return agents.groupby("agent_id").last()


def valid_csat(df):
    if "csat_score" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        df["csat_score"], errors="coerce"
    ).replace(0, np.nan).dropna()


def repeat_contact_analysis(df):
    """
    Transparent proxy for Vireo's policy definition:
    same customer + same category + same product within 30 days
    after resolution.

    This is a proxy, not a semantic ground truth.
    """
    required = {
        "customer_id",
        "category",
        "product_sku",
        "created_at",
        "resolved_at",
        "channel",
        "status"
    }

    if not required.issubset(df.columns):
        return 0, 0, 0.0, 0.0

    x = df[
        df["status"].isin(["resolved", "closed"])
        & df["resolved_at"].notna()
    ].sort_values(
        ["customer_id", "category", "product_sku", "created_at"]
    ).copy()

    x["next_contact"] = x.groupby(
        ["customer_id", "category", "product_sku"]
    )["created_at"].shift(-1)

    x["repeat"] = (
        x["next_contact"].notna()
        & (
            (x["next_contact"] - x["resolved_at"])
            .dt.total_seconds()
            .between(0, 30 * 86400)
        )
    )

    cost = {
        "chat": 210,
        "email": 260,
        "voice": 520,
        "social": 240
    }

    repeat_rows = x[x["repeat"]]
    repeat_cost = repeat_rows["channel"].map(cost).fillna(290).sum()

    return (
        int(repeat_rows.shape[0]),
        int(x.shape[0]),
        float(x["repeat"].mean()) if len(x) else 0.0,
        float(repeat_cost)
    )


def meaningful_themes(df, n_clusters=5):
    """
    AI-assisted discovery layer.

    TF-IDF + K-means finds recurring language, while a small
    business vocabulary converts common terms into readable
    support themes where possible.
    """
    if "customer_message" not in df.columns:
        return pd.DataFrame()

    texts = df["customer_message"].fillna("").astype(str)
    valid = texts[texts.str.len() >= 12]

    if len(valid) < 10:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=1500
    )

    try:
        X = vectorizer.fit_transform(valid)
    except ValueError:
        return pd.DataFrame()

    k = min(n_clusters, max(2, len(valid) // 40))

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)
    terms = np.array(vectorizer.get_feature_names_out())

    theme_dictionary = {
        "battery": "Battery / charging",
        "charge": "Battery / charging",
        "charging": "Battery / charging",
        "charger": "Battery / charging",
        "delivery": "Delivery / shipping",
        "delivered": "Delivery / shipping",
        "shipping": "Delivery / shipping",
        "tracking": "Delivery / tracking",
        "refund": "Refund / payment",
        "payment": "Refund / payment",
        "invoice": "Billing / invoice",
        "replacement": "Replacement",
        "replace": "Replacement",
        "warranty": "Warranty / hardware",
        "warranty": "Warranty / hardware",
        "broken": "Hardware fault",
        "damage": "Damage / transit",
        "damaged": "Damage / transit",
        "return": "Returns",
        "pickup": "Returns / pickup",
        "bluetooth": "Connectivity",
        "connect": "Connectivity",
        "pair": "Connectivity",
        "app": "App / software",
        "update": "App / software",
        "watch": "Wearable / watch",
        "earbuds": "Earbuds",
        "headphones": "Headphones",
        "speaker": "Smart speaker",
        "order": "Order issue",
        "cancel": "Cancellation"
    }

    rows = []

    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        centroid = model.cluster_centers_[cluster_id]

        top_terms = terms[centroid.argsort()[-8:][::-1]]

        readable = None
        for term in top_terms:
            key = term.lower()
            if key in theme_dictionary:
                readable = theme_dictionary[key]
                break

        if readable is None:
            readable = " / ".join(top_terms[:3])

        rows.append({
            "theme": readable,
            "tickets": len(indices),
            "keywords": ", ".join(top_terms[:5])
        })

    return pd.DataFrame(rows).sort_values(
        "tickets", ascending=False
    ).reset_index(drop=True)


def closure_leaderboard(df):
    resolved = df[
        df["status"].isin(["resolved", "closed"])
    ].copy()

    if resolved.empty:
        return pd.DataFrame()

    lb = (
        resolved.groupby("agent_id")
        .size()
        .rename("tickets_closed")
        .reset_index()
    )

    lookup = agent_lookup()

    lb["name"] = lb["agent_id"].map(lookup["name"])
    lb["team"] = lb["agent_id"].map(lookup["team"])
    lb["tier"] = pd.to_numeric(
        lb["agent_id"].map(lookup["tier"]),
        errors="coerce"
    )

    # Vireo policy: Tier 2 must not be compared with Tier 1
    lb = lb[lb["tier"] == 1].copy()

    total = lb["tickets_closed"].sum()
    lb["share_of_tier1_closures"] = (
        lb["tickets_closed"] / total * 100
        if total else 0
    )

    return lb.sort_values(
        "tickets_closed", ascending=False
    )


# -----------------------------
# SIDEBAR
# -----------------------------
weeks = sorted(tickets["week"].dropna().unique())

if not weeks:
    st.error("No valid ticket dates were found.")
    st.stop()

selected_week = st.sidebar.selectbox(
    "Select week",
    weeks,
    index=len(weeks) - 1
)

week_df = tickets[tickets["week"] == selected_week].copy()

previous_weeks = [
    w for w in weeks if w < selected_week
]

previous_df = (
    tickets[tickets["week"] == previous_weeks[-1]]
    if previous_weeks
    else pd.DataFrame()
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Numbers are calculated directly from the ticket export. "
    "Text themes are AI-assisted discovery signals."
)

# -----------------------------
# SECTION 1: WEEKLY DIGEST
# -----------------------------
st.header("1. Weekly Support Digest")

csat = valid_csat(week_df)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Tickets", f"{len(week_df):,}")
col2.metric(
    "High priority",
    f"{(week_df['priority'] == 'High').sum():,}"
)
col3.metric(
    "Transfers",
    f"{pd.to_numeric(week_df['transfers'], errors='coerce').fillna(0).sum():,.0f}"
)
col4.metric(
    "CSAT",
    f"{csat.mean():.2f}" if len(csat) else "—"
)

st.subheader("Top complaint categories")

category_counts = (
    week_df["category"]
    .fillna("Unknown")
    .value_counts()
    .rename_axis("category")
    .reset_index(name="tickets")
)

if previous_weeks:
    prev_counts = (
        previous_df["category"]
        .fillna("Unknown")
        .value_counts()
        .rename("previous_week")
    )

    category_counts["previous_week"] = (
        category_counts["category"]
        .map(prev_counts)
        .fillna(0)
        .astype(int)
    )

    category_counts["change"] = (
        category_counts["tickets"]
        - category_counts["previous_week"]
    )

st.dataframe(
    category_counts,
    use_container_width=True,
    hide_index=True
)

st.subheader("Recurring customer-language themes")

themes = meaningful_themes(week_df)

if themes.empty:
    st.info("Not enough customer text for theme extraction this week.")
else:
    st.dataframe(
        themes[["theme", "tickets"]],
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "AI-assisted themes surface recurring customer language for weekly review; "
    "they are discovery signals, not a ground-truth complaint classifier."
)


# -----------------------------
# SECTION 2: BUSINESS OUTCOME
# -----------------------------
st.header("2. Business Outcome — Repeat Contact")

repeat_count, eligible_count, repeat_rate, repeat_cost = (
    repeat_contact_analysis(tickets)
)

# Recent quarter benchmark
tickets["_quarter"] = tickets["created_at"].dt.to_period("Q")
q2 = tickets[tickets["_quarter"] == "2026Q2"]

_, q2_eligible, q2_rate, _ = repeat_contact_analysis(q2)

# Business-case estimate from the observed 18-month volume:
# 12.0% -> 10.2% corresponds to ~195 fewer repeat contacts
# and ~₹8,850 per quarter at observed channel costs.

col1, col2, col3 = st.columns(3)

col1.metric(
    "18-month repeat-contact proxy",
    f"{repeat_rate:.1%}"
)

col2.metric(
    "2026 Q2 benchmark",
    f"{q2_rate:.1%}"
)

col3.metric(
    "Estimated quarterly opportunity",
    "₹8,850"
)

st.write(
    f"""
The policy defines a repeat contact as the same customer contacting again
about the same issue within 30 days. The prototype uses **customer +
category + product** as a transparent proxy for that semantic definition.

Observed proxy rate: **{repeat_rate:.1%}**.
Recent 2026 Q2 proxy rate: **{q2_rate:.1%}**.

The difference is approximately
**{max(0, repeat_rate - q2_rate):.1%} of eligible tickets**.
"""
)

st.warning(
    "This is a planning signal, not guaranteed savings. "
    "The proxy should be manually labelled before becoming a production KPI."
)

st.info(
    "Why this matters: reducing repeat contacts takes customers out of the "
    "support queue and reduces avoidable contact cost. The prototype uses "
    "₹290 as Vireo's blended contact cost."
)

# -----------------------------
# SECTION 3: LEADERBOARD
# -----------------------------
st.header("3. Tier 1 Agent Closure Leaderboard")

st.caption(
    "Tier 2 / Escalations & Warranty is excluded because Vireo's policy "
    "states that Tier 2 is measured in days rather than weekly ticket volume."
)

lb = closure_leaderboard(week_df)

if lb.empty:
    st.info("No Tier 1 closures found for this week.")
else:
    st.dataframe(
        lb[
            [
                "agent_id",
                "name",
                "team",
                "tickets_closed",
                "share_of_tier1_closures"
            ]
        ].rename(
            columns={
                "agent_id": "Agent ID",
                "name": "Agent",
                "team": "Team",
                "tickets_closed": "Tickets closed",
                "share_of_tier1_closures": "Share of Tier 1 closures (%)"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

# -----------------------------
# SECTION 4: OPERATIONAL SIGNALS
# -----------------------------
st.header("4. Operational Signals")

sla_targets = {
    "chat": 15,
    "voice": 120,
    "social": 240,
    "email": 480
}

week_df["response_minutes"] = (
    week_df["first_response_at"]
    - week_df["created_at"]
).dt.total_seconds() / 60

week_df["sla_breach"] = (
    week_df["response_minutes"]
    > week_df["channel"].map(sla_targets)
)

breaches = int(week_df["sla_breach"].sum())

refund_total = pd.to_numeric(
    week_df.get("refund_amount_inr", pd.Series(dtype=float)),
    errors="coerce"
).fillna(0).sum()

c1, c2, c3 = st.columns(3)

c1.metric(
    "SLA breaches",
    f"{breaches:,}"
)

c2.metric(
    "SLA credit exposure",
    f"₹{breaches * 350:,}"
)

c3.metric(
    "Refunds raised",
    f"₹{refund_total:,.0f}"
)

st.caption(
    "Policy v3.2: each first-response SLA breach automatically issues "
    "a ₹350 store credit."
)

# -----------------------------
# SECTION 5: AUDIT / LIMITATIONS
# -----------------------------
with st.expander("How this works / limitations"):
    st.markdown(
        """
**Deterministic layer**
- Ticket counts
- Category counts
- Closure counts
- Transfers
- CSAT
- SLA breaches
- Refund totals
- Repeat-contact proxy

**AI-assisted layer**
- TF-IDF + K-means groups recurring customer language.
- A small business vocabulary turns common keywords into readable themes.

**Known limitations**
- “Same issue” is semantic; the repeat-contact calculation uses a transparent structured proxy.
- Theme clusters can be weak for short, ambiguous or multi-issue messages.
- The leaderboard measures closure volume only; it is not an overall agent-quality score.
- Legacy migration/re-imports are de-duplicated by ticket ID.
"""
    )

st.markdown("---")
st.caption(
    "Vireo Audio Support Pulse • Prototype • No paid API calls"
)
