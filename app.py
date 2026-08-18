"""
app.py — Cyborg RM, Streamlit shell.

This version proves the access-boundary UX and the data pipeline end to end.
It does not call Gemini yet — Mode 1 shows the ranked client table with
reasons; message drafting is wired in the next step (drafting.py).

Run from the repo root:   streamlit run app.py
"""

import pandas as pd
import streamlit as st

from scripts.filters import SCENARIOS, run_scenario, score_product_fit

PRODUCT_CATEGORIES = [
    "AI & Technology Thematic Fund — Fund A",
    "Infrastructure & Capex Thematic Fund — Fund B",
    "Silver & Precious Metals Fund — Fund C",
]

DISCLAIMER = (
    "Cyborg RM is a completely independent learning project by Dhirendra Saini. "
    "All data shown is synthetic. This is not affiliated with any company, and "
    "does not provide investment advice, research, or a recommendation to buy "
    "or sell any security."
)

DATA_PATH = "data/clients.csv"


@st.cache_data
def load_clients() -> pd.DataFrame:
    """
    Cached so the CSV is read from disk once per session, not on every
    interaction. Streamlit reruns this whole script top-to-bottom on every
    click — @st.cache_data is what stops that from re-reading the file
    a thousand times.
    """
    return pd.read_csv(DATA_PATH)


def render_disclaimer():
    st.caption(DISCLAIMER)


def render_client_table(matches: pd.DataFrame):
    """Client-facing table — no internal client_id, formatted for reading."""
    if matches.empty:
        st.info("No clients in this book match this scenario.")
        return

    display = matches[[
        "first_name", "segment", "risk_profile", "total_aum_inr",
        "exposure_score", "reason",
    ]].copy()

    display["total_aum_inr"] = display["total_aum_inr"].apply(
        lambda v: f"₹{v / 1_00_00_000:.1f} Cr"
    )
    display["exposure_score"] = display["exposure_score"].round(1)
    display.columns = ["Client", "Segment", "Risk Profile", "AUM",
                       "Exposure Score", "Why this client"]

    st.dataframe(display, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Cyborg RM", layout="wide")

    st.title("Cyborg RM")
    st.caption("A relationship-manager copilot — synthetic data, demo build.")
    render_disclaimer()
    st.divider()

    clients = load_clients()

    # --- Access gate: nothing below this renders until an RM is chosen ------
    rm_ids = sorted(clients["rm_id"].unique())
    rm_labels = {
        rm_id: f"{rm_id} — {clients.loc[clients['rm_id'] == rm_id, 'desk'].iloc[0]} "
               f"({(clients['rm_id'] == rm_id).sum()} clients)"
        for rm_id in rm_ids
    }

    selected_rm = st.selectbox(
        "Select your book",
        options=rm_ids,
        format_func=lambda x: rm_labels[x],
        index=None,
        placeholder="Choose an RM to load their book…",
    )

    if selected_rm is None:
        st.info("Select an RM above. You will only ever see that RM's own "
               "clients — this mirrors how access works on a real desk.")
        return

    book = clients[clients["rm_id"] == selected_rm]
    st.success(f"Loaded {len(book)} clients for {rm_labels[selected_rm]}.")

    st.divider()

    # --- Mode 1: reactive ----------------------------------------------------
    st.subheader("Mode 1 — Market event")
    st.caption("Select a scenario. Clients are ranked by actual portfolio "
              "exposure, not just a keyword match.")

    scenario_name = st.selectbox(
        "Scenario", options=list(SCENARIOS.keys()), index=None,
        placeholder="Choose a market event…",
    )

    if scenario_name:
        top_n = st.slider("Show top", min_value=3, max_value=15, value=10)
        matches = run_scenario(clients, selected_rm, scenario_name, top_n=top_n)
        st.write(f"**{len(matches)} clients** most exposed, out of "
                f"{len(book)} in this book:")
        render_client_table(matches)

    st.divider()

    # --- Mode 2: proactive ---------------------------------------------------
    st.subheader("Mode 2 — New product pitch")
    st.caption("Product categories are illustrative only — no real fund, AMC, "
              "or NFO is represented. Every pitch list is shown alongside "
              "who was excluded, and why.")

    product = st.selectbox(
        "Product category", options=PRODUCT_CATEGORIES, index=None,
        placeholder="Choose a product category…",
    )

    if product:
        pitch, excluded = score_product_fit(book, product)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Pitch list — {len(pitch)} clients**")
            st.caption("Ranked by deployment readiness: idle capital and "
                      "time since last investment.")
            if pitch.empty:
                st.info("No clients in this book are suitable for this pitch.")
            else:
                display = pitch[["first_name", "risk_profile",
                                "suitability_score"]].head(10).copy()
                display.columns = ["Client", "Risk Profile", "Readiness Score"]
                st.dataframe(display, use_container_width=True, hide_index=True)

        with col2:
            st.markdown(f"**Excluded — {len(excluded)} clients**")
            st.caption("Shown by design, not as an afterthought — every "
                      "exclusion has a stated, auditable reason.")
            if excluded.empty:
                st.info("No clients were excluded for this product.")
            else:
                display = excluded[["first_name", "exclusion_reason"]].copy()
                display.columns = ["Client", "Reason excluded"]
                st.dataframe(display, use_container_width=True, hide_index=True)

        assert len(pitch) + len(excluded) == len(book), (
            "Pitch and exclusion lists must always account for the entire "
            "book — a client silently missing from both would defeat the "
            "point of showing exclusions at all."
        )

    st.divider()
    render_disclaimer()


if __name__ == "__main__":
    main()
