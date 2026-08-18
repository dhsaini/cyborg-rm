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
from scripts.drafting import draft_message

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


def render_client_table(matches: pd.DataFrame, key_prefix: str):
    """
    One row per client with a Draft button. Drafting is on-demand, not
    automatic — calling Gemini for every row the moment a scenario loads
    would spend quota on drafts nobody asked to see.

    key_prefix must be unique per call site (e.g. distinguishes Mode 1's
    table from Mode 2's) so Streamlit's widget keys don't collide when both
    are on screen.

    Drafts are stored in st.session_state, not a local variable — Streamlit
    reruns this entire script on every click, so anything not in
    session_state is lost the moment the RM clicks a second client's button.
    """
    if matches.empty:
        st.info("No clients in this book match this scenario.")
        return

    for _, row in matches.iterrows():
        draft_key = f"{key_prefix}_{row['client_id']}"

        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.markdown(f"**{row['first_name']}** · {row['segment']} · "
                           f"₹{row['total_aum_inr'] / 1_00_00_000:.1f} Cr")
            with col2:
                st.caption(row.get("reason", ""))
            with col3:
                clicked = st.button("Draft", key=f"btn_{draft_key}")

            if clicked:
                with st.spinner("Drafting…"):
                    message, status = draft_message(
                        first_name=row["first_name"],
                        reason=row.get("reason", ""),
                        language=row.get("preferred_language", "English"),
                    )
                st.session_state[draft_key] = (message, status)

            if draft_key in st.session_state:
                message, status = st.session_state[draft_key]
                if status == "ok":
                    st.text_area("Draft — copy to WhatsApp", value=message,
                                 key=f"area_{draft_key}", height=100)
                elif status == "blocked":
                    st.warning("This draft was flagged by the compliance "
                              "check and withheld. Try again, or draft "
                              "this message yourself.")
                else:
                    st.error(f"Could not generate a draft ({status}). "
                             "Check your Gemini API key in "
                             ".streamlit/secrets.toml.")


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
        render_client_table(matches, key_prefix="mode1")

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
                pitch_top = pitch.head(10).copy()
                pitch_top["reason"] = (
                    f"Suitable for {product} — readiness score "
                    + pitch_top["suitability_score"].astype(str)
                )
                render_client_table(pitch_top, key_prefix="mode2")

        with col2:
            st.markdown(f"**Excluded — {len(excluded)} clients**")
            st.caption("Shown by design, not as an afterthought — every "
                      "exclusion has a stated, auditable reason. Excluded "
                      "clients cannot be drafted for from this screen.")
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
