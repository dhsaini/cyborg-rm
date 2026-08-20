"""
app.py — Cyborg RM, Streamlit shell.

Navigation model: one page, one piece of state — selected_client_id.
    None  -> Book Dashboard (aggregate charts + selectable client list)
    set   -> Client 360 (that client's profile + an "AI Suggestions" tab
             containing Mode 1 and Mode 2, scoped to that one client)

This replaces the earlier flat layout (RM picker -> Mode 1 section -> Mode 2
section, both always visible) with something closer to how a real RM tool is
navigated: book overview first, drill into one relationship, act from there.

Run from the repo root:   streamlit run app.py
"""

import pandas as pd
import streamlit as st

from scripts.filters import (
    SCENARIOS, run_scenario, run_all_scenarios_for_client,
    score_product_fit,
)
from scripts.drafting import draft_message

CR = 1_00_00_000  # 1 crore, for AUM display conversions

PRODUCT_CATEGORIES = [
    "AI & Technology Thematic Fund — Fund A",
    "Infrastructure & Capex Thematic Fund — Fund B",
    "Silver & Precious Metals Fund — Fund C",
]

ALLOC_COLUMNS = [
    "alloc_direct_equity_pct", "alloc_equity_mf_pct", "alloc_debt_mf_bonds_pct",
    "alloc_pms_pct", "alloc_aif_pct", "alloc_unlisted_pct", "alloc_gold_pct",
]
ALLOC_LABELS = {
    "alloc_direct_equity_pct": "Direct Equity",
    "alloc_equity_mf_pct": "Equity MF",
    "alloc_debt_mf_bonds_pct": "Debt MF / Bonds",
    "alloc_pms_pct": "PMS",
    "alloc_aif_pct": "AIF",
    "alloc_unlisted_pct": "Unlisted",
    "alloc_gold_pct": "Gold",
}

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

    key_prefix must be unique per call site (Mode 1's table, Mode 2's pitch
    list, and the single-client suggestions view all reuse this function)
    so Streamlit's widget keys don't collide when more than one is on screen.

    Drafts are stored in st.session_state, not a local variable — Streamlit
    reruns this entire script on every click, so anything not in
    session_state is lost the moment the RM clicks a second client's button.
    """
    if matches.empty:
        st.info("No clients match this scenario.")
        return

    for position, (_, row) in enumerate(matches.iterrows()):
        # position disambiguates when the same client appears more than
        # once in matches — e.g. run_all_scenarios_for_client correctly
        # returns one row per matched scenario, so a client exposed to both
        # a rate move AND a gold rally appears twice, and client_id alone
        # would produce duplicate widget keys.
        draft_key = f"{key_prefix}_{row['client_id']}_{position}"
        stored_language = row.get("preferred_language", "English")

        # Each client sees only English (universal fallback) plus their own
        # stored preference — never every language in the dataset. A
        # Gujarati-preference client should not be offered Bengali or
        # Marathi as an override; those languages are not theirs to choose.
        client_language_options = (
            [stored_language] if stored_language == "English"
            else ["English", stored_language]
        )

        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([2, 3, 1.3, 1])
            with col1:
                st.markdown(f"**{row['first_name']}** · {row['segment']} · "
                           f"₹{row['total_aum_inr'] / CR:.1f} Cr")
            with col2:
                st.caption(row.get("reason", ""))
            with col3:
                default_idx = client_language_options.index(stored_language)
                draft_language = st.selectbox(
                    "Language", options=client_language_options,
                    index=default_idx,
                    key=f"lang_{draft_key}", label_visibility="collapsed",
                )
            with col4:
                clicked = st.button("Draft", key=f"btn_{draft_key}")

            if draft_language != stored_language:
                st.caption(f"Overriding stored preference ({stored_language})")

            if clicked:
                with st.spinner("Drafting…"):
                    message, status = draft_message(
                        first_name=row["first_name"],
                        reason=row.get("reason", ""),
                        headline=row.get("headline", ""),
                        language=draft_language,
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


# ---------------------------------------------------------------------------
# Book Dashboard — default view, no client selected
# ---------------------------------------------------------------------------

def render_book_dashboard(book: pd.DataFrame):
    """
    Landing view for a selected RM: aggregate charts, then a selectable
    client list. Selecting a row hands off to the Client 360 view via
    st.session_state — this function never calls Mode 1/Mode 2 itself.
    """
    st.subheader("Book overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Clients", len(book))
    m2.metric("Total AUM", f"₹{book['total_aum_inr'].sum() / CR:.1f} Cr")
    m3.metric("Avg. relationship", f"{book['relationship_tenure_years'].mean():.1f} yrs")

    col_a, col_b = st.columns(2)

    with col_a:
        st.caption("Average allocation mix across this book")
        alloc_avg = book[ALLOC_COLUMNS].mean().rename(index=ALLOC_LABELS)
        st.bar_chart(alloc_avg)

    with col_b:
        st.caption("Risk profile breakdown")
        risk_counts = book["risk_profile"].value_counts()
        st.bar_chart(risk_counts)

    st.divider()
    st.subheader("Clients")
    st.caption("Select a row to open that client's full view.")

    display = book[["first_name", "segment", "risk_profile", "total_aum_inr",
                    "top_sector_exposure", "days_since_last_contact"]].copy()
    display["total_aum_inr"] = (display["total_aum_inr"] / CR).round(1)
    display.columns = ["Client", "Segment", "Risk Profile", "AUM (Cr)",
                       "Top Sector", "Days Since Contact"]

    event = st.dataframe(
        display, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )

    if event.selection and event.selection.get("rows"):
        selected_idx = event.selection["rows"][0]
        st.session_state["selected_client_id"] = book.iloc[selected_idx]["client_id"]
        st.rerun()


# ---------------------------------------------------------------------------
# Client 360 — one client selected
# ---------------------------------------------------------------------------

def render_client_360(full_df: pd.DataFrame, client_id: str):
    """
    Full-page view for one client: profile header, then a single "AI
    Suggestions" tab holding Mode 1 (scenario matches) and Mode 2 (product
    fit), both scoped to this client only — never the whole book.
    """
    client = full_df[full_df["client_id"] == client_id]
    if client.empty:
        st.error("Selected client not found. Returning to book view.")
        st.session_state["selected_client_id"] = None
        st.rerun()
        return
    client = client.iloc[0]

    if st.button("← Back to book"):
        st.session_state["selected_client_id"] = None
        st.rerun()

    st.subheader(f"{client['first_name']} — {client['segment']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUM", f"₹{client['total_aum_inr'] / CR:.1f} Cr")
    m2.metric("Risk Profile", client["risk_profile"])
    m3.metric("Relationship", f"{client['relationship_tenure_years']} yrs")
    m4.metric("Last Contact", f"{client['days_since_last_contact']}d ago")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.caption("Portfolio allocation")
        alloc = client[ALLOC_COLUMNS].rename(index=ALLOC_LABELS)
        st.bar_chart(alloc)
    with col_b:
        st.caption("Profile")
        st.write(f"**Wealth source:** {client['wealth_source']}")
        st.write(f"**Top sector exposure:** {client['top_sector_exposure']}")
        st.write(f"**Debt duration:** {client['debt_duration_bucket']}")
        if client["unlisted_holding_tag"] != "No holding":
            st.write(f"**Unlisted holding:** {client['unlisted_holding_tag']}")
        st.write(f"**Preferred language:** {client['preferred_language']}")

    st.divider()

    tab_suggestions, = st.tabs(["AI Suggestions"])

    with tab_suggestions:
        st.markdown("**Market events relevant to this client**")
        st.caption("Only scenarios this client's own portfolio is actually "
                  "exposed to are shown — not the full list of 4.")

        matches = run_all_scenarios_for_client(full_df, client_id)
        if matches.empty:
            st.info("No current market scenario shows meaningful exposure "
                   "for this client.")
        else:
            render_client_table(matches, key_prefix="c360_mode1")

        st.divider()
        st.markdown("**Product fit**")

        product = st.selectbox(
            "Check fit against a product category",
            options=PRODUCT_CATEGORIES, index=None,
            placeholder="Choose a product category…",
            key="c360_product_select",
        )

        if product:
            client_row = full_df[full_df["client_id"] == client_id]
            pitch, excluded = score_product_fit(client_row, product)

            if not pitch.empty:
                st.success("This client fits the pitch criteria for this product.")
                pitch_row = pitch.copy()
                pitch_row["reason"] = (
                    f"Suitable for {product} — readiness score "
                    + pitch_row["suitability_score"].astype(str)
                )
                render_client_table(pitch_row, key_prefix="c360_mode2")
            else:
                reason = excluded.iloc[0]["exclusion_reason"]
                st.warning(f"This client is excluded from this pitch: {reason}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Cyborg RM", layout="wide")

    st.title("Cyborg RM")
    st.caption("A relationship-manager copilot — synthetic data, demo build.")
    render_disclaimer()
    st.divider()

    clients = load_clients()

    if "selected_client_id" not in st.session_state:
        st.session_state["selected_client_id"] = None

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

    # --- Language filter: narrows the book itself, before either view runs --
    available_languages = sorted(book["preferred_language"].unique())
    language_filter = st.multiselect(
        "Filter by preferred language",
        options=available_languages,
        default=available_languages,
        help="Narrows this RM's book to clients with the selected preferred "
             "language(s).",
    )

    if not language_filter:
        st.warning("Select at least one language to see clients.")
        return

    book = book[book["preferred_language"].isin(language_filter)]
    st.caption(f"{len(book)} of {(clients['rm_id'] == selected_rm).sum()} "
              f"clients match the selected language(s).")

    st.divider()

    # --- State switch: book dashboard, or one client's 360 view -------------
    if st.session_state["selected_client_id"] is None:
        render_book_dashboard(book)
    else:
        render_client_360(clients, st.session_state["selected_client_id"])

    st.divider()
    render_disclaimer()


if __name__ == "__main__":
    main()
