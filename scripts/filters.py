"""
filters.py — deterministic scenario matching for Cyborg RM.

This module decides WHO is exposed to a market scenario and WHY, ranked by
exposure strength. It never calls an LLM and never decides suitability alone —
per the PRD (§8), the model only drafts language once this module has already
chosen the recipients.

Each scenario function takes a DataFrame already filtered to one RM's book
and returns a new DataFrame with two added columns:
    - exposure_score   float, higher = more exposed
    - reason           str, plain-English justification for this client

Adding a fifth scenario later means writing one function with this shape and
registering it in SCENARIOS — nothing else in the app needs to change.
"""

import pandas as pd

DEFAULT_TOP_N = 10


def _finalize(df: pd.DataFrame, score: pd.Series, reason: pd.Series,
              min_score: float) -> pd.DataFrame:
    """Attach score + reason, drop non-matches, sort by exposure descending."""
    out = df.copy()
    out["exposure_score"] = score
    out["reason"] = reason
    out = out[out["exposure_score"] >= min_score]
    return out.sort_values("exposure_score", ascending=False)


def scenario_rbi_rate_move(df: pd.DataFrame) -> pd.DataFrame:
    """
    RBI rate move. Exposure = duration risk in the fixed-income book.

    Long-duration debt is the most rate-sensitive. Score is the debt
    allocation itself, weighted up for Long duration and down for Short,
    so a client with 40% debt in Long duration ranks above one with 40% in
    Short, and both rank above a client with only 10% debt regardless of
    duration.
    """
    duration_weight = df["debt_duration_bucket"].map(
        {"Long": 1.0, "Medium": 0.5, "Short": 0.15, "None": 0.0}
    ).fillna(0.0)

    score = df["alloc_debt_mf_bonds_pct"] * duration_weight

    reason = (
        df["debt_duration_bucket"] + " duration debt at "
        + df["alloc_debt_mf_bonds_pct"].round(1).astype(str) + "% of portfolio"
    )

    return _finalize(df, score, reason, min_score=8.0)


def scenario_it_sector_selloff(df: pd.DataFrame) -> pd.DataFrame:
    """
    IT sector selloff. Exposure = direct equity concentration in IT.

    Only clients whose top sector IS IT are exposed at all — a client with
    5% equity and no IT tilt is not "a little exposed," they are not exposed.
    Among IT-tilted clients, score scales with direct equity weight, since
    that is the allocation actually sitting in single-stock/sector risk.
    """
    is_it = df["top_sector_exposure"] == "IT"
    score = df["alloc_direct_equity_pct"].where(is_it, 0.0)

    reason = (
        "IT-concentrated direct equity book at "
        + df["alloc_direct_equity_pct"].round(1).astype(str) + "%"
    )

    return _finalize(df, score, reason, min_score=12.0)


def scenario_rupee_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rupee depreciation / gold rally. Three distinct beneficiary groups under
    one headline — the mapping a generic tool would not make:

      1. Gold allocation gains directly.
      2. International equity gains on FX translation.
      3. IT-sector exposure gains as export earnings re-rate.

    Each client is scored on whichever driver applies to them; the reason
    names which one. A client can match on more than one driver — score is
    additive, so a client with both gold and IT exposure ranks higher, which
    is correct: they benefit twice.
    """
    gold_component = df["alloc_gold_pct"].clip(lower=0)
    intl_component = df["intl_equity_exposure_pct"] * 2.5  # smaller % baseline
    it_component = df["alloc_direct_equity_pct"].where(
        df["top_sector_exposure"] == "IT", 0.0
    ) * 0.5  # secondary/indirect beneficiary, weighted down vs direct holders

    score = gold_component + intl_component + it_component

    reasons = []
    for g, i, it_flag, it_val in zip(
        df["alloc_gold_pct"], df["intl_equity_exposure_pct"],
        df["top_sector_exposure"] == "IT", df["alloc_direct_equity_pct"]
    ):
        parts = []
        if g >= 8:
            parts.append(f"gold at {g:.1f}%")
        if i >= 2:
            parts.append(f"international equity at {i:.1f}%")
        if it_flag and it_val >= 10:
            parts.append("IT-sector export earnings tailwind")
        reasons.append("; ".join(parts) if parts else "")

    return _finalize(df, score, pd.Series(reasons, index=df.index), min_score=8.0)


def scenario_ipo_filing(df: pd.DataFrame, company_tag: str = "Unlisted Co. A") -> pd.DataFrame:
    """
    IPO filing by a company held unlisted. Binary exposure: a client either
    holds this specific name or they do not. No other client is "a little"
    exposed to one company's IPO.

    company_tag must match a value in unlisted_holding_tag exactly (see
    scripts/generate_clients.py: UNLISTED_TAGS).
    """
    holds_it = df["unlisted_holding_tag"] == company_tag
    score = df["alloc_unlisted_pct"].where(holds_it, 0.0)

    reason = (
        f"Holds {company_tag} (unlisted) at "
        + df["alloc_unlisted_pct"].round(1).astype(str) + "% of portfolio"
    )

    return _finalize(df, score, reason, min_score=0.01)


PRODUCT_PROFILES = {
    "AI & Technology Thematic Fund — Fund A": {
        "relevant_sector": "IT",
        "suitable_risk_profiles": ("Moderate", "Aggressive"),
        "min_ticket_inr": 0,
    },
    "Infrastructure & Capex Thematic Fund — Fund B": {
        "relevant_sector": "Infra",
        "suitable_risk_profiles": ("Moderate", "Aggressive"),
        "min_ticket_inr": 0,
    },
    "Silver & Precious Metals Fund — Fund C": {
        "relevant_sector": None,  # not sector-linked; a diversifier, so open to Conservative too
        "suitable_risk_profiles": ("Conservative", "Moderate", "Aggressive"),
        "min_ticket_inr": 0,
    },
}


def score_product_fit(df: pd.DataFrame, product_category: str,
                      min_ticket_inr: float = 0,
                      suitable_risk_profiles: tuple = None,
                      max_existing_thematic_pct: float = 15.0,
                      quiet_period_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mode 2 — proactive product pitch. Splits one RM's book into a pitch list
    and an exclusion list for a generic product category.

    product_category must be a key in PRODUCT_PROFILES. This function never
    sees or needs a real fund name, AMC, or NFO (PRD §9) — but it DOES need
    product_category to actually change the outcome; a filter that returns
    the same split for every product isn't filtering by product at all.

    Exclusion rules (PRD §9.4 — the exclusion list is structural; these four
    specific rules are configuration and may be revised):
      1. Already meaningfully exposed to this theme — a pitch would just add
         concentration risk, not diversification.
      2. Risk profile does not fit this specific product's suitable range.
      3. Ticket size would fall below the product's stated minimum.
      4. Contacted very recently — avoid re-approaching a client mid-conversation.

    A client can trigger more than one rule; only the first applicable reason
    is shown, in the priority order above, so the list stays readable.

    Returns (pitch_df, excluded_df). pitch_df is ranked by suitability;
    excluded_df preserves book order since exclusion reasons, not ranking,
    are the point.
    """
    profile = PRODUCT_PROFILES.get(product_category, {})
    if suitable_risk_profiles is None:
        suitable_risk_profiles = profile.get(
            "suitable_risk_profiles", ("Moderate", "Aggressive")
        )
    min_ticket_inr = min_ticket_inr or profile.get("min_ticket_inr", 0)
    relevant_sector = profile.get("relevant_sector")

    exclusion_reason = pd.Series([""] * len(df), index=df.index, dtype="object")

    over_exposed = df["existing_thematic_exposure_pct"] >= max_existing_thematic_pct
    exclusion_reason = exclusion_reason.mask(
        over_exposed & (exclusion_reason == ""),
        "Already carries " + df["existing_thematic_exposure_pct"].round(1).astype(str)
        + "% thematic exposure — pitching adds concentration, not diversification"
    )

    wrong_risk = ~df["risk_profile"].isin(suitable_risk_profiles)
    exclusion_reason = exclusion_reason.mask(
        wrong_risk & (exclusion_reason == ""),
        df["risk_profile"] + " risk profile does not fit this product category"
    )

    if min_ticket_inr > 0:
        # A plausible ticket is estimated from the client's own deployment
        # behaviour, not invented — their typical lumpsum size.
        too_small = df["avg_lumpsum_ticket_inr"] < min_ticket_inr
        exclusion_reason = exclusion_reason.mask(
            too_small & (exclusion_reason == ""),
            "Typical ticket size below this product's stated minimum"
        )

    too_recent = df["days_since_last_contact"] < quiet_period_days
    exclusion_reason = exclusion_reason.mask(
        too_recent & (exclusion_reason == ""),
        f"Contacted within the last {quiet_period_days} days — avoid overlap "
        "with an active conversation"
    )

    is_excluded = exclusion_reason != ""

    excluded = df[is_excluded].copy()
    excluded["exclusion_reason"] = exclusion_reason[is_excluded]

    # Suitability score for the remaining pitch pool: idle capital + how long
    # since they last deployed anything, so the RM sees the clients most
    # ready to act. Sector-linked products additionally boost clients whose
    # existing book already tilts toward that sector — someone overweight IT
    # is a more natural AI-fund conversation than someone with none.
    pitch = df[~is_excluded].copy()
    readiness = (
        pitch["ledger_balance_inr"] / pitch["total_aum_inr"].clip(lower=1) * 100
    ) + (pitch["days_since_last_lumpsum"] / 30)

    if relevant_sector:
        sector_match = pitch["top_sector_exposure"] == relevant_sector
        readiness = readiness + sector_match.astype(float) * 10

    pitch["suitability_score"] = readiness.round(1)
    pitch = pitch.sort_values("suitability_score", ascending=False)

    return pitch, excluded


SCENARIOS = {
    "RBI rate move": {
        "headline": "RBI cuts repo rate by 25bps, third consecutive cut this year",
        "filter_fn": scenario_rbi_rate_move,
    },
    "IT sector selloff": {
        "headline": "IT stocks slide 4% on weak US client spending guidance",
        "filter_fn": scenario_it_sector_selloff,
    },
    "Rupee depreciation / gold rally": {
        "headline": "Rupee slides past 89/USD as gold hits fresh record highs",
        "filter_fn": scenario_rupee_gold,
    },
    "IPO filing (unlisted holding)": {
        "headline": "Unlisted Co. A files draft IPO papers with SEBI",
        "filter_fn": scenario_ipo_filing,
    },
}


def run_scenario(df: pd.DataFrame, rm_id: str, scenario_name: str,
                  top_n: int = DEFAULT_TOP_N, **kwargs) -> pd.DataFrame:
    """
    Main entry point. Filters to one RM's book, runs the named scenario,
    returns at most top_n clients ranked by exposure, with the scenario's
    real headline attached to every row as `headline`.

    Carrying the headline on the output — not just the reason — is what lets
    drafting.py write a message that names the actual news event, instead of
    a generic sentence about an allocation percentage.

    This function is the enforcement point for the access boundary (PRD §3):
    every caller goes through rm_id, so it is structurally impossible to see
    another RM's clients through this module.
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name!r}. "
                         f"Choose from {list(SCENARIOS)}.")

    book = df[df["rm_id"] == rm_id]
    if book.empty:
        raise ValueError(f"No clients found for rm_id={rm_id!r}.")

    scenario = SCENARIOS[scenario_name]
    matched = scenario["filter_fn"](book, **kwargs) if kwargs else \
        scenario["filter_fn"](book)

    matched = matched.head(top_n).copy()
    matched["headline"] = scenario["headline"]
    return matched


def run_all_scenarios_for_client(df: pd.DataFrame, client_id: str) -> pd.DataFrame:
    """
    Client-360 counterpart to run_scenario. Where run_scenario ranks many
    clients against ONE scenario, this checks ONE client against EVERY
    scenario and returns only the ones that actually apply to them.

    "Ranked, top 10 of 120" has no meaning for a single person — the right
    question for one client is "does this news touch them, yes or no." A
    client with no long-duration debt and no IT exposure may correctly match
    zero scenarios; an empty result here is a true answer, not a bug.

    Reuses each scenario's own filter_fn rather than duplicating threshold
    logic — a client "matches" a scenario if they clear its own min_score,
    exactly the same bar a book-wide run would apply to them.
    """
    client_row = df[df["client_id"] == client_id]
    if client_row.empty:
        raise ValueError(f"No client found for client_id={client_id!r}.")

    results = []
    for name, scenario in SCENARIOS.items():
        matched = scenario["filter_fn"](client_row)
        if not matched.empty:
            row = matched.iloc[[0]].copy()
            row["headline"] = scenario["headline"]
            row["scenario_name"] = name
            results.append(row)

    if not results:
        return pd.DataFrame(columns=list(client_row.columns) +
                            ["exposure_score", "reason", "headline",
                             "scenario_name"])

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    # Manual smoke test — run directly to sanity-check every scenario
    # against every RM without needing Streamlit running.
    df = pd.read_csv("data/clients.csv")

    print(f"{'RM':<8}{'Scenario':<34}{'Matches':>8}{'Top score':>12}")
    for rm_id in sorted(df["rm_id"].unique()):
        for name in SCENARIOS:
            book = df[df["rm_id"] == rm_id]
            full_match = SCENARIOS[name]["filter_fn"](book)
            top = run_scenario(df, rm_id, name)
            top_score = f"{top['exposure_score'].iloc[0]:.1f}" if len(top) else "-"
            print(f"{rm_id:<8}{name:<34}{len(full_match):>8}{top_score:>12}")

    print("\nSample output — RM-06, IT sector selloff, top 5:")
    sample = run_scenario(df, "RM-06", "IT sector selloff", top_n=5)
    cols = ["client_id", "first_name", "top_sector_exposure",
            "alloc_direct_equity_pct", "exposure_score", "reason", "headline"]
    print(sample[cols].to_string(index=False))
