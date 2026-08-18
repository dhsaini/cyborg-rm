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


SCENARIOS = {
    "RBI rate move": scenario_rbi_rate_move,
    "IT sector selloff": scenario_it_sector_selloff,
    "Rupee depreciation / gold rally": scenario_rupee_gold,
    "IPO filing (unlisted holding)": scenario_ipo_filing,
}


def run_scenario(df: pd.DataFrame, rm_id: str, scenario_name: str,
                  top_n: int = DEFAULT_TOP_N, **kwargs) -> pd.DataFrame:
    """
    Main entry point. Filters to one RM's book, runs the named scenario,
    returns at most top_n clients ranked by exposure.

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

    matched = SCENARIOS[scenario_name](book, **kwargs) if kwargs else \
        SCENARIOS[scenario_name](book)

    return matched.head(top_n)


if __name__ == "__main__":
    # Manual smoke test — run directly to sanity-check every scenario
    # against every RM without needing Streamlit running.
    df = pd.read_csv("data/clients.csv")

    print(f"{'RM':<8}{'Scenario':<34}{'Matches':>8}{'Top score':>12}")
    for rm_id in sorted(df["rm_id"].unique()):
        for name in SCENARIOS:
            book = df[df["rm_id"] == rm_id]
            full_match = SCENARIOS[name](book)
            top = run_scenario(df, rm_id, name)
            top_score = f"{top['exposure_score'].iloc[0]:.1f}" if len(top) else "-"
            print(f"{rm_id:<8}{name:<34}{len(full_match):>8}{top_score:>12}")

    print("\nSample output — RM-06, IT sector selloff, top 5:")
    sample = run_scenario(df, "RM-06", "IT sector selloff", top_n=5)
    cols = ["client_id", "first_name", "top_sector_exposure",
            "alloc_direct_equity_pct", "exposure_score", "reason"]
    print(sample[cols].to_string(index=False))
