"""
generate_clients.py — synthetic client dataset for Cyborg RM.

Deterministic: the same SEED always produces the identical CSV.
All data is fabricated. No real client, fund, or issuer is represented.

Run from the repo root:   python scripts/generate_clients.py
Output:                   data/clients.csv
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

CR = 1_00_00_000      # 1 crore
L = 1_00_000          # 1 lakh

PMS_MINIMUM = 50 * L  # SEBI minimum ticket for PMS
AIF_MINIMUM = 1 * CR  # SEBI minimum ticket for AIF

OUTPUT_PATH = "data/clients.csv"


# --------------------------------------------------------------------------
# Desk configuration — book sizes and client sizes both vary, as on a real desk
# --------------------------------------------------------------------------

RM_CONFIG = [
    # rm_id,   desk,         clients, median AUM (Cr), unlisted rate, SIP rate
    ("RM-01", "UHNI",            55,  18.0, 0.35, 0.12),
    ("RM-02", "UHNI",            62,  15.0, 0.35, 0.12),
    ("RM-03", "HNI Senior",      88,   6.0, 0.12, 0.30),
    ("RM-04", "HNI",            105,   4.0, 0.12, 0.30),
    ("RM-05", "HNI",            112,   3.5, 0.12, 0.30),
    ("RM-06", "HNI",            120,   3.0, 0.12, 0.30),
    ("RM-07", "HNI",             98,   4.5, 0.12, 0.30),
    ("RM-08", "Emerging",       145,   1.8, 0.03, 0.55),
    ("RM-09", "Emerging",       135,   2.0, 0.03, 0.55),
    ("RM-10", "Wealth",          80,   5.0, 0.12, 0.30),
]

FIRST_NAMES = [
    "Aarav", "Aditi", "Akash", "Ananya", "Arjun", "Bhavna", "Chirag", "Deepak",
    "Devika", "Gaurav", "Harsh", "Ishaan", "Jaya", "Kabir", "Kavita", "Kiran",
    "Lakshmi", "Manish", "Meera", "Mohit", "Naveen", "Neha", "Nikhil", "Pallavi",
    "Parth", "Pooja", "Prakash", "Priya", "Rahul", "Rajesh", "Rakesh", "Ramesh",
    "Rohan", "Rupal", "Sanjay", "Shalini", "Shreya", "Siddharth", "Sneha",
    "Sunil", "Swati", "Tanvi", "Uday", "Varun", "Vidya", "Vikram", "Vinod", "Yash",
]

SECTORS = ["Diversified", "BFSI", "IT", "Infra", "Pharma", "Auto", "FMCG"]
SECTOR_WEIGHTS = [0.30, 0.18, 0.14, 0.10, 0.10, 0.09, 0.09]

# One dominant name, as in real unlisted trading — also what makes an IPO
# scenario hit enough clients within a single book to be worth demonstrating.
UNLISTED_TAGS = ["Unlisted Co. A", "Unlisted Co. B", "Unlisted Co. C",
                 "Unlisted Co. D", "Unlisted Co. E"]
UNLISTED_WEIGHTS = [0.55, 0.18, 0.12, 0.09, 0.06]

# Target allocation by risk profile, in the column order used throughout.
ALLOC_COLUMNS = [
    "alloc_direct_equity_pct", "alloc_equity_mf_pct", "alloc_debt_mf_bonds_pct",
    "alloc_pms_pct", "alloc_aif_pct", "alloc_unlisted_pct", "alloc_gold_pct",
]

BASE_ALLOCATION = {
    "Conservative": [10, 22, 45, 8, 3, 2, 10],
    "Moderate":     [20, 25, 22, 15, 8, 5, 5],
    "Aggressive":   [28, 24, 12, 18, 10, 6, 2],
}

RISK_WEIGHTS = {
    "UHNI":       [0.20, 0.45, 0.35],
    "HNI Senior": [0.28, 0.47, 0.25],
    "HNI":        [0.30, 0.48, 0.22],
    "Wealth":     [0.30, 0.48, 0.22],
    "Emerging":   [0.35, 0.48, 0.17],
}

WEALTH_SOURCES = ["Business Owner", "Salaried CXO", "Professional",
                  "Inherited", "Promoter"]
WEALTH_WEIGHTS = {
    "UHNI":       [0.38, 0.12, 0.10, 0.20, 0.20],
    "HNI Senior": [0.32, 0.24, 0.20, 0.16, 0.08],
    "HNI":        [0.30, 0.28, 0.22, 0.15, 0.05],
    "Wealth":     [0.30, 0.28, 0.22, 0.15, 0.05],
    "Emerging":   [0.22, 0.40, 0.28, 0.08, 0.02],
}


def draw_allocations(risk_profiles):
    """
    Draw allocations that vary per client but average to the risk-profile target.

    A Dirichlet distribution is used because it produces vectors that sum to
    exactly 1 by construction — so allocations always total 100% with no
    rescaling. The multiplier controls spread: higher means tighter clustering
    around the target.
    """
    out = np.zeros((len(risk_profiles), len(ALLOC_COLUMNS)))
    for i, profile in enumerate(risk_profiles):
        target = np.array(BASE_ALLOCATION[profile], dtype=float)
        out[i] = rng.dirichlet(target * 0.45) * 100
    return out


def build_book(rm_id, desk, n_clients, median_aum_cr, unlisted_rate, sip_rate,
               start_index):
    """Generate one RM's book."""

    # AUM: lognormal, so the median lands on the desk's target and the tail
    # runs long — a few very large relationships, as on a real book.
    aum = median_aum_cr * CR * rng.lognormal(mean=0.0, sigma=0.55, size=n_clients)
    aum = np.clip(aum, 1 * CR, 200 * CR).round(-5)  # nearest lakh

    risk = rng.choice(["Conservative", "Moderate", "Aggressive"],
                      size=n_clients, p=RISK_WEIGHTS[desk])

    alloc = draw_allocations(risk)
    alloc_df = pd.DataFrame(alloc, columns=ALLOC_COLUMNS)

    # --- Unlisted: a holding most clients simply do not have ----------------
    holds_unlisted = rng.random(n_clients) < unlisted_rate
    alloc_df.loc[~holds_unlisted, "alloc_unlisted_pct"] = 0.0

    unlisted_tag = np.where(
        holds_unlisted,
        rng.choice(UNLISTED_TAGS, size=n_clients, p=UNLISTED_WEIGHTS),
        "No holding",  # pandas reads the literal strings "None" and "" back as NaN
    )

    # --- Regulatory floors --------------------------------------------------
    # A percentage allocation is only permissible if the resulting ticket
    # clears the SEBI minimum. Below it, the client cannot hold the product
    # at all — so the allocation is zeroed rather than reduced.
    pms_value = alloc_df["alloc_pms_pct"] / 100 * aum
    alloc_df.loc[pms_value < PMS_MINIMUM, "alloc_pms_pct"] = 0.0

    aif_value = alloc_df["alloc_aif_pct"] / 100 * aum
    alloc_df.loc[aif_value < AIF_MINIMUM, "alloc_aif_pct"] = 0.0

    # Freed percentage goes back to the liquid buckets, 60/40 equity to debt.
    freed = 100 - alloc_df[ALLOC_COLUMNS].sum(axis=1)
    alloc_df["alloc_equity_mf_pct"] += freed * 0.6
    alloc_df["alloc_debt_mf_bonds_pct"] += freed * 0.4

    alloc_df = alloc_df.round(1)
    # Rounding leaves a small residual; absorb it in the largest bucket so the
    # row still totals exactly 100.
    residual = 100 - alloc_df[ALLOC_COLUMNS].sum(axis=1)
    largest = alloc_df[ALLOC_COLUMNS].idxmax(axis=1)
    for i in range(len(alloc_df)):
        col = largest.iloc[i]
        alloc_df.loc[alloc_df.index[i], col] = round(
            alloc_df.loc[alloc_df.index[i], col] + residual.iloc[i], 1
        )

    # --- Exposure hooks -----------------------------------------------------
    sector = rng.choice(SECTORS, size=n_clients, p=SECTOR_WEIGHTS)

    duration = rng.choice(["Short", "Medium", "Long"], size=n_clients,
                          p=[0.42, 0.36, 0.22])
    duration = np.where(alloc_df["alloc_debt_mf_bonds_pct"] < 5, "None", duration)

    has_intl = rng.random(n_clients) < 0.18
    intl = np.where(has_intl, rng.uniform(1, 8, n_clients).round(1), 0.0)

    has_thematic = rng.random(n_clients) < 0.25
    thematic = np.where(has_thematic, rng.uniform(2, 20, n_clients).round(1), 0.0)

    # --- Deployment behaviour ----------------------------------------------
    sip_active = rng.random(n_clients) < sip_rate
    sip_amount = np.where(
        sip_active,
        (aum * rng.uniform(0.0005, 0.0025, n_clients) / 5000).round() * 5000,
        0,
    )

    lumpsum_ticket = (aum * rng.uniform(0.03, 0.09, n_clients) / L).round() * L
    days_since_lumpsum = rng.integers(5, 420, n_clients)
    # Idle funds: most clients hold little, a few hold a lot.
    ledger = (aum * rng.exponential(0.004, n_clients) / L).round() * L
    ledger = np.minimum(ledger, aum * 0.05).round(-3)

    return pd.DataFrame({
        "client_id": [f"C-{start_index + i:04d}" for i in range(n_clients)],
        "rm_id": rm_id,
        "desk": desk,
        "first_name": rng.choice(FIRST_NAMES, size=n_clients),
        "age": np.clip(rng.normal(48, 9, n_clients), 32, 68).round().astype(int),
        "city_tier": rng.choice(["Tier 1", "Tier 2"], size=n_clients, p=[0.7, 0.3]),
        "segment": np.where(aum >= 10 * CR, "UHNI", "HNI"),
        "wealth_source": rng.choice(WEALTH_SOURCES, size=n_clients,
                                    p=WEALTH_WEIGHTS[desk]),
        "risk_profile": risk,
        "preferred_language": rng.choice(
            ["English", "Hindi", "Gujarati", "Marathi"],
            size=n_clients, p=[0.55, 0.20, 0.13, 0.12]),
        "relationship_tenure_years": rng.integers(1, 19, n_clients),
        "days_since_last_contact": rng.integers(1, 400, n_clients),
        "total_aum_inr": aum.astype("int64"),
        **{col: alloc_df[col].to_numpy() for col in ALLOC_COLUMNS},
        "top_sector_exposure": sector,
        "debt_duration_bucket": duration,
        "unlisted_holding_tag": unlisted_tag,
        "intl_equity_exposure_pct": intl,
        "existing_thematic_exposure_pct": thematic,
        "sip_active": sip_active,
        "sip_monthly_inr": sip_amount.astype("int64"),
        "avg_lumpsum_ticket_inr": lumpsum_ticket.astype("int64"),
        "days_since_last_lumpsum": days_since_lumpsum,
        "ledger_balance_inr": ledger.astype("int64"),
    })


def main():
    books = []
    index = 1
    for rm_id, desk, n, aum_cr, unlisted_rate, sip_rate in RM_CONFIG:
        books.append(build_book(rm_id, desk, n, aum_cr, unlisted_rate,
                                sip_rate, index))
        index += n

    df = pd.concat(books, ignore_index=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} clients to {OUTPUT_PATH}\n")
    report(df)


def report(df):
    """Verification summary — confirms the dataset is demonstrable, not just valid."""

    totals = df[ALLOC_COLUMNS].sum(axis=1).round(1)
    print(f"Allocations sum to 100:  {bool((totals == 100.0).all())}")

    pms = df.loc[df["alloc_pms_pct"] > 0, "alloc_pms_pct"] / 100 * \
        df.loc[df["alloc_pms_pct"] > 0, "total_aum_inr"]
    aif = df.loc[df["alloc_aif_pct"] > 0, "alloc_aif_pct"] / 100 * \
        df.loc[df["alloc_aif_pct"] > 0, "total_aum_inr"]
    print(f"All PMS tickets >= 50L:  {bool((pms >= PMS_MINIMUM).all())}")
    print(f"All AIF tickets >= 1Cr:  {bool((aif >= AIF_MINIMUM).all())}")

    print("\nAIF holders by desk (should concentrate in UHNI):")
    aif_by_desk = df.assign(has_aif=df["alloc_aif_pct"] > 0) \
                    .groupby("desk", observed=True)["has_aif"].agg(["sum", "count"])
    for desk, row in aif_by_desk.iterrows():
        pct = 100 * row["sum"] / row["count"]
        print(f"  {desk:<12} {int(row['sum']):>4} / {int(row['count']):<4} ({pct:4.1f}%)")

    print("\nScenario hit counts per RM (target: 8-15):")
    scenarios = {
        "Rate move (Long duration)":
            (df["debt_duration_bucket"] == "Long") & (df["alloc_debt_mf_bonds_pct"] >= 25),
        "IT selloff":
            (df["top_sector_exposure"] == "IT") & (df["alloc_direct_equity_pct"] >= 15),
        "Rupee/gold (intl or gold)":
            (df["intl_equity_exposure_pct"] >= 3) | (df["alloc_gold_pct"] >= 15),
        "IPO (Unlisted Co. A)":
            df["unlisted_holding_tag"] == "Unlisted Co. A",
    }
    header = "  " + "RM".ljust(8) + "".join(s[:14].rjust(16) for s in scenarios)
    print(header)
    for rm_id in df["rm_id"].unique():
        book = df[df["rm_id"] == rm_id]
        counts = [int(mask[df["rm_id"] == rm_id].sum()) for mask in scenarios.values()]
        print("  " + rm_id.ljust(8) + "".join(str(c).rjust(16) for c in counts))


if __name__ == "__main__":
    main()
