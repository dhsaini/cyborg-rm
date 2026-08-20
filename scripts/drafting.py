"""
drafting.py — message drafting for Cyborg RM.

Scope, per PRD §8: this module decides HOW to phrase a message. It never
decides WHO receives one or WHETHER they should — that decision was already
made by filters.py before a client row ever reaches this module. The
"reason" string passed in comes from deterministic Python, not the model.

Compliance is enforced twice, deliberately:
  1. The system prompt instructs the model what not to do.
  2. check_compliance() independently re-checks the actual output afterward.
Point 1 alone is not trusted — models drift from instructions under real
traffic. Point 2 is what makes the difference between "we told it not to"
and "we verified it didn't."
"""

import re

import streamlit as st
from google import genai

MODEL_NAME = "gemini-3.6-flash"  # verify at build time — see PRD/TR-7

SYSTEM_PROMPT = """You are drafting a short WhatsApp message for a wealth \
relationship manager to send to one client. You are given the client's \
first name, the actual news headline driving this outreach, and a specific \
fact about how their portfolio connects to it. Write ONLY the message text \
— no preamble, no headers, no signature.

The message must do three things, in this order:
1. Reference what actually happened in the news — specifically enough that \
the client would recognise the event, not a vague gesture at "market \
conditions."
2. Name the specific fact about their own portfolio that makes this \
relevant to THEM — the exact holding, sector, or allocation given to you. \
Do not generalise it into a category the client would not recognise as \
their own.
3. Invite a conversation — a call or a meeting — without concluding one.

Hard rules, no exceptions:
- 2 to 4 sentences. Plain conversational text, no bullet points, no bold.
- Never name a specific fund, scheme, AMC, or issuer. If a product category \
is mentioned, use the generic category name given to you, nothing more \
specific.
- Never state or imply a return, yield, or performance figure of any kind.
- Never use the words "buy", "sell", "invest in", "should", or "recommend".
- Never guarantee or predict any market outcome.
- The message must invite a conversation, not conclude one. End by \
suggesting a call or a meeting, not a decision.
- Do not fabricate any fact not given to you. If a number is not given to \
you, do not introduce one.

A message that could apply to any client regardless of their actual \
holdings has failed the task, even if every hard rule above is technically \
satisfied. Specificity to this client's real exposure is the entire point.

Write only the message. Nothing else."""

# Independent of the prompt above — checked against the actual output.
# Kept intentionally small and literal, so it's auditable at a glance
# rather than a black-box "safety" pass.
BLOCKED_PATTERNS = [
    r"\bbuy\b", r"\bsell\b", r"\binvest in\b", r"\bshould\b", r"\brecommend",
    r"\bguarantee", r"\breturn[s]?\s+of\b", r"\d+(\.\d+)?\s?%",
    r"\bAMC\b", r"\bNAV\b", r"\bNFO\b",
]


def check_compliance(draft: str) -> tuple[bool, str | None]:
    """
    Re-checks generated text against the blocklist, independent of whatever
    the model was instructed to do. Returns (is_clean, matched_pattern).

    This is intentionally simple regex, not another LLM call — a second
    model checking a first model's compliance is still just a model, and
    cannot be the thing standing between a client and a mis-selling
    complaint. A literal, readable blocklist can be audited by a human in
    thirty seconds; that auditability is the point, not a limitation.
    """
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, draft, re.IGNORECASE):
            return False, pattern
    return True, None


def draft_message(first_name: str, reason: str, headline: str = "",
                  language: str = "English",
                  max_attempts: int = 2) -> tuple[str | None, str]:
    """
    Generates one draft message. Returns (message, status).

    headline is the actual news event triggering this outreach (e.g. "RBI
    cuts repo rate by 25bps"). Passing "" is valid for Mode 2 (product
    pitches, which have no news event) — the prompt adapts accordingly.

    status is one of: "ok", "blocked", "error" — the caller decides how to
    surface each to the RM. On "blocked", message is None: a flagged draft
    is never shown, not even with a warning label, because a wealth RM
    seeing a redacted-but-visible non-compliant draft could still act on it.
    """
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    if headline:
        context_block = (
            f"News headline driving this outreach: {headline}\n"
            f"How this client's portfolio connects to it: {reason}\n"
        )
    else:
        context_block = (
            f"Reason this message is relevant (no specific news event — "
            f"this is a proactive product pitch, not a market reaction): "
            f"{reason}\n"
        )

    user_prompt = (
        f"Client first name: {first_name}\n"
        f"{context_block}"
        f"Preferred language for the message: {language}\n"
        "Write the message now."
    )

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config={"system_instruction": SYSTEM_PROMPT},
            )
        except Exception as exc:
            return None, f"error: {exc}"

        draft = (response.text or "").strip()
        is_clean, matched = check_compliance(draft)

        if is_clean and draft:
            return draft, "ok"

        # One retry with a sharper instruction, citing exactly what tripped.
        user_prompt += (
            f"\n\nYour previous attempt was rejected for containing "
            f"disallowed content matching '{matched}'. Rewrite it, "
            f"following the rules exactly."
        )

    return None, "blocked"


if __name__ == "__main__":
    # Offline test of the compliance checker only — no API call, no cost.
    # Confirms the blocklist actually catches what it should before this
    # module is ever wired to a real Gemini call.
    test_cases = [
        ("You might want to buy this fund before it closes.", False),
        ("This fund has delivered 18% returns this year.", False),
        ("I recommend you invest in this NFO today.", False),
        ("The AI-themed fund's NAV rose sharply.", False),
        ("Given the recent RBI move, your long-duration debt allocation "
         "may be worth a conversation. Would you be free for a quick call "
         "this week?", True),
        ("Given rising sector momentum, worth discussing your exposure. "
         "Let's connect this week?", True),
    ]

    print("Compliance checker self-test (no API calls):\n")
    all_passed = True
    for text, expected_clean in test_cases:
        is_clean, matched = check_compliance(text)
        status = "PASS" if is_clean == expected_clean else "FAIL"
        all_passed &= (status == "PASS")
        flag = f" (matched: {matched})" if matched else ""
        print(f"[{status}] clean={is_clean:<5}{flag}\n       \"{text}\"\n")

    print("ALL PASSED" if all_passed else "SOME FAILED — check above")
