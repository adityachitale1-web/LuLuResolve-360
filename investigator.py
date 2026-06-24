"""
LuluCare 360 — MODULE 2: THE INVESTIGATOR (the trust engine)
============================================================

ONE JOB: decide whether we can TRUST the customer, and whether OUR OWN RECORDS
back up any claim they make. Output is two labels + a human-readable reason.

    can-we-trust      -> genuineness : GENUINE / SUSPICIOUS / LIKELY_ABUSER
    do-records-agree  -> claim_status: CONFIRMED / CONTRADICTED / UNVERIFIED

This module does NOT decide payouts, coupons or refunds — that is Module 3
(the Economist). It only reports trust + claim facts. Staying in this lane is
what keeps the final stitch clean.

DESIGN PRINCIPLE (the anti-manipulation backbone):
    "Never trust an unverifiable claim from the person who benefits from it,
     and never let the loudness of a message substitute for evidence."
    The message says what the customer *claims*; the records say what *happened*.
    Where they disagree, the records win. Trust therefore comes from HISTORY,
    never from the text — so this module reads the profile, not the message body.

--------------------------------------------------------------------------------
CONTRACT (stitch-ready)
--------------------------------------------------------------------------------
    investigate(reader_output, profile) -> dict

    reader_output : Module 1's output  {'issue_type','frustration','confidence'}
    profile       : one customer row from customers.csv as a dict
    returns       : {
        'genuineness' : 'GENUINE' | 'SUSPICIOUS' | 'LIKELY_ABUSER',  # REQUIRED
        'claim_status': 'CONFIRMED' | 'CONTRADICTED' | 'UNVERIFIED',  # REQUIRED
        'reason'      : str,                                          # REQUIRED
        # --- additive diagnostics (Module 3 may ignore these safely) ---
        'risk_score'  : int 0..100,
        'risk_flags'  : [str, ...],
    }

The three REQUIRED keys match the handbook contract exactly. The extra keys are
purely additive transparency for the demo / Module 3 / human review and do not
break any downstream consumer that only reads the three required keys.
"""

import re

# ----------------------------------------------------------------------
# THRESHOLDS  (derived from the dataset's archetype boundaries, see README)
# ----------------------------------------------------------------------
# Why these numbers: in generate_data.py the archetypes are generated as
#   GENUINE      : ratio <= 0.10, kept in {0,1}, comp_30 <= 2, age >= 6, orders >= 15
#   SUSPICIOUS   : ratio 0.25-0.45 (moderate)  OR  age <= 2 / orders 1-4 (new)
#   LIKELY_ABUSER: ratio 0.5-0.9, kept in {2,3,4,5}, comp_30 in {3..6}
# So ratio >= 0.5 OR kept >= 2 each cleanly separate abusers with ZERO overlap
# into genuine/suspicious. (Note: the handbook's kept >= 3 LEAKS — it lets an
# abuser who keeps exactly 2 items slip through. We use kept >= 2.)
ABUSER_RATIO        = 0.50   # refund-to-order ratio at/above this = abuse
ABUSER_KEPT         = 2      # items kept after refund (handbook used 3 -> loophole)
ABUSER_BURST        = 4      # complaints in last 30 days

SUSPECT_RATIO       = 0.25   # moderate refund ratio
SUSPECT_BURST       = 3      # a rising complaint burst
NEW_ACCOUNT_MONTHS  = 2      # brand-new accounts carry higher fraud risk
LOW_ORDER_VOLUME    = 5      # too few orders for the ratio to be trustworthy

# Soft-flag (human-review) thresholds — these DO NOT change the label by
# themselves, so a genuine high-volume customer is never auto-denied.
DILUTION_ABS_REFUNDS = 8     # many refunds in absolute terms ...
DILUTION_MAX_RATIO   = 0.25  # ... while the ratio is kept artificially low
CHRONIC_COMPLAINT_RATIO = 0.6  # complains on most orders (time-cost, not fraud)


# ----------------------------------------------------------------------
# SAFE FIELD ACCESS  (defensive: upstream data may be missing/str/None)
# ----------------------------------------------------------------------
def _num(profile, key, default=0.0):
    """Read a numeric field robustly; bad/missing -> default."""
    try:
        v = profile.get(key, default)
        if v is None or v == "":
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _bool(profile, key, default=False):
    """Read a boolean field robustly. Accepts True/False, 'True'/'False', 1/0."""
    v = profile.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y")
    if isinstance(v, (int, float)):
        return v != 0
    return default


# ======================================================================
# CHECK 1 — GENUINENESS  (are they gaming us?)  — from HISTORY, not text
# ======================================================================
def assess_genuineness(profile):
    """
    Returns (label, flags). Hard rules drive the label; abuse signals are
    evaluated FIRST and IGNORE tenure — a long-standing account that has turned
    abusive (the 'sleeper') is not given a free pass by its age.
    """
    ratio   = _num(profile, "refund_to_order_ratio")
    kept    = _num(profile, "items_kept_after_refund")
    burst   = _num(profile, "complaints_last_30_days")
    age     = _num(profile, "account_age_months")
    orders  = _num(profile, "total_orders")
    refunds = _num(profile, "total_refunds_received")
    complaints = _num(profile, "total_complaints")
    first   = _bool(profile, "is_first_purchase")

    flags = []

    # --- ABUSE signals (checked first; tenure cannot override these) -----
    # A high refund ratio only counts as ABUSE when the denominator is large
    # enough to be meaningful. On a tiny order count the ratio is statistical
    # noise (1 refund / 2 orders = 0.50), and branding a 2-order account a
    # confirmed serial abuser is exactly the false accusation the project warns
    # against. Such low-volume-high-ratio cases fall through to SUSPICIOUS below
    # (capped + verified). Kept-items and complaint bursts still flag abuse at
    # ANY order count, so no genuine abuser escapes through a low denominator.
    ratio_is_abuse = ratio >= ABUSER_RATIO and orders >= LOW_ORDER_VOLUME
    if ratio_is_abuse:
        flags.append("high_refund_ratio(%.2f)" % ratio)
    if kept >= ABUSER_KEPT:
        flags.append("kept_items_after_refund(%d)" % int(kept))
    if burst >= ABUSER_BURST:
        flags.append("complaint_burst_30d(%d)" % int(burst))

    is_abuser = ratio_is_abuse or (kept >= ABUSER_KEPT) or (burst >= ABUSER_BURST)

    # --- SUSPICION signals ---------------------------------------------
    if ratio >= SUSPECT_RATIO:
        flags.append("moderate_refund_ratio(%.2f)" % ratio)
    if age <= NEW_ACCOUNT_MONTHS:
        flags.append("new_account(%dmo)" % int(age))
    if first:
        flags.append("first_purchase")
    if orders < LOW_ORDER_VOLUME:
        flags.append("low_order_volume(%d)" % int(orders))
    if burst >= SUSPECT_BURST:
        flags.append("rising_complaints_30d(%d)" % int(burst))
    # low volume + any refund: the ratio is statistically meaningless on a tiny
    # denominator, so a refund this early is itself a yellow flag.
    if orders < 8 and refunds >= 1:
        flags.append("early_refunder")

    is_suspicious = (
        ratio >= SUSPECT_RATIO
        or age <= NEW_ACCOUNT_MONTHS
        or first
        or orders < LOW_ORDER_VOLUME
        or burst >= SUSPECT_BURST
        or (orders < 8 and refunds >= 1)
    )

    # --- SOFT flags (human-review only; do NOT change the label) ---------
    # Ratio-dilution attack: place many cheap orders to push the ratio down
    # while still extracting many refunds in absolute terms. We do NOT auto-deny
    # on this (a genuine high-volume shopper would be wrongly punished) — we flag
    # it for Module 3 / a human to glance at.
    if refunds >= DILUTION_ABS_REFUNDS and ratio < DILUTION_MAX_RATIO:
        flags.append("possible_ratio_dilution(refunds=%d)" % int(refunds))
    # Chronic complainer: complains on most orders but rarely gets refunds — a
    # time-cost, not a fraud signal. Surface it, don't penalise trust for it.
    if orders > 0 and (complaints / orders) >= CHRONIC_COMPLAINT_RATIO and ratio < SUSPECT_RATIO:
        flags.append("chronic_complainer")

    if is_abuser:
        return "LIKELY_ABUSER", flags
    if is_suspicious:
        return "SUSPICIOUS", flags
    return "GENUINE", flags


# ======================================================================
# CHECK 2 — CLAIM VERIFICATION  (cross-examine against OUR records)
# ======================================================================
# Negation-aware phrase matching. We check CONTRADICTION patterns BEFORE
# CONFIRMATION patterns, because a naive substring search for "promised" is
# trivially fooled by a note like "we never promised anything" (contains the
# token 'promise'). Records win, and a contradicted/negated promise must not be
# read as a confirmation.

_CONTRADICT_PATTERNS = [
    r"\bno\s+refund\b",
    r"\bnon[-\s]?returnable\b",
    r"\bnot\s+returnable\b",
    r"\bno\s+(prior\s+)?promise\b",
    r"\bnever\s+promised\b",
    r"\bdid\s+not\s+promise\b",
    r"\bno\s+promise\s+was\s+made\b",
    r"\b(refund\s+)?denied\b",
    r"\bdeclined\b",
    r"\bnot\s+eligible\b",
    r"\bineligible\b",
]

_AFFIRM_PATTERNS = [
    r"\bpromised\b",
    r"\bassured\b",
    r"\bcommitted\b",
    r"\bguarantee(d)?\b",
    r"\bwill\s+refund\b",
    r"\bagreed\s+to\s+refund\b",
    r"\breplacement\b",
]


def _matches_any(text, patterns):
    return any(re.search(p, text) for p in patterns)


def verify_claim(profile):
    """
    Returns (claim_status, flags). Decision order is deliberate:
      1. CONTRADICTED  — our notes refute the claim (negation-aware, checked first)
      2. CONFIRMED     — a logged promise OR an affirmative note, BUT only if there
                         is a contact record to attach it to (anti 'phantom promise')
      3. UNVERIFIED    — no record of the claimed contact -> neither honour nor accuse
    """
    notes = str(profile.get("customer_care_notes", "") or "").lower()
    promise_logged = _bool(profile, "prior_promise_logged")
    contacts = _num(profile, "prior_contacts_this_issue")

    flags = []

    # 1) CONTRADICTION first — negation-aware, beats any stray affirmative token.
    if _matches_any(notes, _CONTRADICT_PATTERNS):
        flags.append("records_contradict_claim")
        return "CONTRADICTED", flags

    # 2) CONFIRMATION — needs corroboration AND a contact to hang it on.
    affirmative = promise_logged or _matches_any(notes, _AFFIRM_PATTERNS)
    if affirmative:
        if promise_logged or contacts >= 1:
            flags.append("records_confirm_claim")
            return "CONFIRMED", flags
        # Promise-like wording but ZERO contact records to corroborate it:
        # treat as a phantom promise, do not auto-honour.
        flags.append("phantom_promise_no_contact_record")
        return "UNVERIFIED", flags

    # 3) No record at all.
    if contacts == 0 and not notes.strip():
        flags.append("no_record_of_contact")
    else:
        flags.append("no_supporting_evidence")
    return "UNVERIFIED", flags


# ======================================================================
# RISK SCORE  (additive transparency — a 0..100 number for the demo UI)
# ======================================================================
def _risk_score(genuineness, claim_status, flags):
    base = {"GENUINE": 5, "SUSPICIOUS": 45, "LIKELY_ABUSER": 90}[genuineness]
    if claim_status == "CONTRADICTED":
        base += 8
    soft = {"possible_ratio_dilution", "chronic_complainer", "phantom_promise_no_contact_record"}
    for f in flags:
        name = f.split("(")[0]
        base += 1 if name in soft else 2
    return max(0, min(100, int(base)))


# ======================================================================
# THE CONTRACT FUNCTION  (this is what Module 3 / the stitch calls)
# ======================================================================
def investigate(reader_output, profile):
    """
    reader_output : {'issue_type','frustration','confidence'}  (Module 1)
    profile       : customer row dict (customers.csv)
    -> verdict dict (see module docstring)
    """
    if profile is None:
        return {
            "genuineness": "SUSPICIOUS",
            "claim_status": "UNVERIFIED",
            "reason": "No customer profile found — cannot establish history; treat with caution.",
            "risk_score": 50,
            "risk_flags": ["unknown_customer"],
        }

    genuineness, g_flags = assess_genuineness(profile)
    claim_status, c_flags = verify_claim(profile)
    flags = g_flags + c_flags
    score = _risk_score(genuineness, claim_status, flags)

    ratio = _num(profile, "refund_to_order_ratio")
    orders = int(_num(profile, "total_orders"))
    kept = int(_num(profile, "items_kept_after_refund"))

    reason = (
        "%s (refund ratio %.2f over %d orders, %d item(s) kept after refund); "
        "claim is %s. Signals: %s."
        % (genuineness, ratio, orders, kept, claim_status,
           ", ".join(flags) if flags else "none")
    )

    return {
        "genuineness": genuineness,   # REQUIRED — contract
        "claim_status": claim_status, # REQUIRED — contract
        "reason": reason,             # REQUIRED — contract
        "risk_score": score,          # additive
        "risk_flags": flags,          # additive
    }


if __name__ == "__main__":
    demo_reader = {"issue_type": "Refund_Return", "frustration": "High", "confidence": 0.88}
    demo_profile = {
        "customer_id": "C9999", "refund_to_order_ratio": 0.83, "items_kept_after_refund": 3,
        "complaints_last_30_days": 5, "account_age_months": 4, "total_orders": 6,
        "total_refunds_received": 5, "total_complaints": 6, "is_first_purchase": False,
        "prior_promise_logged": False, "prior_contacts_this_issue": 0, "customer_care_notes": "",
    }
    from pprint import pprint
    pprint(investigate(demo_reader, demo_profile))
