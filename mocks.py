"""
Mock inputs for Module 2 — lets you build & test the Investigator WITHOUT
waiting for Module 1 (Reader) or Module 3 (Economist).

Every profile below is a hand-built dict with the SAME keys as a customers.csv
row, so swapping in a real lookup_profile() later changes nothing.
"""

# A stand-in for Module 1's output. The Investigator only needs the keys to
# exist; trust does NOT come from this (it comes from history) — so the values
# here are irrelevant to the verdict by design.
MOCK_READER = {"issue_type": "Refund_Return", "frustration": "High", "confidence": 0.86}


def _profile(**overrides):
    """A clean, GENUINE baseline profile; override only what a case needs."""
    base = {
        "customer_id": "C0000",
        "account_age_months": 36,
        "loyalty_tier": "Gold",
        "lifetime_spend": 80000,
        "total_orders": 60,
        "total_complaints": 3,
        "total_refunds_received": 3,
        "refund_to_order_ratio": 0.05,
        "items_kept_after_refund": 0,
        "complaints_last_30_days": 1,
        "is_first_purchase": False,
        "order_value": 1500,
        "product_category": "Home_Goods",
        "is_perishable_or_hygiene": False,
        "resale_value": 600,
        "reverse_logistics_cost": 120,
        "prior_contacts_this_issue": 0,
        "prior_promise_logged": False,
        "customer_care_notes": "",
        "clv_estimate": 26000,
    }
    base.update(overrides)
    return base


# (name, profile, expected_genuineness, expected_claim_status)
CASES = [
    # ---- GENUINENESS cases ------------------------------------------------
    ("loyal_regular",
     _profile(account_age_months=40, total_orders=80, total_refunds_received=4,
              refund_to_order_ratio=0.05, items_kept_after_refund=0,
              complaints_last_30_days=1, loyalty_tier="Gold"),
     "GENUINE", "UNVERIFIED"),

    ("serial_refunder",
     _profile(account_age_months=8, total_orders=6, total_refunds_received=5,
              refund_to_order_ratio=0.83, items_kept_after_refund=3,
              complaints_last_30_days=5),
     "LIKELY_ABUSER", "UNVERIFIED"),

    # The loophole the handbook MISSES: keeps exactly 2 items (handbook used
    # kept>=3, so it would call this GENUINE/SUSPICIOUS). We catch it.
    ("keep_exactly_two_evader",
     _profile(account_age_months=18, total_orders=20, total_refunds_received=4,
              refund_to_order_ratio=0.20, items_kept_after_refund=2,
              complaints_last_30_days=2),
     "LIKELY_ABUSER", "UNVERIFIED"),

    ("newbie_chancer",
     _profile(account_age_months=0, total_orders=1, total_refunds_received=0,
              refund_to_order_ratio=0.0, items_kept_after_refund=0,
              complaints_last_30_days=1, is_first_purchase=True,
              order_value=45000, product_category="Electronics"),
     "SUSPICIOUS", "UNVERIFIED"),

    ("moderate_ratio_suspicious",
     _profile(account_age_months=12, total_orders=20, total_refunds_received=8,
              refund_to_order_ratio=0.40, items_kept_after_refund=1,
              complaints_last_30_days=2),
     "SUSPICIOUS", "UNVERIFIED"),

    # Sleeper: long tenure should NOT excuse fresh abuse signals.
    ("aged_sleeper_turned_abuser",
     _profile(account_age_months=60, total_orders=15, total_refunds_received=9,
              refund_to_order_ratio=0.60, items_kept_after_refund=2,
              complaints_last_30_days=4, loyalty_tier="Platinum"),
     "LIKELY_ABUSER", "UNVERIFIED"),

    # Complaint burst alone (ratio low, kept low) -> still abuse-level.
    ("complaint_burst_only",
     _profile(account_age_months=20, total_orders=30, total_refunds_received=2,
              refund_to_order_ratio=0.07, items_kept_after_refund=0,
              complaints_last_30_days=5),
     "LIKELY_ABUSER", "UNVERIFIED"),

    # ---- CLAIM-VERIFICATION cases ----------------------------------------
    ("false_promise_unverified",   # claims a promise, our records show nothing
     _profile(prior_promise_logged=False, prior_contacts_this_issue=0,
              customer_care_notes=""),
     "GENUINE", "UNVERIFIED"),

    ("confirmed_promise",          # notes + logged promise back the claim
     _profile(prior_promise_logged=True, prior_contacts_this_issue=2,
              customer_care_notes="Agent promised 20% coupon on previous call, not yet issued."),
     "GENUINE", "CONFIRMED"),

    ("contradicted_claim",         # notes say the opposite was agreed
     _profile(prior_promise_logged=False, prior_contacts_this_issue=1,
              customer_care_notes="Informed customer no refund applicable per policy; customer agreed."),
     "GENUINE", "CONTRADICTED"),

    # ADVERSARIAL: note CONTAINS the token 'promised' but in a NEGATED context.
    # A naive substring search would wrongly return CONFIRMED. We return CONTRADICTED.
    ("negation_trap_note",
     _profile(prior_promise_logged=False, prior_contacts_this_issue=1,
              customer_care_notes="Customer claims we promised a refund but we never promised anything."),
     "GENUINE", "CONTRADICTED"),

    # PHANTOM PROMISE: affirmative wording but ZERO contact records to back it.
    ("phantom_promise_no_contact",
     _profile(prior_promise_logged=False, prior_contacts_this_issue=0,
              customer_care_notes="promised"),
     "GENUINE", "UNVERIFIED"),

    # Abuser WITH a confirmed promise — we report BOTH facts truthfully and let
    # Module 3 decide whether to honour a promise made to an abuser.
    ("abuser_with_confirmed_promise",
     _profile(account_age_months=6, total_orders=8, total_refunds_received=5,
              refund_to_order_ratio=0.62, items_kept_after_refund=3,
              complaints_last_30_days=4, prior_promise_logged=True,
              prior_contacts_this_issue=2,
              customer_care_notes="Prior agent committed to a refund, pending processing."),
     "LIKELY_ABUSER", "CONFIRMED"),

    # Robustness: messy/missing fields (strings, None) must not crash.
    ("messy_fields_robustness",
     {"customer_id": "Cxxxx", "refund_to_order_ratio": "0.9",
      "items_kept_after_refund": None, "complaints_last_30_days": "2",
      "account_age_months": "", "total_orders": "3", "total_refunds_received": "2",
      "is_first_purchase": "True", "prior_promise_logged": "False",
      "prior_contacts_this_issue": None, "customer_care_notes": None},
     "SUSPICIOUS", "UNVERIFIED"),  # ratio 0.9 but only 3 orders -> low-denominator guard
]
