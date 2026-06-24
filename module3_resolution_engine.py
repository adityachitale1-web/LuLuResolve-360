"""
LuluCare 360 — Module 3: Fair Resolution & Business Protection Engine ("The Economist")
=======================================================================================

Transparent, rule-based, explainable resolution policy. Consumes the Reader's
reading (Module 1) and the Investigator's verdict (Module 2) plus the customer
profile, and emits the exact decision contract Module 4 (The Voice) consumes.

Design contract (immutable — Module 4 depends on these keys):
    action            : ACKNOWLEDGE | COUPON | WALLET_CREDIT | REFUND | ESCALATE
    refund_type       : PICKUP | KEEP_ITEM | NONE
    coupon_percent    : 0 | 20 | 50
    wallet_credit     : float (USD; 0 when not applicable)
    escalate          : bool
    email_trigger     : bool   (True ONLY for COUPON / WALLET_CREDIT / REFUND)
    risk_score        : int    (0..100, fraud/business risk)
    resolution_score  : int    (0..100, strength of deserved positive resolution)
    value_band        : LOW | MEDIUM | HIGH
    reason            : str     (single-line human rationale)
    audit_trail       : list[str] (ordered decision provenance)

Integration (one-directional pipeline):
    reader   = read_message(message)                 # Module 1 (LSTM, NLU)
    verdict  = investigate(reader, profile)          # Module 2 (trust rules)
    decision = decide_resolution(reader, verdict, profile)   # THIS MODULE
    reply    = generate_reply(decision, message)     # Module 4 (FLAN-T5, NLG)
    email    = fire_email(profile, decision, reply)  # Module 4 (conditional)

No external dependencies. Standard library only. Defensive against missing fields.
Author: <team> | Timezone: Asia/Dubai (ISO 8601) | Currency: USD
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Mapping, MutableMapping

# ---------------------------------------------------------------------------
# Tunable policy constants (centralised — every threshold the team will debate)
# ---------------------------------------------------------------------------

HIGH_ORDER_VALUE: float = 5000.0      # "high stakes" money threshold (escalation)
SERIOUS_DELIVERY_VALUE: float = 2000.0  # delivery failure severe enough to refund
RESALE_PICKUP_FLOOR: float = 2000.0   # resale value at/above which pickup is worth it

WALLET_CAP_STANDARD: float = 500.0    # genuine goodwill cap
WALLET_RATE_STANDARD: float = 0.20
WALLET_CAP_SUSPICIOUS: float = 100.0  # suspicious-but-not-abusive cap
WALLET_RATE_SUSPICIOUS: float = 0.10

LOW_CONFIDENCE: float = 0.50          # below this the reading is "unsure"
VERY_HIGH_RISK: int = 80              # risk score that, with high stakes, escalates

CLV_HIGH: float = 40000.0
CLV_MEDIUM: float = 12000.0


# ---------------------------------------------------------------------------
# Enumerations (strict, string-valued so dict comparisons remain trivial)
# ---------------------------------------------------------------------------

class Action(str, Enum):
    ACKNOWLEDGE = "ACKNOWLEDGE"
    COUPON = "COUPON"
    WALLET_CREDIT = "WALLET_CREDIT"
    REFUND = "REFUND"
    ESCALATE = "ESCALATE"


class RefundType(str, Enum):
    PICKUP = "PICKUP"
    KEEP_ITEM = "KEEP_ITEM"
    NONE = "NONE"


class ValueBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Genuineness(str, Enum):
    GENUINE = "GENUINE"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_ABUSER = "LIKELY_ABUSER"


class ClaimStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class Frustration(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Email fires only for money-moving actions.
_EMAIL_ACTIONS = {Action.COUPON, Action.WALLET_CREDIT, Action.REFUND}

# Issue types treated as "clear product failure" -> default to refund when genuine.
_PRODUCT_FAILURE_ISSUES = {"Damaged_Defective", "Product_Quality"}
# Low-stakes informational issues -> acknowledge unless high value + high frustration.
_LOW_STAKES_ISSUES = {"App_Technical", "General_Query"}


# ---------------------------------------------------------------------------
# Defensive profile access — never KeyError on a partial profile.
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS: Mapping[str, Any] = {
    "customer_id": "UNKNOWN",
    "account_age_months": 0,
    "loyalty_tier": "Bronze",
    "lifetime_spend": 0.0,
    "total_orders": 0,
    "total_complaints": 0,
    "total_refunds_received": 0,
    "refund_to_order_ratio": 0.0,
    "items_kept_after_refund": 0,
    "complaints_last_30_days": 0,
    "is_first_purchase": False,
    "order_value": 0.0,
    "product_category": "Unknown",
    "is_perishable_or_hygiene": False,
    "resale_value": 0.0,
    "reverse_logistics_cost": 0.0,
    "prior_contacts_this_issue": 0,
    "prior_promise_logged": False,
    "customer_care_notes": "",
    "clv_estimate": 0.0,
}

_READER_DEFAULTS: Mapping[str, Any] = {
    "issue_type": "General_Query",
    "frustration": "Low",
    "confidence": 1.0,
}

_VERDICT_DEFAULTS: Mapping[str, Any] = {
    "genuineness": "GENUINE",
    "claim_status": "UNVERIFIED",
    "reason": "",
}


def _coalesce(src: Mapping[str, Any] | None, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `src` over `defaults`; tolerate None and missing keys. Pure, no mutation."""
    merged = dict(defaults)
    if src:
        for k, v in src.items():
            if v is not None:
                merged[k] = v
    return merged


def _as_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Decision object — typed, then serialised to the plain-dict contract.
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    action: Action = Action.ACKNOWLEDGE
    refund_type: RefundType = RefundType.NONE
    coupon_percent: int = 0
    wallet_credit: float = 0.0
    escalate: bool = False
    email_trigger: bool = False
    risk_score: int = 0
    resolution_score: int = 0
    value_band: ValueBand = ValueBand.LOW
    reason: str = ""
    audit_trail: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.audit_trail.append(msg)

    def to_contract(self) -> dict[str, Any]:
        """Serialise enums to their string values for downstream JSON/Module 4."""
        d = asdict(self)
        d["action"] = self.action.value
        d["refund_type"] = self.refund_type.value
        d["value_band"] = self.value_band.value
        # email_trigger is derived strictly from action — enforce invariant here.
        d["email_trigger"] = self.action in _EMAIL_ACTIONS
        d["escalate"] = self.action is Action.ESCALATE
        return d


# ===========================================================================
# Required functions (exact names / signatures the team will integrate against)
# ===========================================================================

def value_band(profile: Mapping[str, Any]) -> str:
    """
    Customer value band from loyalty_tier and clv_estimate.
      HIGH   if Platinum OR clv_estimate >= 40000
      MEDIUM if Gold/Silver OR clv_estimate >= 12000
      LOW    otherwise
    """
    p = _coalesce(profile, _PROFILE_DEFAULTS)
    tier = str(p["loyalty_tier"]).strip().title()
    clv = _as_float(p["clv_estimate"])

    if tier == "Platinum" or clv >= CLV_HIGH:
        return ValueBand.HIGH.value
    if tier in ("Gold", "Silver") or clv >= CLV_MEDIUM:
        return ValueBand.MEDIUM.value
    return ValueBand.LOW.value


def calculate_risk_score(profile: Mapping[str, Any],
                         investigator_verdict: Mapping[str, Any]) -> int:
    """
    Fraud / business-risk score in [0, 100]. Monotonic in every abuse signal.
    Drivers: refund_to_order_ratio (dominant), items_kept_after_refund,
    complaints_last_30_days, account age, first-purchase flag, and the
    Investigator's genuineness verdict.
    """
    p = _coalesce(profile, _PROFILE_DEFAULTS)
    v = _coalesce(investigator_verdict, _VERDICT_DEFAULTS)

    ratio = _as_float(p["refund_to_order_ratio"])
    kept = _as_int(p["items_kept_after_refund"])
    burst = _as_int(p["complaints_last_30_days"])
    age = _as_int(p["account_age_months"])
    first = bool(p["is_first_purchase"])

    score = 0.0
    score += _clamp(ratio * 60.0, 0, 40)     # ratio is THE signal: up to 40 pts
    score += _clamp(kept * 10.0, 0, 20)      # kept-after-refund: up to 20 pts
    score += _clamp(burst * 7.0, 0, 21)      # recent complaint burst: up to 21 pts
    if age <= 2:
        score += 10                          # new accounts carry higher risk
    if first:
        score += 8                           # first-purchase paradox

    genuineness = str(v["genuineness"]).upper()
    if genuineness == Genuineness.LIKELY_ABUSER.value:
        score += 25
    elif genuineness == Genuineness.SUSPICIOUS.value:
        score += 12

    return int(_clamp(score, 0, 100))


def calculate_resolution_score(reader_output: Mapping[str, Any],
                               investigator_verdict: Mapping[str, Any],
                               profile: Mapping[str, Any]) -> int:
    """
    Strength of deserved positive resolution in [0, 100].
    Frustration contributes only marginally (tone/urgency, NOT deservingness).
    Genuineness and a verified product failure dominate.
    """
    r = _coalesce(reader_output, _READER_DEFAULTS)
    v = _coalesce(investigator_verdict, _VERDICT_DEFAULTS)
    p = _coalesce(profile, _PROFILE_DEFAULTS)

    score = 0.0

    genuineness = str(v["genuineness"]).upper()
    if genuineness == Genuineness.GENUINE.value:
        score += 30
    elif genuineness == Genuineness.SUSPICIOUS.value:
        score += 10
    else:  # LIKELY_ABUSER
        score -= 50

    claim = str(v["claim_status"]).upper()
    if claim == ClaimStatus.CONFIRMED.value:
        score += 30
    elif claim == ClaimStatus.CONTRADICTED.value:
        score -= 20

    issue = str(r["issue_type"])
    if issue in _PRODUCT_FAILURE_ISSUES:
        score += 25
    elif issue == "Delivery":
        score += 15
    elif issue in ("Billing", "Refund_Return"):
        score += 10
    else:  # App_Technical / General_Query
        score += 5

    frustration = str(r["frustration"]).title()
    score += {"High": 10, "Medium": 5, "Low": 0}.get(frustration, 0)

    score += {"HIGH": 15, "MEDIUM": 8, "LOW": 0}[value_band(p)]

    if bool(p["is_perishable_or_hygiene"]) and issue in _PRODUCT_FAILURE_ISSUES:
        score += 5  # perishable defect: unambiguous, no recovery possible

    return int(_clamp(score, 0, 100))


def refund_logistics(profile: Mapping[str, Any]) -> str:
    """
    Decide PICKUP / KEEP_ITEM purely on item economics (called only on REFUND).
      1) perishable/hygiene -> KEEP_ITEM (cannot be resold; collection is waste)
      2) resale_value >= 2000 -> PICKUP (always recover costly resalable goods)
      3) reverse_logistics_cost > resale_value -> KEEP_ITEM (shipping costs more)
      4) otherwise -> PICKUP
    """
    p = _coalesce(profile, _PROFILE_DEFAULTS)
    if bool(p["is_perishable_or_hygiene"]):
        return RefundType.KEEP_ITEM.value
    resale = _as_float(p["resale_value"])
    reverse = _as_float(p["reverse_logistics_cost"])
    if resale >= RESALE_PICKUP_FLOOR:
        return RefundType.PICKUP.value
    if reverse > resale:
        return RefundType.KEEP_ITEM.value
    return RefundType.PICKUP.value


def should_escalate(reader_output: Mapping[str, Any],
                    investigator_verdict: Mapping[str, Any],
                    profile: Mapping[str, Any],
                    proposed_action: str,
                    risk_score: int) -> bool:
    """
    Escalation valve — fire ONLY when stakes AND uncertainty are both high.
      - SUSPICIOUS + proposed REFUND + order_value > 5000
      - reader confidence < 0.5 + HIGH-value customer
      - CONTRADICTED claim + high order value
      - very high risk score + high order value
    Over-escalation destroys the business case; keep this tight.
    """
    r = _coalesce(reader_output, _READER_DEFAULTS)
    v = _coalesce(investigator_verdict, _VERDICT_DEFAULTS)
    p = _coalesce(profile, _PROFILE_DEFAULTS)

    order_value = _as_float(p["order_value"])
    confidence = _as_float(r["confidence"], 1.0)
    band = value_band(p)
    genuineness = str(v["genuineness"]).upper()
    claim = str(v["claim_status"]).upper()
    proposed_refund = str(proposed_action).upper() == Action.REFUND.value

    if genuineness == Genuineness.SUSPICIOUS.value and proposed_refund and order_value > HIGH_ORDER_VALUE:
        return True
    if confidence < LOW_CONFIDENCE and band == ValueBand.HIGH.value:
        return True
    if claim == ClaimStatus.CONTRADICTED.value and order_value > HIGH_ORDER_VALUE:
        return True
    if risk_score >= VERY_HIGH_RISK and order_value > HIGH_ORDER_VALUE:
        return True
    return False


def decide_resolution(reader_output: Mapping[str, Any],
                      investigator_verdict: Mapping[str, Any],
                      profile: Mapping[str, Any]) -> dict[str, Any]:
    """
    Main entry point. Returns the Module 4 decision contract (plain dict).

    Pipeline: hard guardrails (priority-ordered) -> resolution tiers ->
    refund-logistics economics -> escalation valve -> email-trigger invariant
    -> audit trail. Every branch logs its rationale.
    """
    r = _coalesce(reader_output, _READER_DEFAULTS)
    v = _coalesce(investigator_verdict, _VERDICT_DEFAULTS)
    p = _coalesce(profile, _PROFILE_DEFAULTS)

    band = ValueBand(value_band(p))
    risk = calculate_risk_score(p, v)
    resolution = calculate_resolution_score(r, v, p)

    d = Decision(value_band=band, risk_score=risk, resolution_score=resolution)

    genuineness = str(v["genuineness"]).upper()
    claim = str(v["claim_status"]).upper()
    issue = str(r["issue_type"])
    frustration = str(r["frustration"]).title()
    confidence = _as_float(r["confidence"], 1.0)
    order_value = _as_float(p["order_value"])

    d.log(f"Customer classified as {genuineness}")
    d.log(f"Value band: {band.value}")
    d.log(f"Issue type: {issue}")
    d.log(f"Risk score: {risk} | Resolution score: {resolution}")

    # -- GUARDRAIL 1: never reward abuse. Short-circuits everything. ----------
    if genuineness == Genuineness.LIKELY_ABUSER.value:
        d.action = Action.ACKNOWLEDGE
        d.reason = "Likely abuser: acknowledge only, no compensation, no email."
        d.log("GUARDRAIL: LIKELY_ABUSER -> ACKNOWLEDGE, no payout")
        return _finalize(d)

    # -- GUARDRAIL 2: honour a confirmed company promise. ---------------------
    if claim == ClaimStatus.CONFIRMED.value:
        d.action = Action.REFUND
        d.refund_type = RefundType(refund_logistics(p))
        d.reason = (f"Confirmed prior promise honoured: refund with "
                    f"{d.refund_type.value} logistics.")
        d.log("GUARDRAIL: CONFIRMED promise -> honour -> REFUND")
        d.log(f"Refund logistics resolved to {d.refund_type.value}")
        return _maybe_escalate(d, r, v, p, risk)

    # -- GUARDRAIL 3: suspicious + high stakes -> escalate, no payout. --------
    if genuineness == Genuineness.SUSPICIOUS.value and order_value > HIGH_ORDER_VALUE:
        d.action = Action.ESCALATE
        d.reason = ("Suspicious customer with high order value: "
                    "escalate to human, no automatic payout.")
        d.log("GUARDRAIL: SUSPICIOUS + order_value>5000 -> ESCALATE")
        return _finalize(d)

    # -- GUARDRAIL 4: contradicted claim at high stakes. ----------------------
    if claim == ClaimStatus.CONTRADICTED.value and order_value > HIGH_ORDER_VALUE:
        if risk >= VERY_HIGH_RISK // 2:
            d.action = Action.ESCALATE
            d.reason = "Claim contradicted by records at high value with elevated risk: escalate."
            d.log("GUARDRAIL: CONTRADICTED + high value + risk -> ESCALATE")
        else:
            d.action = Action.ACKNOWLEDGE
            d.reason = "Claim contradicted by records: respond from the record, no payout."
            d.log("GUARDRAIL: CONTRADICTED + high value -> ACKNOWLEDGE")
        return _finalize(d)

    # -- SUSPICIOUS (not high stakes): capped wallet credit, verify. ----------
    if genuineness == Genuineness.SUSPICIOUS.value:
        d.action = Action.WALLET_CREDIT
        d.wallet_credit = round(min(order_value * WALLET_RATE_SUSPICIOUS, WALLET_CAP_SUSPICIOUS), 2)
        d.reason = (f"Suspicious but not abusive: capped goodwill of "
                    f"USD {d.wallet_credit} in wallet credit, pending verification.")
        d.log("TIER: SUSPICIOUS low-stakes -> WALLET_CREDIT (tight cap)")
        return _maybe_escalate(d, r, v, p, risk)

    # ========================  GENUINE customer  ============================

    # -- GUARDRAIL 5: clear product failure -> refund + logistics. ------------
    if issue in _PRODUCT_FAILURE_ISSUES:
        d.action = Action.REFUND
        d.refund_type = RefundType(refund_logistics(p))
        d.reason = (f"Genuine {issue.replace('_', '/').lower()} failure: full refund "
                    f"with {d.refund_type.value} logistics.")
        d.log("TIER: genuine product failure -> REFUND")
        d.log(f"Refund logistics resolved to {d.refund_type.value}")
        return _maybe_escalate(d, r, v, p, risk)

    # -- GUARDRAIL 9: low-stakes info issues. ---------------------------------
    if issue in _LOW_STAKES_ISSUES:
        if band == ValueBand.HIGH and frustration == Frustration.HIGH.value:
            d.action = Action.COUPON
            d.coupon_percent = 50
            d.reason = "High-value customer, high frustration on a service issue: 50% retention coupon."
            d.log("TIER: low-stakes + HIGH value + High frustration -> COUPON 50%")
        elif frustration == Frustration.HIGH.value:
            d.action = Action.COUPON
            d.coupon_percent = 20
            d.reason = "Service issue with high frustration: 20% goodwill coupon."
            d.log("TIER: low-stakes + High frustration -> COUPON 20%")
        else:
            d.action = Action.ACKNOWLEDGE
            d.reason = "Low-stakes service issue, low/medium frustration: acknowledge, no payout."
            d.log("TIER: low-stakes routine -> ACKNOWLEDGE")
        return _maybe_escalate(d, r, v, p, risk)

    # -- GUARDRAILS 6-8: Delivery / Billing / Refund_Return by frustration. ---
    if frustration == Frustration.HIGH.value:
        if band == ValueBand.HIGH:
            # GUARDRAIL 8: high-value + high frustration -> 50% coupon or refund by severity.
            if issue == "Delivery" and order_value >= SERIOUS_DELIVERY_VALUE:
                d.action = Action.REFUND
                d.refund_type = RefundType(refund_logistics(p))
                d.reason = "High-value customer, serious delivery failure: full refund."
                d.log("TIER: HIGH value + serious Delivery -> REFUND")
                d.log(f"Refund logistics resolved to {d.refund_type.value}")
            else:
                d.action = Action.COUPON
                d.coupon_percent = 50
                d.reason = "High-value customer, high frustration: 50% retention coupon."
                d.log("TIER: HIGH value + High frustration -> COUPON 50%")
        else:
            # GUARDRAIL 6: genuine delivery/high frustration, normal value.
            if issue == "Delivery" and order_value >= SERIOUS_DELIVERY_VALUE:
                d.action = Action.REFUND
                d.refund_type = RefundType(refund_logistics(p))
                d.reason = "Genuine serious delivery failure: full refund."
                d.log("TIER: serious Delivery, normal value -> REFUND")
                d.log(f"Refund logistics resolved to {d.refund_type.value}")
            else:
                d.action = Action.WALLET_CREDIT
                d.wallet_credit = round(min(order_value * WALLET_RATE_STANDARD, WALLET_CAP_STANDARD), 2)
                d.reason = (f"Genuine high-frustration issue: USD {d.wallet_credit} "
                            f"wallet credit (capped goodwill).")
                d.log("TIER: High frustration, normal value -> WALLET_CREDIT")

    elif frustration == Frustration.MEDIUM.value:
        # GUARDRAIL 7: medium frustration -> coupon 20 or wallet credit by value.
        if band in (ValueBand.HIGH, ValueBand.MEDIUM):
            d.action = Action.WALLET_CREDIT
            d.wallet_credit = round(min(order_value * WALLET_RATE_STANDARD, WALLET_CAP_STANDARD), 2)
            d.reason = f"Genuine medium-frustration issue: USD {d.wallet_credit} wallet credit."
            d.log("TIER: Medium frustration + MEDIUM/HIGH value -> WALLET_CREDIT")
        else:
            d.action = Action.COUPON
            d.coupon_percent = 20
            d.reason = "Genuine medium-frustration issue, lower value: 20% goodwill coupon."
            d.log("TIER: Medium frustration + LOW value -> COUPON 20%")

    else:  # Low frustration, genuine, non-product-failure, non-low-stakes.
        d.action = Action.ACKNOWLEDGE
        d.reason = "Genuine routine issue, low frustration: acknowledge with no payout."
        d.log("TIER: Low frustration routine -> ACKNOWLEDGE")

    return _maybe_escalate(d, r, v, p, risk)


# Handbook-compatibility alias: the handbook (and Module 4's starter code) call
# `decide(verdict, reader, profile)`. This thin wrapper maps that signature onto
# decide_resolution(reader, verdict, profile) so the M3->M4 stitch is painless.
def decide(investigator_verdict: Mapping[str, Any],
           reader_output: Mapping[str, Any],
           profile: Mapping[str, Any]) -> dict[str, Any]:
    return decide_resolution(reader_output, investigator_verdict, profile)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_escalate(d: Decision,
                    reader: Mapping[str, Any],
                    verdict: Mapping[str, Any],
                    profile: Mapping[str, Any],
                    risk: int) -> dict[str, Any]:
    """Apply the escalation valve over a proposed action, then finalize."""
    if should_escalate(reader, verdict, profile, d.action.value, risk):
        # GUARDRAIL 10 et al.: low-confidence high-value, or high-stakes uncertainty.
        prior = d.action.value
        d.action = Action.ESCALATE
        d.refund_type = RefundType.NONE
        d.coupon_percent = 0
        d.wallet_credit = 0.0
        d.reason = f"Escalation valve fired (was {prior}): high stakes with low certainty/trust."
        d.log(f"ESCALATION VALVE: overrode {prior} -> ESCALATE")
    return _finalize(d)


def _finalize(d: Decision) -> dict[str, Any]:
    """Enforce contract invariants and serialise. Single source of truth for output."""
    if d.action is not Action.REFUND:
        d.refund_type = RefundType.NONE
    if d.action is not Action.COUPON:
        d.coupon_percent = 0
    if d.action is not Action.WALLET_CREDIT:
        d.wallet_credit = 0.0
    # email_trigger & escalate are derived in to_contract() — never hand-set.
    d.log(f"Email trigger: {d.action in _EMAIL_ACTIONS}")
    return d.to_contract()


# ===========================================================================
# Test runner — mock profiles for the 10 mandated scenarios + sample outputs.
# ===========================================================================

def _mock(**overrides: Any) -> dict[str, Any]:
    base = dict(_PROFILE_DEFAULTS)
    base.update(overrides)
    return base


def _run_test_suite() -> None:
    import json

    cases: list[tuple[str, dict, dict, dict, str]] = [
        (
            "1. Gold, spoiled milk, low refund ratio -> REFUND + KEEP_ITEM + email",
            {"issue_type": "Damaged_Defective", "frustration": "Medium", "confidence": 0.9},
            {"genuineness": "GENUINE", "claim_status": "UNVERIFIED"},
            _mock(customer_id="C0042", loyalty_tier="Gold", clv_estimate=18000,
                  refund_to_order_ratio=0.04, order_value=45,
                  product_category="Fresh Grocery", is_perishable_or_hygiene=True,
                  resale_value=0, reverse_logistics_cost=100, account_age_months=36),
            "REFUND/KEEP_ITEM, email True",
        ),
        (
            "2. Silver, cracked TV, low refund ratio -> REFUND + PICKUP + email",
            {"issue_type": "Damaged_Defective", "frustration": "High", "confidence": 0.88},
            {"genuineness": "GENUINE", "claim_status": "UNVERIFIED"},
            _mock(customer_id="C0101", loyalty_tier="Silver", clv_estimate=15000,
                  refund_to_order_ratio=0.06, order_value=3500,
                  product_category="Electronics", is_perishable_or_hygiene=False,
                  resale_value=12000, reverse_logistics_cost=400, account_age_months=20),
            "REFUND/PICKUP, email True",
        ),
        (
            "3. Serial refunder, high frustration -> ACKNOWLEDGE + no email",
            {"issue_type": "Delivery", "frustration": "High", "confidence": 0.92},
            {"genuineness": "LIKELY_ABUSER", "claim_status": "UNVERIFIED",
             "reason": "High refund ratio and kept items"},
            _mock(customer_id="C0666", loyalty_tier="Bronze", clv_estimate=3000,
                  refund_to_order_ratio=0.83, items_kept_after_refund=3,
                  complaints_last_30_days=4, total_orders=6, order_value=300),
            "ACKNOWLEDGE, email False",
        ),
        (
            "4. Claims agent promised refund, NO records -> no auto-refund",
            {"issue_type": "Refund_Return", "frustration": "Medium", "confidence": 0.8},
            {"genuineness": "GENUINE", "claim_status": "UNVERIFIED",
             "reason": "No prior contact or promise on record"},
            _mock(customer_id="C0204", loyalty_tier="Silver", clv_estimate=13000,
                  refund_to_order_ratio=0.1, order_value=600,
                  prior_promise_logged=False, customer_care_notes=""),
            "NOT REFUND (COUPON/WALLET_CREDIT)",
        ),
        (
            "5. Records confirm prior promise -> REFUND + email",
            {"issue_type": "Delivery", "frustration": "High", "confidence": 0.85},
            {"genuineness": "GENUINE", "claim_status": "CONFIRMED",
             "reason": "Agent note: promised refund 12 June, not yet issued"},
            _mock(customer_id="C0042", loyalty_tier="Gold", clv_estimate=18000,
                  refund_to_order_ratio=0.05, order_value=900,
                  prior_promise_logged=True, resale_value=0,
                  reverse_logistics_cost=80, is_perishable_or_hygiene=False),
            "REFUND, email True",
        ),
        (
            "6. Cheap phone case, resale < pickup cost -> REFUND + KEEP_ITEM",
            {"issue_type": "Damaged_Defective", "frustration": "Medium", "confidence": 0.87},
            {"genuineness": "GENUINE", "claim_status": "UNVERIFIED"},
            _mock(customer_id="C0310", loyalty_tier="Silver", clv_estimate=12500,
                  refund_to_order_ratio=0.08, order_value=80,
                  product_category="Accessories", is_perishable_or_hygiene=False,
                  resale_value=80, reverse_logistics_cost=250),
            "REFUND/KEEP_ITEM, email True",
        ),
        (
            "7. First purchase didn't arrive -> capped generosity (escalate if high value)",
            {"issue_type": "Delivery", "frustration": "High", "confidence": 0.82},
            {"genuineness": "SUSPICIOUS", "claim_status": "UNVERIFIED",
             "reason": "Account 0 months, first purchase"},
            _mock(customer_id="C0999", loyalty_tier="Bronze", clv_estimate=2000,
                  account_age_months=0, is_first_purchase=True,
                  refund_to_order_ratio=0.0, order_value=400),
            "WALLET_CREDIT capped, email True",
        ),
        (
            "8. Bronze, app crashing, low frustration -> ACKNOWLEDGE + no email",
            {"issue_type": "App_Technical", "frustration": "Low", "confidence": 0.9},
            {"genuineness": "GENUINE", "claim_status": "UNVERIFIED"},
            _mock(customer_id="C0500", loyalty_tier="Bronze", clv_estimate=3000,
                  refund_to_order_ratio=0.02, order_value=0),
            "ACKNOWLEDGE, email False",
        ),
        (
            "9. Suspicious account, high order value -> ESCALATE + no email",
            {"issue_type": "Refund_Return", "frustration": "High", "confidence": 0.8},
            {"genuineness": "SUSPICIOUS", "claim_status": "UNVERIFIED",
             "reason": "Elevated ratio, shaky history"},
            _mock(customer_id="C0777", loyalty_tier="Silver", clv_estimate=14000,
                  refund_to_order_ratio=0.3, order_value=8000,
                  account_age_months=8),
            "ESCALATE, email False",
        ),
        (
            "10. Platinum, genuine defect, calm complaint -> REFUND, protect relationship",
            {"issue_type": "Damaged_Defective", "frustration": "Low", "confidence": 0.94},
            {"genuineness": "GENUINE", "claim_status": "UNVERIFIED"},
            _mock(customer_id="C0001", loyalty_tier="Platinum", clv_estimate=60000,
                  refund_to_order_ratio=0.02, order_value=4500,
                  product_category="Appliances", is_perishable_or_hygiene=False,
                  resale_value=9000, reverse_logistics_cost=500, account_age_months=48),
            "REFUND/PICKUP, email True",
        ),
    ]

    passed = 0
    escalations = 0
    for title, reader, verdict, profile, expected in cases:
        decision = decide_resolution(reader, verdict, profile)
        if decision["escalate"]:
            escalations += 1
        ok = _assert_case(title, decision)
        passed += int(ok)
        print("=" * 78)
        print(title)
        print(f"   expected: {expected}")
        print(f"   verdict : action={decision['action']}, refund_type={decision['refund_type']}, "
              f"coupon={decision['coupon_percent']}, wallet=USD {decision['wallet_credit']}, "
              f"email={decision['email_trigger']}  [{'PASS' if ok else 'FAIL'}]")
        print("   contract:")
        print(json.dumps(decision, indent=4))

    n = len(cases)
    automation_rate = (n - escalations) / n * 100
    print("=" * 78)
    print(f"RESULTS: {passed}/{n} assertions passed | "
          f"automation rate = {automation_rate:.0f}% ({escalations} escalations)")


def _assert_case(title: str, d: dict[str, Any]) -> bool:
    """Lightweight expectations keyed off the scenario number — integration-level checks."""
    n = title.split(".", 1)[0]
    a, rt, email = d["action"], d["refund_type"], d["email_trigger"]
    checks = {
        "1": a == "REFUND" and rt == "KEEP_ITEM" and email,
        "2": a == "REFUND" and rt == "PICKUP" and email,
        "3": a == "ACKNOWLEDGE" and not email,
        "4": a != "REFUND",
        "5": a == "REFUND" and email,
        "6": a == "REFUND" and rt == "KEEP_ITEM" and email,
        "7": a in ("WALLET_CREDIT", "COUPON", "ESCALATE"),
        "8": a == "ACKNOWLEDGE" and not email,
        "9": a == "ESCALATE" and not email,
        "10": a == "REFUND" and email,
    }
    return checks.get(n, True)


if __name__ == "__main__":
    _run_test_suite()
