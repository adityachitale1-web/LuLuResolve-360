"""
Integration test for the full stitch: Module 2 -> Module 3 over the real
customers.csv. Checks the contract invariants Module 4 depends on and reports
the automation rate. Run:  python3 test_pipeline.py
"""
import csv
from investigator import investigate
from module3_resolution_engine import decide_resolution, decide

REQUIRED = {"action", "refund_type", "coupon_percent", "wallet_credit",
            "escalate", "email_trigger", "reason"}
EMAIL_ACTIONS = {"COUPON", "WALLET_CREDIT", "REFUND"}


def _profile(r):
    d = dict(r)
    for k in ("account_age_months", "total_orders", "total_complaints",
              "total_refunds_received", "items_kept_after_refund",
              "complaints_last_30_days", "order_value", "resale_value",
              "reverse_logistics_cost", "prior_contacts_this_issue",
              "clv_estimate", "lifetime_spend"):
        d[k] = float(r[k]) if r.get(k) not in (None, "") else 0
    d["refund_to_order_ratio"] = float(r["refund_to_order_ratio"])
    for k in ("is_first_purchase", "is_perishable_or_hygiene", "prior_promise_logged"):
        d[k] = str(r[k]).strip().lower() == "true"
    return d


def run():
    rows = list(csv.DictReader(open("customers.csv")))
    reader = {"issue_type": "Refund_Return", "frustration": "High", "confidence": 0.9}

    key_violations = email_violations = esc_violations = 0
    escalations = 0
    from collections import Counter
    actions = Counter()

    for r in rows:
        p = _profile(r)
        verdict = investigate(reader, p)
        d = decide_resolution(reader, verdict, p)
        actions[d["action"]] += 1
        if not REQUIRED.issubset(d):
            key_violations += 1
        if d["email_trigger"] != (d["action"] in EMAIL_ACTIONS):
            email_violations += 1
        if d["escalate"] != (d["action"] == "ESCALATE"):
            esc_violations += 1
        if d["action"] == "ESCALATE":
            escalations += 1
        # decide() alias must equal decide_resolution()
        assert decide(verdict, reader, p) == d

    n = len(rows)
    rate = (n - escalations) / n * 100
    print("Full pipeline (M1 mock -> M2 -> M3) over %d real customers" % n)
    print("  contract-key violations  :", key_violations)
    print("  email-invariant breaches :", email_violations)
    print("  escalate-invariant breaks:", esc_violations)
    print("  decide() alias           : matches decide_resolution()")
    print("  action distribution      :", dict(actions))
    print("  automation rate          : %.1f%% (%d escalations)" % (rate, escalations))

    ok = key_violations == email_violations == esc_violations == 0 and rate >= 60
    print("\n%s" % ("PASS — contract clean, automation healthy"
                    if ok else "FAIL — see violations above"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
