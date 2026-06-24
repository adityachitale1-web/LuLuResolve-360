"""
LuluCare 360 — Module 1 + Module 2 STITCH
=========================================
Wires the Reader (Module 1) into the Investigator (Module 2):

    message + customer_id
        |
        v
    read_message(text)        # Module 1  -> {issue_type, frustration, confidence}
        |                     (the Reader; trained in Module1_The_Reader.ipynb)
        v
    investigate(reader, profile)   # Module 2 -> {genuineness, claim_status, reason, ...}

The Reader is a TensorFlow model that only exists after the Module 1 notebook is
run, so this file is written to be DEPENDENCY-INJECTED: you pass in `read_message`
and `lookup_profile`. That keeps the stitch testable WITHOUT TensorFlow — the demo
at the bottom uses a tiny keyword-based mock Reader so the wiring can be verified
on any machine. In Colab, pass the REAL `read_message` from the notebook instead.

    from pipeline import run_pipeline
    run_pipeline("My delivery is late!", "C0002", read_message, lookup_profile)
"""

import csv
from investigator import investigate


def run_pipeline(message, customer_id, read_message, lookup_profile):
    """
    message       : raw customer complaint (str)
    customer_id   : id to look up history (str)
    read_message  : Module 1 contract fn  text -> {issue_type,frustration,confidence}
    lookup_profile: fn  customer_id -> profile dict (or None)
    -> the combined Module1+2 result (Module 3 will consume `verdict`)
    """
    reader = read_message(message)                 # Module 1: understand the text
    profile = lookup_profile(customer_id)          # fetch history
    verdict = investigate(reader, profile)         # Module 2: judge trust
    return {
        "message": message,
        "customer_id": customer_id,
        "reader": reader,        # Module 1 output
        "verdict": verdict,      # Module 2 output -> hand to Module 3 (Economist)
    }


# ----------------------------------------------------------------------
# Standalone demo helpers (no TensorFlow needed)
# ----------------------------------------------------------------------
def make_csv_lookup(path="customers.csv"):
    """Build a lookup_profile() backed by a CSV (mirrors the notebook helper)."""
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["customer_id"]] = r
    return lambda cid: rows.get(cid)


def mock_read_message(text):
    """
    A tiny keyword-based stand-in for the real LSTM Reader, ONLY for verifying the
    stitch offline. The real Module 1 read_message replaces this in Colab.
    """
    t = text.lower()
    if any(w in t for w in ("deliver", "courier", "arrive", "package", "shipment")):
        issue = "Delivery"
    elif any(w in t for w in ("refund", "return", "money back")):
        issue = "Refund_Return"
    elif any(w in t for w in ("charged", "bill", "invoice", "payment")):
        issue = "Billing"
    elif any(w in t for w in ("crack", "broken", "damaged", "defective", "faulty")):
        issue = "Damaged_Defective"
    elif any(w in t for w in ("app", "website", "crash", "login")):
        issue = "App_Technical"
    elif any(w in t for w in ("quality", "cheap", "material", "wore out")):
        issue = "Product_Quality"
    else:
        issue = "General_Query"
    if any(w in t for w in ("furious", "outrageous", "unacceptable", "done with", "!!")):
        frus = "High"
    elif any(w in t for w in ("annoyed", "disappointed", "frustrating", "not happy")):
        frus = "Medium"
    else:
        frus = "Low"
    return {"issue_type": issue, "frustration": frus, "confidence": 0.80}


if __name__ == "__main__":
    lookup = make_csv_lookup("customers.csv")

    # C0001 = LIKELY_ABUSER, C0002 = GENUINE in the shipped dataset.
    demos = [
        ("My refund still has not arrived and I am FURIOUS!! This is unacceptable!!", "C0001"),
        ("Hi, my delivery is a bit late, could you check on it?",                     "C0002"),
        ("Your rep promised me a refund last week and nothing happened.",             "C0002"),
    ]
    for msg, cid in demos:
        out = run_pipeline(msg, cid, mock_read_message, lookup)
        print("=" * 78)
        print("MESSAGE   :", msg)
        print("CUSTOMER  :", cid)
        print("READER    :", out["reader"])
        print("VERDICT   : genuineness=%s  claim_status=%s  risk_score=%s"
              % (out["verdict"]["genuineness"], out["verdict"]["claim_status"],
                 out["verdict"]["risk_score"]))
        print("REASON    :", out["verdict"]["reason"])
