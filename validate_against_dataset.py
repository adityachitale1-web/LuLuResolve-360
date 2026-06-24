"""
Validate the genuineness rules against the dataset's ground-truth _archetype.
This is the honest truth-teller: it proves the rules separate the archetypes the
data generator actually planted. (claim_status is generated INDEPENDENTLY of the
archetype, so only genuineness is validatable here.)

Run:  python3 validate_against_dataset.py
"""
import csv
from collections import Counter, defaultdict
from investigator import assess_genuineness

LABELS = ["GENUINE", "SUSPICIOUS", "LIKELY_ABUSER"]

rows = []
with open("customers.csv", newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

correct = 0
conf = defaultdict(Counter)          # truth -> predicted counts
misses = []
for r in rows:
    truth = r.get("_archetype", "")
    pred, flags = assess_genuineness(r)
    conf[truth][pred] += 1
    if pred == truth:
        correct += 1
    else:
        misses.append((r["customer_id"], truth, pred,
                       r["refund_to_order_ratio"], r["items_kept_after_refund"],
                       r["complaints_last_30_days"], r["account_age_months"],
                       r["total_orders"]))

n = len(rows)
print("Validated %d customers against ground-truth _archetype" % n)
print("Genuineness accuracy: %d/%d = %.1f%%" % (correct, n, 100.0 * correct / n))
print()
print("Confusion matrix (rows = truth, cols = predicted):")
print("%-15s %s" % ("", "  ".join("%-13s" % l for l in LABELS)))
for truth in LABELS:
    print("%-15s %s" % (truth, "  ".join("%-13d" % conf[truth][p] for p in LABELS)))

print()
if misses:
    print("Misclassifications (%d):" % len(misses))
    print("%-7s %-14s %-14s %-7s %-5s %-5s %-5s %-6s"
          % ("id", "truth", "pred", "ratio", "kept", "c30", "age", "orders"))
    for m in misses:
        print("%-7s %-14s %-14s %-7s %-5s %-5s %-5s %-6s" % m)
else:
    print("Zero misclassifications — rules perfectly recover the planted archetypes.")
