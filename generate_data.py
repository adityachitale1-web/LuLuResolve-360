"""
LuluCare 360 — Synthetic Dataset Generator
============================================
Produces two CSVs that join on customer_id:

  1. messages.csv   — one row per complaint message  (text -> trains the LSTM)
  2. customers.csv  — one row per customer            (history -> feeds the policy)

Design goals:
  * BALANCED: roughly equal counts across the 7 issue types and 3 frustration levels
  * UNBIASED: frustration is NOT trivially predictable from issue type;
              genuineness is NOT correlated with frustration (an abuser can be
              calm or furious; a genuine customer can be calm or furious)
  * REALISTIC: message text uses varied templates + fillers so the LSTM learns
              language patterns, not a single giveaway keyword
"""

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------
ISSUE_TYPES = ["Delivery", "Damaged_Defective", "Refund_Return",
               "Billing", "Product_Quality", "App_Technical", "General_Query"]

FRUSTRATION = ["Low", "Medium", "High"]

# Target: balanced messages across issue types and frustration.
N_PER_ISSUE_FRUSTRATION = 30          # 7 issues x 3 frustration x 30 = 630 messages
N_CUSTOMERS = 220                     # enough distinct customers

PRODUCT_CATEGORIES = {
    # category: (is_perishable_or_hygiene, typical_price_range, resale_fraction)
    "Fresh_Produce":  (True,  (20, 400),    0.0),
    "Dairy":          (True,  (30, 300),    0.0),
    "Meat_Seafood":   (True,  (100, 900),   0.0),
    "Frozen_Food":    (True,  (80, 600),    0.0),
    "Cosmetics":      (True,  (150, 3000),  0.0),   # opened = hygiene, no resale
    "Electronics":    (False, (3000, 90000), 0.65),
    "Home_Appliance": (False, (2000, 60000), 0.6),
    "Fashion":        (False, (300, 6000),  0.4),
    "Home_Goods":     (False, (150, 5000),  0.45),
    "Toys_Books":     (False, (100, 3000),  0.4),
}

LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]

# ----------------------------------------------------------------------
# 2. MESSAGE TEMPLATES  (varied, so no single keyword gives away the label)
# ----------------------------------------------------------------------
# Each issue type has multiple phrasings. Frustration is injected SEPARATELY
# via openers/closers, so frustration is independent of issue type.

ISSUE_TEMPLATES = {
    "Delivery": [
        "my order still has not arrived",
        "the delivery is late again",
        "I have been waiting for my package for days",
        "the courier never showed up",
        "my shipment is stuck and not moving",
        "the delivery date keeps getting pushed",
        "nobody delivered my order today as promised",
    ],
    "Damaged_Defective": [
        "the item arrived broken",
        "the product was damaged in the box",
        "my {cat} stopped working on day one",
        "the screen was cracked when I opened it",
        "the package was crushed and the item dented",
        "the {cat} is defective and will not switch on",
        "there were parts missing and the unit is faulty",
    ],
    "Refund_Return": [
        "I want to return this item and get my money back",
        "my refund has not been processed yet",
        "I returned the product but no refund came",
        "how do I send this back for a refund",
        "the return pickup was never scheduled",
        "I am still waiting for the refund you promised",
        "please refund me for the order I sent back",
    ],
    "Billing": [
        "I was charged twice for the same order",
        "there is an extra charge on my bill I do not understand",
        "the amount deducted is more than the order total",
        "my coupon was not applied at checkout",
        "I see a payment I never authorized",
        "the invoice amount is wrong",
        "you billed me after I cancelled",
    ],
    "Product_Quality": [
        "the quality of this product is poor",
        "the {cat} feels cheap and not worth the price",
        "the material is low quality and wore out fast",
        "this is not the quality I expected from Lulu",
        "the product looks nothing like the description",
        "the {cat} quality has gone down compared to before",
        "the item works but the build quality is bad",
    ],
    "App_Technical": [
        "the app keeps crashing when I try to checkout",
        "I cannot log into my account on the app",
        "the website freezes when I add items to cart",
        "the payment page is not loading",
        "the app shows an error every time I open it",
        "I cannot track my order in the app",
        "the search on the app is not working",
    ],
    "General_Query": [
        "I just wanted to ask about your return policy",
        "can you tell me the store timings",
        "how do I add an address to my account",
        "is this product available in the Dubai store",
        "I have a question about my loyalty points",
        "do you deliver to my area",
        "how can I change my registered phone number",
    ],
}

# Frustration markers injected independently of issue type.
FRUSTRATION_OPENERS = {
    "Low": ["Hi,", "Hello,", "Hi team,", "Good morning,", "Hello Lulu,", "Hi there,"],
    "Medium": ["I'm a bit annoyed that", "This is frustrating —", "I'm disappointed that",
               "Not happy that", "It's bothering me that", "I'm unhappy because"],
    "High": ["I am ABSOLUTELY furious!!", "This is the LAST straw.", "I am done with Lulu!",
             "Completely unacceptable!!", "I have HAD enough —", "This is outrageous!!"],
}

FRUSTRATION_CLOSERS = {
    "Low": ["Thanks for your help.", "Appreciate it.", "Please let me know.",
            "Thank you.", "Kind regards.", "Looking forward to your reply."],
    "Medium": ["Please sort this out soon.", "I'd like this fixed.",
               "Hoping for a quick resolution.", "Please look into it.",
               "This needs attention.", "Kindly resolve."],
    "High": ["Fix this NOW or I'm leaving for good!", "I want this resolved immediately!!",
             "I will never shop here again!", "Sort it out or refund me!!",
             "This is the third time — enough!!", "I'm escalating this everywhere!"],
}

def make_message(issue, frustration):
    """Compose a message that signals issue via body, frustration via opener/closer."""
    body = random.choice(ISSUE_TEMPLATES[issue])
    if "{cat}" in body:
        cat_word = random.choice(["TV", "phone", "blender", "headphones",
                                  "laptop", "kettle", "speaker", "watch"])
        body = body.replace("{cat}", cat_word)
    opener = random.choice(FRUSTRATION_OPENERS[frustration])
    closer = random.choice(FRUSTRATION_CLOSERS[frustration])
    # Sometimes drop opener or closer to add natural variety (but keep frustration signal)
    parts = []
    if frustration == "High" or random.random() > 0.25:
        parts.append(opener)
    parts.append(body + ".")
    if frustration == "High" or random.random() > 0.25:
        parts.append(closer)
    text = " ".join(parts)
    # Capitalize first letter
    return text[0].upper() + text[1:]

# ----------------------------------------------------------------------
# 3. GENERATE CUSTOMERS  (history independent of message frustration)
# ----------------------------------------------------------------------
# Genuineness archetypes — assigned so that genuineness is INDEPENDENT of
# frustration (set later). Balanced mix.
#   GENUINE      : low refund ratio, established or new-but-clean
#   SUSPICIOUS   : moderate ratio / new account / mild burst
#   LIKELY_ABUSER: high refund ratio, kept items, complaint bursts

GENUINENESS_MIX = (["GENUINE"] * 120 + ["SUSPICIOUS"] * 55 + ["LIKELY_ABUSER"] * 45)
random.shuffle(GENUINENESS_MIX)

CARE_NOTE_BANK_CONFIRM = [
    "Agent promised 20% coupon on previous call, not yet issued.",
    "Prior agent committed to a refund, pending processing.",
    "Customer was assured replacement on last contact.",
]
CARE_NOTE_BANK_CONTRADICT = [
    "Explained item is non-returnable (perishable); customer acknowledged.",
    "Informed customer no refund applicable per policy; customer agreed.",
    "Clarified no prior promise was made; resolved on call.",
]

def make_customer(cid, archetype):
    tier = random.choices(LOYALTY_TIERS, weights=[40, 30, 20, 10])[0]

    if archetype == "GENUINE":
        account_age = random.randint(6, 72)
        total_orders = random.randint(15, 120)
        ratio = round(random.uniform(0.0, 0.10), 3)        # low
        kept = random.choice([0, 0, 0, 1])
        comp_30 = random.choice([0, 0, 1, 1, 2])
        first = False
    elif archetype == "SUSPICIOUS":
        # Either a brand-new account OR a moderate-ratio account
        if random.random() < 0.5:
            account_age = random.randint(0, 2)             # very new
            total_orders = random.randint(1, 4)
            ratio = round(random.uniform(0.0, 0.4), 3)
            first = (total_orders == 1)
        else:
            account_age = random.randint(3, 30)
            total_orders = random.randint(6, 30)
            ratio = round(random.uniform(0.25, 0.45), 3)
            first = False
        kept = random.choice([0, 1, 1])
        comp_30 = random.choice([1, 2, 2, 3])
    else:  # LIKELY_ABUSER
        account_age = random.randint(1, 24)
        total_orders = random.randint(4, 20)
        ratio = round(random.uniform(0.5, 0.9), 3)         # high
        kept = random.choice([2, 3, 4, 5])
        comp_30 = random.choice([3, 4, 5, 6])
        first = False

    total_refunds = int(round(ratio * total_orders))
    total_complaints = max(total_refunds, comp_30, random.randint(total_refunds, total_orders))
    # lifetime spend scales with orders + tier
    tier_mult = {"Bronze": 1.0, "Silver": 1.6, "Gold": 2.5, "Platinum": 4.0}[tier]
    lifetime_spend = int(total_orders * random.uniform(300, 1500) * tier_mult)

    # This-order product + economics
    cat = random.choice(list(PRODUCT_CATEGORIES.keys()))
    perish, (lo, hi), resale_frac = PRODUCT_CATEGORIES[cat]
    order_value = int(random.uniform(lo, hi))
    resale_value = int(order_value * resale_frac)
    # reverse logistics cost: heavier/cheaper items vary; independent-ish noise
    reverse_cost = int(random.uniform(40, 300))

    # Claim verification fields — independent of archetype, assigned realistically
    r = random.random()
    if r < 0.15:
        prior_contacts = random.randint(1, 3)
        prior_promise = True
        note = random.choice(CARE_NOTE_BANK_CONFIRM)
    elif r < 0.35:
        prior_contacts = random.randint(1, 2)
        prior_promise = False
        note = random.choice(CARE_NOTE_BANK_CONTRADICT)
    else:
        prior_contacts = 0
        prior_promise = False
        note = ""

    clv = int(lifetime_spend / max(account_age, 1) * 12)   # simple annualized CLV

    return {
        "customer_id": cid,
        "account_age_months": account_age,
        "loyalty_tier": tier,
        "lifetime_spend": lifetime_spend,
        "total_orders": total_orders,
        "total_complaints": total_complaints,
        "total_refunds_received": total_refunds,
        "refund_to_order_ratio": round(total_refunds / max(total_orders, 1), 3),
        "items_kept_after_refund": kept,
        "complaints_last_30_days": comp_30,
        "is_first_purchase": first,
        "order_value": order_value,
        "product_category": cat,
        "is_perishable_or_hygiene": perish,
        "resale_value": resale_value,
        "reverse_logistics_cost": reverse_cost,
        "prior_contacts_this_issue": prior_contacts,
        "prior_promise_logged": prior_promise,
        "customer_care_notes": note,
        "_archetype": archetype,         # ground-truth label (drop before shipping if desired)
        "clv_estimate": clv,
    }

customers = []
for i in range(N_CUSTOMERS):
    cid = f"C{i+1:04d}"
    customers.append(make_customer(cid, GENUINENESS_MIX[i]))

customers_df = pd.DataFrame(customers)

# ----------------------------------------------------------------------
# 4. GENERATE MESSAGES  (balanced; frustration independent of issue)
# ----------------------------------------------------------------------
rows = []
mid = 1
customer_ids = customers_df["customer_id"].tolist()

for issue in ISSUE_TYPES:
    for frust in FRUSTRATION:
        for _ in range(N_PER_ISSUE_FRUSTRATION):
            text = make_message(issue, frust)
            cust = random.choice(customer_ids)          # random customer (no leakage)
            rows.append({
                "message_id": f"M{mid:04d}",
                "customer_id": cust,
                "text": text,
                "issue_type": issue,
                "frustration": frust,
            })
            mid += 1

messages_df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)

# ----------------------------------------------------------------------
# 5. SAVE
# ----------------------------------------------------------------------
messages_df.to_csv("messages.csv", index=False)
customers_df.to_csv("customers.csv", index=False)

print("messages.csv :", messages_df.shape)
print("customers.csv:", customers_df.shape)
print()
print("=== BALANCE CHECKS ===")
print("\nIssue type counts:\n", messages_df.issue_type.value_counts().sort_index())
print("\nFrustration counts:\n", messages_df.frustration.value_counts())
print("\nIssue x Frustration crosstab (should be ~equal = 30 each):")
print(pd.crosstab(messages_df.issue_type, messages_df.frustration))
print("\nGenuineness archetype counts:\n", customers_df._archetype.value_counts())
print("\nLoyalty tier counts:\n", customers_df.loyalty_tier.value_counts())
print("\nClaim-status field distribution:")
print("  prior_promise_logged True:", int(customers_df.prior_promise_logged.sum()))
print("  has care notes         :", int((customers_df.customer_care_notes != '').sum()))
