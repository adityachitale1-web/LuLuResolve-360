# Module 2 — The Investigator (LuluCare 360)

**Owner:** Atharva · **Role:** the trust engine · **Concept:** dialogue policy (trust)

The Investigator answers two questions about every complaint, using **transparent
rules, not a model** — so every decision can be read, defended, and debated:

1. **Can we trust this customer?** → `genuineness` ∈ {GENUINE, SUSPICIOUS, LIKELY_ABUSER}
2. **Do our own records back their claim?** → `claim_status` ∈ {CONFIRMED, CONTRADICTED, UNVERIFIED}

It does **not** decide payouts/coupons/refunds — that's Module 3 (the Economist).
Staying in lane is what makes the final stitch clean.

> **Core principle:** *Never trust an unverifiable claim from the person who
> benefits from it, and never let the loudness of a message substitute for
> evidence.* Trust comes from **history**, never from the message text — so this
> module reads the customer profile, not the message body.

---

## Files

| File | Purpose |
|---|---|
| `investigator.py` | The module. Public function: `investigate(reader_output, profile)`. |
| `mocks.py` | Hardcoded mock profiles + reader output — build/test without Module 1 or 3. |
| `test_investigator.py` | Unit tests over every fraud + claim case. `python3 test_investigator.py` |
| `validate_against_dataset.py` | Runs rules on all 220 customers vs ground-truth `_archetype`. |
| `customers.csv` | Copy of the shared dataset, so validation runs out of the box. |

**Results:** 14/14 unit cases pass · **100.0% genuineness accuracy (220/220)** vs the
planted archetypes, zero misclassifications.

---

## The contract (stitch-ready)

```python
investigate(reader_output, profile) -> {
    'genuineness' : 'GENUINE' | 'SUSPICIOUS' | 'LIKELY_ABUSER',  # REQUIRED
    'claim_status': 'CONFIRMED' | 'CONTRADICTED' | 'UNVERIFIED',  # REQUIRED
    'reason'      : str,                                          # REQUIRED
    'risk_score'  : int,   # 0..100, additive (Module 3 may ignore)
    'risk_flags'  : list,  # additive transparency for the demo / human review
}
```

- `reader_output` = Module 1's dict `{'issue_type','frustration','confidence'}`.
- `profile` = one `customers.csv` row as a dict (use the shared `lookup_profile(id)`).
- The three **REQUIRED** keys match the handbook contract exactly; extras are
  purely additive and won't break any consumer that reads only the three.

### How it plugs into the stitch
```python
reader  = read_message(message)          # Module 1
verdict = investigate(reader, profile)   # Module 2  <-- THIS
decision = decide(verdict, reader, profile)  # Module 3 reads verdict['genuineness'] etc.
```

---

## Check 1 — Genuineness rules (and why these thresholds)

Thresholds were **derived from the archetype boundaries in `generate_data.py`**, not guessed:

| Archetype | ratio | items kept | complaints/30d | account age | orders |
|---|---|---|---|---|---|
| GENUINE | ≤ 0.10 | 0–1 | ≤ 2 | ≥ 6 | ≥ 15 |
| SUSPICIOUS | 0.25–0.45 *(moderate)* or new | ≤ 1 | 1–3 | ≤ 2 *(new)* | 1–4 *(new)* |
| LIKELY_ABUSER | 0.5–0.9 | 2–5 | 3–6 | 1–24 | 4–20 |

**Rule (abuse checked first; tenure can never excuse it):**

- `LIKELY_ABUSER` if `ratio ≥ 0.50` (and orders ≥ 5) **or** `items_kept ≥ 2` **or** `complaints_30d ≥ 4`
- else `SUSPICIOUS` if `ratio ≥ 0.25` **or** `age ≤ 2` **or** `first_purchase` **or** `orders < 5` **or** `complaints_30d ≥ 3` **or** (`orders < 8` and any refund)
- else `GENUINE`

---

## Thinking like a fraudster — attacks and the defenses built in

The handbook's starter rules are a good baseline but leak. Each row below is an
attack a real abuser would try, and what this module does about it.

| # | Fraud play | Why the naive rule fails | Defense here |
|---|---|---|---|
| 1 | **Keep exactly 2 items** | Handbook flags `kept ≥ 3`, so keeping 2 evades it | Lowered to `kept ≥ 2` (the data's true GENUINE/ABUSER boundary) |
| 2 | **Negation trap in notes** — "we *never promised* anything" | Substring search for `promised` returns a false CONFIRMED | Contradiction patterns checked **first**, negation-aware regex with word boundaries |
| 3 | **Phantom promise** — claims a promise with no contact on record | A logged-promise flag alone is trusted | CONFIRMED requires a **contact record to attach it to**; else UNVERIFIED |
| 4 | **Self-serving claim in the message** | Reading the claim from the angry message | Verification is **records-only**; the message text never feeds trust |
| 5 | **Aged sleeper turns abuser** | Long tenure used as a trust shortcut | Abuse signals checked **before** tenure; age never overrides them |
| 6 | **Low-denominator noise** — 2 orders, 1 refund = ratio 0.50 | Naive `ratio ≥ 0.5` brands a 2-order account a serial abuser (false accusation) | Ratio counts as abuse **only with ≥ 5 orders**; small samples fall to SUSPICIOUS (capped + verified) |
| 7 | **Ratio dilution** — many cheap orders to push ratio down | Pure-ratio rule is gamed by volume | Soft flag `possible_ratio_dilution` for human/Module 3 review (doesn't auto-deny a genuine high-volume shopper) |
| 8 | **Performed outrage** — furious message, clean history | Tone mistaken for deservingness | Trust is computed from history only; fury earns nothing |
| 9 | **Malformed/missing fields** | Crash or silent mis-read | Defensive `_num` / `_bool` coercion with safe defaults |

**Design stance:** *hard rules* drive the label for clear signals; *soft flags*
surface ambiguous fraud patterns for human review instead of auto-denying — which
directly serves the project's rule that a genuine customer must **never** be
wrongly punished.

---

## Check 2 — Claim verification (a small NLP task)

Decision order is deliberate:

1. **CONTRADICTED** — notes refute the claim (negation-aware; e.g. "no refund",
   "non-returnable", "never promised"). Checked first so a negated promise can't
   be misread as a confirmation.
2. **CONFIRMED** — a logged promise *or* an affirmative note (`promised`,
   `assured`, `committed`…), **but only with a contact record** to corroborate it.
3. **UNVERIFIED** — no record of the claimed contact. Neither honour nor accuse;
   proceed on verified facts only.

> An `LIKELY_ABUSER` can still have a `CONFIRMED` promise — we report **both facts
> truthfully** and let Module 3 decide whether to honour a promise made to an abuser.

---

## Run it

```bash
python3 test_investigator.py          # unit tests over all mock cases
python3 validate_against_dataset.py   # accuracy vs ground-truth _archetype
python3 investigator.py               # quick single-case smoke test
```
