# Module 1 — The Reader (LuluCare 360)

The **Reader** is the language-understanding component of the LuluCare 360 co-pilot. It
turns a raw customer complaint (free text) into a **structured, machine-readable
understanding** that the rest of the system (Investigator, Economist, Voice) can act on.

This README is the **integration contract**. Any teammate building Module 2, 3 or 4 can
develop and test against it in parallel — the contract *is* the integration.

---

## 1. What this module does

| Input | Output |
|---|---|
| `text` — the customer's complaint as a string | A `dict` with `issue_type`, `frustration`, and `confidence` |

It is powered by **two LSTM models** trained in `Module1_The_Reader.ipynb`:

1. **Issue classifier** — `Embedding → LSTM(64) → Dropout → Dense(32) → Dense(7, softmax)`
   classifies the message into one of **7 issue types**.
2. **Frustration classifier** — same architecture with a `Dense(3, softmax)` head,
   classifies emotional intensity as **Low / Medium / High**.

A `SimpleRNN` variant is trained only for comparison and is **not** part of the contract.

---

## 2. The contract function

```python
def read_message(text: str) -> dict:
    """Module 1 contract: text in -> structured understanding out."""
    seq = pad_sequences(tok.texts_to_sequences([text]), maxlen=MAXLEN)
    prob = lstm_model.predict(seq, verbose=0)[0]
    idx = int(prob.argmax())
    return {
        'issue_type':  issue_labels[idx],          # one of the 7 issue types
        'frustration': predict_frustration(text),  # 'Low' | 'Medium' | 'High'
        'confidence':  round(float(prob[idx]), 3)  # softmax confidence, 0.0–1.0
    }
```

### Output schema

```jsonc
{
  "issue_type":  "Delivery",   // string, one of the 7 values below
  "frustration": "High",       // string: "Low" | "Medium" | "High"
  "confidence":  0.972          // float in [0.0, 1.0], the max softmax probability
}
```

### Allowed values

**`issue_type`** (exactly these 7 strings):

```
Delivery
Damaged_Defective
Refund_Return
Billing
Product_Quality
App_Technical
General_Query
```

**`frustration`**: `Low`, `Medium`, `High`

**`confidence`**: a float in `[0.0, 1.0]`. Higher = the model is more certain.

---

## 3. How downstream modules consume the output

| Field | Consumed by | Used for |
|---|---|---|
| `issue_type` | **Economist (Module 3)** | Picks the remediation branch (e.g. `Damaged_Defective` → refund + keep-vs-pickup tree). |
| `frustration` | **Economist (Module 3)** | Scales the *generosity* of the remedy (e.g. `Medium` → coupon, `High` + high-value customer → refund). Never used alone — frustration ≠ genuineness. |
| `confidence` | **Investigator / Economist** | Fed into `should_escalate()`; a low value routes the case to a human. |
| original `text` | **Voice (Module 4)** | Combined with the Economist's decision to draft the reply. |

Because the output is a plain `dict` with fixed keys, each downstream team can **mock** it
during development and the final integration is a simple stitch.

---

## 4. Quick start

### Run the notebook (Google Colab — recommended)

1. Open `Module1_The_Reader.ipynb` in Google Colab.
2. Upload the data files when prompted (or via the folder icon on the left):
   - `messages.csv` — training data (required)
   - `customers (2).csv` — customer history (used by Modules 2–3)
3. Run all cells top to bottom. The models train in a few minutes on a CPU runtime.
4. The final cells expose `read_message()` and `predict_frustration()`.

> Colab starts with an empty file system, so the CSVs **must** be uploaded each session.
> TensorFlow does not yet support the very latest local Python builds, so Colab is the
> supported environment.

### Use the contract

```python
read_message('My order never arrived and I am furious!')
# -> {'issue_type': 'Delivery', 'frustration': 'High', 'confidence': 0.97}
```

---

## 5. Integrating into the full pipeline

To use the Reader from another module **in the same runtime/notebook**, simply call
`read_message()` after the training cells have run. To package it as a reusable component,
expose the trained objects through a small wrapper:

```python
# reader.py (conceptual)  -- the objects the notebook builds:
#   lstm_model, frus_model, tok, MAXLEN, issue_labels, frus_labels
class Reader:
    def __init__(self, lstm_model, frus_model, tok, maxlen, issue_labels, frus_labels):
        self.lstm_model, self.frus_model = lstm_model, frus_model
        self.tok, self.maxlen = tok, maxlen
        self.issue_labels, self.frus_labels = issue_labels, frus_labels

    def _seq(self, text):
        return pad_sequences(self.tok.texts_to_sequences([text]), maxlen=self.maxlen)

    def read_message(self, text: str) -> dict:
        prob = self.lstm_model.predict(self._seq(text), verbose=0)[0]
        idx = int(prob.argmax())
        fprob = self.frus_model.predict(self._seq(text), verbose=0)[0]
        return {
            'issue_type':  self.issue_labels[idx],
            'frustration': self.frus_labels[int(fprob.argmax())],
            'confidence':  round(float(prob[idx]), 3),
        }
```

### Persisting the trained artifacts (so other modules can load them)

```python
# Save (run once, after training)
lstm_model.save('issue_model.keras')
frus_model.save('frustration_model.keras')
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tok, f)

# Load (in any other module)
from tensorflow.keras.models import load_model
issue_model = load_model('issue_model.keras')
frus_model  = load_model('frustration_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)
```

Keep `MAXLEN`, `issue_labels` and `frus_labels` identical to the values in the notebook —
they are part of the contract.

---

## 6. Key constants (must match across modules)

| Constant | Value | Meaning |
|---|---|---|
| `VOCAB` | `3000` | Tokenizer vocabulary size (`num_words`) |
| `MAXLEN` | `40` | Padded sequence length |
| `oov_token` | `'<OOV>'` | Out-of-vocabulary placeholder |
| `issue_labels` | 7 strings (see §2) | Index → issue name mapping |
| `frus_labels` | `['Low', 'Medium', 'High']` | Index → frustration level mapping |

---

## 7. Files in this folder

| File | Purpose |
|---|---|
| `Module1_The_Reader.ipynb` | The deliverable: builds, trains, evaluates and exposes the Reader. |
| `messages.csv` | Labelled complaints (630 rows) used to train the two models. |
| `customers (2).csv` | Customer history (220 rows) for Modules 2–3. |
| `generate_data.py` | Deterministic (`SEED=42`) generator that produces the CSVs. |
| `README.md` | This integration contract. |

---

## 8. Confidence policy (how much to trust the Reader)

`confidence` is the **automation dial** the Economist uses:

- **High confidence** → safe to act automatically (draft and send reply).
- **Medium confidence** → act, but prefer reversible / capped actions.
- **Low confidence** (e.g. `< 0.5`) → **route to a human**, especially when money or a
  high-value customer is involved.

This turns a single softmax number into a transparent, defensible control over how much
the co-pilot is trusted to act on its own.

---

## 9. Module 2 integration (the stitch)

Module 2 — **The Investigator** — is now wired into this repo. It consumes the
Reader's `{issue_type, frustration, confidence}` plus the customer's history and
returns a trust verdict for Module 3.

| File | Purpose |
|---|---|
| `investigator.py` | Module 2: `investigate(reader_output, profile)` → `{genuineness, claim_status, reason, risk_score, risk_flags}`. Transparent rules. |
| `pipeline.py` | The stitch: `run_pipeline(message, customer_id, read_message, lookup_profile)`. Runs offline with a mock Reader, or with the real `read_message` in Colab. |
| `mocks.py` / `test_investigator.py` | 14 mock fraud/claim cases — `python3 test_investigator.py`. |
| `validate_against_dataset.py` | Validates the rules vs the dataset's ground-truth `_archetype` — **100% (220/220)**. |
| `MODULE2_README.md` | Full rule set, the fraud loopholes it closes, and the stitch contract. |

```bash
python3 pipeline.py              # end-to-end stitch demo (no TensorFlow needed)
python3 test_investigator.py     # 14/14 unit cases pass
python3 validate_against_dataset.py
```

In the notebook, the **"Module 2 — The Investigator (live stitch)"** section at the
end runs the real Reader into the Investigator end-to-end. Trust is judged from
**history, not tone** — an angry message from a serial abuser is still flagged
`LIKELY_ABUSER`.


### Interactive UI — `lulucare_ui.html`

A self-contained, single-file web demo of the Module 1 + Module 2 stitch — open it
in any browser, no server or install needed. Type a complaint and pick a customer:
the Reader classifies the message and the Investigator returns the trust verdict
(genuineness, claim status, risk score, and the exact signals that fired). The
Module 2 rules run live in JavaScript, mirroring `investigator.py` one-to-one, so
the angry-message-from-an-abuser case still resolves to `LIKELY_ABUSER` — trust
from history, not tone. Light/dark themed.

```bash
open lulucare_ui.html      # or just double-click it
```

---

## 10. Module 3 integration (the Economist)

Module 3 — **The Economist** — is now wired into the pipeline. It consumes the
Reader output + Investigator verdict + profile and emits the decision contract
Module 4 consumes.

| File | Purpose |
|---|---|
| `module3_resolution_engine.py` | Module 3: `decide_resolution(reader, verdict, profile)` (and a handbook-signature `decide(verdict, reader, profile)` alias) → `{action, refund_type, coupon_percent, wallet_credit, escalate, email_trigger, value_band, risk_score, resolution_score, reason, audit_trail}`. Pure stdlib, transparent rules. |
| `pipeline.py` | `run_full_pipeline(...)` now chains M1 → M2 → M3. |
| `test_pipeline.py` | M2 → M3 over all 220 customers: contract invariants + automation rate. |
| `lulucare_ui.html` | UI now shows all three stages, including the Economist's decision. |

Covers every required check: value band (Check 3), remediation tier (Check 4),
refund-logistics tree (Check 5), escalation valve (Check 6), email trigger (Check 7),
and the automation-rate metric.

```bash
python3 module3_resolution_engine.py   # 10/10 mandated scenarios, ~90% automation
python3 test_pipeline.py               # full stitch over 220 customers, 94.5% automation
```

Verified: 0 contract violations across 220 customers; email fires only on
COUPON / WALLET_CREDIT / REFUND; the JS port in the UI matches the Python engine 1:1.

> Note for Module 4: `wallet_credit` is denominated in USD here; align the
> currency label in the Voice's reply text with the rest of the team.
