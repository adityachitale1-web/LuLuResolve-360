"""
Unit tests for Module 2 — runs every mock case and asserts the verdict.
Run:  python3 test_investigator.py
"""
from investigator import investigate
from mocks import CASES, MOCK_READER


def run():
    passed = 0
    print("%-32s %-15s %-15s %-7s" % ("CASE", "GENUINENESS", "CLAIM_STATUS", "RESULT"))
    print("-" * 78)
    for name, profile, exp_g, exp_c in CASES:
        v = investigate(MOCK_READER, profile)
        ok_g = v["genuineness"] == exp_g
        ok_c = v["claim_status"] == exp_c
        ok = ok_g and ok_c
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print("%-32s %-15s %-15s %-7s" % (name, v["genuineness"], v["claim_status"], mark))
        if not ok:
            print("    expected genuineness=%s claim=%s" % (exp_g, exp_c))
            print("    reason: %s" % v["reason"])
    print("-" * 78)
    print("%d/%d cases passed" % (passed, len(CASES)))
    return passed == len(CASES)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
