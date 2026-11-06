#!/usr/bin/env python3
"""Validation suite for FinFlow-Eval graders: positive, formatting-variant, and adversarial checks."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graders.grade_task import grade_prediction


def run_grader_validation_suite() -> dict:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = os.path.join(base, "reference")
    tasks_dir = os.path.join(base, "tasks")

    summary = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "task_validations": [],
    }

    for i in range(1, 11):
        tid = f"task_{i:02d}"
        with open(os.path.join(ref_dir, f"{tid}_solution.json"), "r", encoding="utf-8") as f:
            ref = json.load(f)
        with open(os.path.join(tasks_dir, f"{tid}.json"), "r", encoding="utf-8") as f:
            task = json.load(f)

        exp = ref["expected_output"]
        cases = []

        # Exact reference
        r1 = grade_prediction(json.dumps(exp), ref, task)
        ok1 = r1["passed"] is True and r1["total_score"] == 1.0
        cases.append(("positive_exact_reference", ok1, r1["total_score"]))

        # Fenced markdown with reordered pipe-delimited spending changes
        v = dict(exp)
        if "|" in v["spending_changes_needed"]:
            parts = v["spending_changes_needed"].split("|")
            v["spending_changes_needed"] = "|".join(reversed(parts))
        fenced = f"Here is the audit result:\n```json\n{json.dumps(v, indent=2)}\n```"
        r2 = grade_prediction(fenced, ref, task)
        ok2 = r2["passed"] is True and r2["total_score"] == 1.0
        cases.append(("positive_fenced_and_reordered", ok2, r2["total_score"]))

        # Naive balance-minus-minimum heuristic
        naive = dict(exp)
        prof = task["input_data"]["user_profile"]
        req = task["input_data"]["request"]
        headroom = min(
            float(req["requested_amount"]),
            max(0.0, float(prof["current_available_balance"]) - float(prof["minimum_balance_to_keep"])),
        )
        naive["amount_safe_to_pay"] = headroom
        naive["affordability_status"] = "affordable_now" if headroom >= float(req["requested_amount"]) else "affordable_later"
        naive["recommended_payment_method"] = "full_payment" if headroom >= float(req["requested_amount"]) else "wait"
        naive["spending_changes_needed"] = "none"
        if tid == "task_01":
            naive["amount_safe_to_pay"] = float(req["requested_amount"]) - 816.20
        r3 = grade_prediction(json.dumps(naive), ref, task)
        ok3 = r3["passed"] is False
        cases.append(("negative_naive_or_cancelled_trap", ok3, r3["total_score"]))

        # Wrong affordability status & method
        ws = dict(exp)
        ws["affordability_status"] = (
            "not_affordable" if exp["affordability_status"] != "not_affordable" else "affordable_now"
        )
        r4 = grade_prediction(json.dumps(ws), ref, task)
        ok4 = r4["passed"] is False
        cases.append(("negative_wrong_status", ok4, r4["total_score"]))

        # Out-of-bounds amount_safe_to_pay
        oob = dict(exp)
        oob["amount_safe_to_pay"] = float(req["requested_amount"]) * 1.25 + 500.0
        r5 = grade_prediction(json.dumps(oob), ref, task)
        ok5 = r5["passed"] is False and r5["checks"]["schema_valid"] is False
        cases.append(("negative_out_of_bounds_amount", ok5, r5["total_score"]))

        # Anti-gaming kitchen-sink multiple JSON objects
        alt = dict(exp)
        alt["affordability_status"] = "not_affordable"
        ks = f"```json\n{json.dumps(exp)}\n```\nOr alternatively:\n```json\n{json.dumps(alt)}\n```"
        r6 = grade_prediction(ks, ref, task)
        ok6 = r6["passed"] is False and r6["total_score"] == 0.0
        cases.append(("negative_anti_gaming_multi_json", ok6, r6["total_score"]))

        # Corrupted spending_changes_needed
        bc = dict(exp)
        bc["spending_changes_needed"] = "stop:event_999999"
        r7 = grade_prediction(json.dumps(bc), ref, task)
        ok7 = r7["passed"] is False
        cases.append(("negative_wrong_spending_changes", ok7, r7["total_score"]))

        # Disallowed extra key
        ek = dict(exp)
        ek["extra_debug_field"] = "not_allowed"
        r8 = grade_prediction(json.dumps(ek), ref, task)
        ok8 = r8["passed"] is False and r8["checks"]["schema_valid"] is False
        cases.append(("negative_disallowed_extra_keys", ok8, r8["total_score"]))

        # Ungrounded explanation
        ug = dict(exp)
        ug["decision_explanation"] = "This payment decision was chosen after reviewing the user's monthly cash flow and upcoming expenses."
        r9 = grade_prediction(json.dumps(ug), ref, task)
        ok9 = r9["passed"] is False and r9["checks"]["explanation_valid"] is False
        cases.append(("negative_ungrounded_explanation", ok9, r9["total_score"]))

        for name, passed_check, score in cases:
            summary["total_tests"] += 1
            if passed_check:
                summary["passed_tests"] += 1
            else:
                summary["failed_tests"] += 1

        summary["task_validations"].append({
            "task_id": tid,
            "all_checks_passed": all(c[1] for c in cases),
            "cases": [
                {"test_name": c[0], "assertion_passed": c[1], "grader_score": c[2]}
                for c in cases
            ],
        })

    return summary


if __name__ == "__main__":
    report = run_grader_validation_suite()
    print(json.dumps(report, indent=2))
    if report["failed_tests"] > 0:
        sys.exit(1)
    print(f"\nAll {report['passed_tests']}/{report['total_tests']} positive & adversarial grader checks PASSED.")
