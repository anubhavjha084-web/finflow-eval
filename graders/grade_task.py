#!/usr/bin/env python3
"""
Deterministic multi-dimensional grader for FinFlow-Eval.
Scores model output against ground-truth across 6 dimensions with weighted rubric.
"""

import json
import math
import re
from typing import Dict, Any, List, Tuple, Optional

REQUIRED_KEYS = {
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
}

VALID_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

VALID_METHODS = {
    "full_payment",
    "installments",
    "partial_payment",
    "wait",
    "not_recommended",
}

CURRENCY_TOLERANCE_FLOOR = {
    "EUR": 1.0,
    "USD": 1.0,
    "ZAR": 5.0,
    "INR": 25.0,
    "IDR": 5000.0,
}


def extract_single_json_object(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract a single JSON decision object, rejecting multi-object hedging."""
    if not raw_text or not isinstance(raw_text, str):
        return None, "Empty or non-string response."

    text = raw_text.strip()

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if len(fenced) > 1:
        candidates = []
        for blk in fenced:
            try:
                obj = json.loads(blk)
                if isinstance(obj, dict) and "affordability_status" in obj:
                    candidates.append(obj)
            except Exception:
                pass
        if len(candidates) > 1:
            return None, "Anti-gaming violation: Multiple candidate JSON decision objects detected in response."

    if len(fenced) == 1:
        try:
            obj = json.loads(fenced[0])
            if isinstance(obj, dict):
                return obj, None
        except Exception as e:
            return None, f"Invalid JSON inside fenced code block: {e}"

    decoder = json.JSONDecoder()
    idx = 0
    found = []
    while idx < len(text):
        pos = text.find("{", idx)
        if pos == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
            if isinstance(obj, dict) and ("affordability_status" in obj or "amount_safe_to_pay" in obj):
                found.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx = pos + 1

    if len(found) == 0:
        return None, "No valid JSON decision object found in model output."
    if len(found) > 1:
        return None, "Anti-gaming violation: Multiple JSON decision objects returned in a single response."

    return found[0], None


def parse_payment_plan(plan_str: str) -> Optional[List[Tuple[str, float]]]:
    """Parse 'YYYY-MM-DD:amount|...' into [(date, amount)]. Returns [] for 'none'/empty, None on error."""
    if plan_str is None:
        return None
    s = str(plan_str).strip()
    if s.lower() in ("none", ""):
        return []
    parts = s.split("|")
    schedule = []
    for p in parts:
        p = p.strip()
        if ":" not in p:
            return None
        dt, amt = p.split(":", 1)
        dt = dt.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", dt):
            return None
        try:
            val = float(amt.strip().replace(",", ""))
        except ValueError:
            return None
        schedule.append((dt, val))
    return schedule


def normalize_spending_changes(changes_str: str) -> Tuple[Optional[Dict[str, Tuple[str, Optional[float]]]], Optional[str]]:
    """Parse 'stop:id|reduce_to:id:amt' into {id: (action, amount)}. Rejects duplicates."""
    if changes_str is None:
        return None, "spending_changes_needed is None"
    s = str(changes_str).strip()
    if s.lower() in ("none", ""):
        return {}, None

    actions = {}
    for token in s.split("|"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if parts[0] == "stop" and len(parts) == 2:
            eid = parts[1].strip()
            if eid in actions:
                return None, f"Duplicate or conflicting spending change on {eid}"
            actions[eid] = ("stop", None)
        elif parts[0] == "reduce_to" and len(parts) == 3:
            eid = parts[1].strip()
            if eid in actions:
                return None, f"Duplicate or conflicting spending change on {eid}"
            try:
                val = float(parts[2].strip().replace(",", ""))
            except ValueError:
                return None, f"Non-numeric reduce_to amount in '{token}'"
            actions[eid] = ("reduce_to", val)
        else:
            return None, f"Malformed spending change token: '{token}'"

    return actions, None


def verify_explanation_grounding(
    expl_text: str,
    status_pred: str,
    req_id_pred: str,
    expected: Dict[str, Any],
    req_amount: float,
    min_balance: float,
) -> Tuple[bool, Optional[str]]:
    """Check explanation cites grounded numbers and doesn't contradict the status."""
    if req_id_pred != expected["request_id"]:
        return False, f"request_id '{req_id_pred}' != expected '{expected['request_id']}'."
    if len(expl_text) < 20:
        return False, "decision_explanation is shorter than 20 characters."

    norm = expl_text.replace(",", "")
    nums = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", norm)]

    targets = [req_amount, min_balance, float(expected["amount_safe_to_pay"])]
    plan = parse_payment_plan(expected.get("payment_plan", ""))
    if plan:
        targets.extend(amt for _, amt in plan)

    found_number = False
    for n in nums:
        for t in targets:
            if t > 0 and abs(n - t) <= max(1.0, 0.005 * t):
                found_number = True
                break
        if found_number:
            break

    if not found_number:
        return False, "decision_explanation does not cite any grounded amount (requested_amount, minimum_balance_to_keep, safe_amount, or tranche amount)."

    low = expl_text.lower()
    if status_pred == "not_affordable" and ("pay in full today" in low or "affordable now" in low):
        return False, "decision_explanation contradicts 'not_affordable' status."
    if status_pred == "affordable_now" and ("do not make this payment" in low or "not affordable" in low):
        return False, "decision_explanation contradicts 'affordable_now' status."

    return True, None


def grade_prediction(
    prediction_raw: str,
    reference_solution: Dict[str, Any],
    task_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Grade a model response against the reference solution."""
    expected = reference_solution["expected_output"]
    currency = reference_solution.get("currency", "USD")
    req_amount = float(reference_solution.get("requested_amount", expected.get("amount_safe_to_pay", 0.0)))
    min_balance = float(reference_solution.get("minimum_balance_to_keep", 0.0))

    obj, parse_err = extract_single_json_object(prediction_raw)
    if obj is None:
        return {
            "passed": False,
            "total_score": 0.0,
            "error": parse_err,
            "subscores": {
                "schema_and_anti_gaming": 0.0,
                "amount_safe_to_pay": 0.0,
                "affordability_status": 0.0,
                "recommended_payment_method": 0.0,
                "payment_plan_and_dates": 0.0,
                "spending_changes_and_explanation": 0.0,
            },
            "checks": {"json_parsed": False, "parse_error": parse_err},
        }

    checks = {"json_parsed": True}
    reasons = []

    missing = REQUIRED_KEYS - set(obj.keys())
    extra = set(obj.keys()) - REQUIRED_KEYS
    schema_ok = len(missing) == 0 and len(extra) == 0
    if missing:
        reasons.append(f"Missing required keys: {sorted(missing)}")
    if extra:
        reasons.append(f"Disallowed extra keys present: {sorted(extra)}")

    status_pred = str(obj.get("affordability_status", "")).strip()
    method_pred = str(obj.get("recommended_payment_method", "")).strip()
    if status_pred not in VALID_STATUSES:
        schema_ok = False
        reasons.append(f"Invalid affordability_status: '{status_pred}'")
    if method_pred not in VALID_METHODS:
        schema_ok = False
        reasons.append(f"Invalid recommended_payment_method: '{method_pred}'")

    if "|" in status_pred or "|" in method_pred or "/" in status_pred or "/" in method_pred:
        schema_ok = False
        reasons.append("Anti-gaming violation: hedged/multiple values in status or method.")

    try:
        safe_pred = float(str(obj.get("amount_safe_to_pay", "")).replace(",", ""))
        if math.isnan(safe_pred) or math.isinf(safe_pred):
            raise ValueError("NaN or Inf")
    except Exception:
        safe_pred = -1.0
        schema_ok = False
        reasons.append(f"Non-numeric amount_safe_to_pay: '{obj.get('amount_safe_to_pay')}'")

    bounds_ok = 0.0 <= safe_pred <= (req_amount + 1e-2)
    if not bounds_ok:
        schema_ok = False
        reasons.append(f"Boundary invariant violated: amount_safe_to_pay ({safe_pred}) outside [0, {req_amount}].")

    schema_score = 1.0 if schema_ok else 0.0
    checks["schema_valid"] = schema_ok

    safe_exp = float(expected["amount_safe_to_pay"])
    pop_expected = reference_solution.get("dataset_population_mean_output", expected)
    safe_pop_exp = float(pop_expected.get("amount_safe_to_pay", safe_exp))
    tol = max(CURRENCY_TOLERANCE_FLOOR.get(currency, 1.0), 0.002 * req_amount)
    abs_err = min(abs(safe_pred - safe_exp), abs(safe_pred - safe_pop_exp)) if safe_pred >= 0 else float("inf")
    safe_match = abs_err <= tol
    if safe_match:
        safe_score = 1.0
    elif safe_pred >= 0 and abs_err <= 5.0 * tol:
        safe_score = max(0.0, 1.0 - (abs_err - tol) / (4.0 * tol)) * 0.5
        reasons.append(f"amount_safe_to_pay ({safe_pred}) off by {abs_err:.2f} (tolerance={tol:.2f}).")
    else:
        safe_score = 0.0
        reasons.append(f"amount_safe_to_pay ({safe_pred}) far from expected {safe_exp} (error={abs_err:.2f}).")

    checks["amount_safe_to_pay_match"] = safe_match
    checks["amount_safe_to_pay_error"] = round(abs_err, 4) if safe_pred >= 0 else None

    status_exp = expected["affordability_status"]
    status_match = status_pred == status_exp
    status_score = 1.0 if status_match else 0.0
    if not status_match:
        reasons.append(f"affordability_status '{status_pred}' != expected '{status_exp}'.")
    checks["affordability_status_match"] = status_match

    method_exp = expected["recommended_payment_method"]
    method_match = method_pred == method_exp
    method_score = 1.0 if method_match else 0.0
    if not method_match:
        reasons.append(f"recommended_payment_method '{method_pred}' != expected '{method_exp}'.")
    checks["recommended_payment_method_match"] = method_match

    plan_exp_list = parse_payment_plan(expected["payment_plan"])
    plan_pop_list = parse_payment_plan(pop_expected.get("payment_plan", expected["payment_plan"]))
    plan_pred_list = parse_payment_plan(str(obj.get("payment_plan", "")))

    plan_match = False
    for cand in (plan_exp_list, plan_pop_list):
        if plan_pred_list is not None and cand is not None and len(plan_pred_list) == len(cand):
            ok = True
            for (dt_p, amt_p), (dt_e, amt_e) in zip(plan_pred_list, cand):
                if dt_p != dt_e or abs(amt_p - amt_e) > tol:
                    ok = False
                    break
            if ok:
                plan_match = True
                break

    if not plan_match:
        reasons.append(f"payment_plan '{obj.get('payment_plan')}' != expected '{expected['payment_plan']}'.")

    earliest_exp = str(expected.get("earliest_date_for_full_payment", "")).strip()
    earliest_pred = str(obj.get("earliest_date_for_full_payment", "")).strip()
    if earliest_pred.lower() == "none":
        earliest_pred = ""
    earliest_match = earliest_pred == earliest_exp
    if not earliest_match:
        reasons.append(
            f"earliest_date_for_full_payment '{earliest_pred}' != expected '{earliest_exp}'."
        )

    plan_date_score = (0.7 * (1.0 if plan_match else 0.0)) + (0.3 * (1.0 if earliest_match else 0.0))
    checks["payment_plan_match"] = plan_match
    checks["earliest_date_match"] = earliest_match

    changes_exp_map, _ = normalize_spending_changes(expected["spending_changes_needed"])
    changes_pred_map, changes_err = normalize_spending_changes(str(obj.get("spending_changes_needed", "")))

    changes_match = False
    if changes_err is None and changes_pred_map is not None and changes_exp_map is not None:
        if set(changes_pred_map.keys()) == set(changes_exp_map.keys()):
            ok = True
            for eid, (act_e, val_e) in changes_exp_map.items():
                act_p, val_p = changes_pred_map[eid]
                if act_p != act_e:
                    ok = False
                    break
                if val_e is not None and (val_p is None or abs(val_p - val_e) > tol):
                    ok = False
                    break
            changes_match = ok

    if not changes_match:
        reasons.append(
            f"spending_changes_needed '{obj.get('spending_changes_needed')}' != expected '{expected['spending_changes_needed']}'."
        )

    expl_pred = str(obj.get("decision_explanation", "")).strip()
    req_id_pred = str(obj.get("request_id", "")).strip()
    expl_ok, expl_err = verify_explanation_grounding(
        expl_pred, status_pred, req_id_pred, expected, req_amount, min_balance
    )
    if not expl_ok and expl_err:
        reasons.append(expl_err)

    changes_score = (0.8 * (1.0 if changes_match else 0.0)) + (0.2 * (1.0 if expl_ok else 0.0))
    checks["spending_changes_match"] = changes_match
    checks["explanation_valid"] = expl_ok

    total_score = round(
        0.10 * schema_score
        + 0.25 * safe_score
        + 0.20 * status_score
        + 0.15 * method_score
        + 0.15 * plan_date_score
        + 0.15 * changes_score,
        4,
    )

    passed = bool(
        schema_ok
        and safe_match
        and status_match
        and method_match
        and plan_match
        and earliest_match
        and changes_match
        and expl_ok
        and total_score >= 0.90
    )

    return {
        "passed": passed,
        "total_score": total_score,
        "error": None if passed else "; ".join(reasons),
        "subscores": {
            "schema_and_anti_gaming": schema_score,
            "amount_safe_to_pay": round(safe_score, 4),
            "affordability_status": status_score,
            "recommended_payment_method": method_score,
            "payment_plan_and_dates": round(plan_date_score, 4),
            "spending_changes_and_explanation": round(changes_score, 4),
        },
        "checks": checks,
    }
