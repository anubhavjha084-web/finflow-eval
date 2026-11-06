#!/usr/bin/env python3
"""Daily cash-flow forecasting and payment-plan evaluation for FinFlow tasks."""

from __future__ import annotations

import calendar
import itertools
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def add_months(day: date, offset: int, anchor_day: int | None = None) -> date:
    mi = day.month - 1 + offset
    yr = day.year + mi // 12
    mo = mi % 12 + 1
    td = min(anchor_day or day.day, calendar.monthrange(yr, mo)[1])
    return date(yr, mo, td)


def fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.2f}"


def _fx_converter(inp: dict[str, Any], home: str):
    rates: dict[tuple[str, str], list[tuple[date | None, float]]] = defaultdict(list)
    for row in inp.get("exchange_rates", []):
        fc = row.get("from_currency")
        tc = row.get("to_currency")
        r = float(row["rate"])
        rd_raw = row.get("rate_date") or row.get("effective_date") or row.get("date")
        rd = parse_date(rd_raw) if rd_raw else None
        rates[(fc, tc)].append((rd, r))
    for vals in rates.values():
        vals.sort(key=lambda item: item[0] or date.min)

    def convert(amount: float, currency: str, when: date) -> float:
        if currency == home:
            return round(amount, 2)
        cands = rates.get((currency, home), [])
        prior = [item for item in cands if item[0] is None or item[0] <= when]
        if prior:
            rate = prior[-1][1]
        elif cands:
            rate = cands[0][1]
        else:
            raise ValueError(f"No FX rate from {currency} to {home} for {when}.")
        return round(amount * rate, 2)

    return convert


def parse_message_overrides(messages: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "salary_amount": None,
        "salary_currency": None,
        "salary_effective_date": None,
        "salary_day": None,
        "temporary_amount": None,
        "income_terminated": False,
        "exclude_commissions": False,
    }
    patterns = (
        r"gaji bulanan anda naik menjadi\s+([A-Z]{3})\s+([\d.,]+)",
        r"gaji pokok yang dikonfirmasi adalah\s+([A-Z]{3})\s+([\d.,]+)",
        r"temporary monthly pay is\s+([A-Z]{3})\s+([\d.,]+)",
    )
    for msg in messages:
        text = msg.get("message_text", "")
        low = text.lower()
        if "contract has ended" in low or "no off-season income" in low:
            result["income_terminated"] = True
        if ("belum disetujui" in low and "komisi" in low) or (
            "commission" in low and ("not approved" in low or "not yet approved" in low)
        ):
            result["exclude_commissions"] = True
        for pat in patterns:
            match = re.search(pat, text, flags=re.IGNORECASE)
            if not match:
                continue
            cur, raw_amt = match.groups()
            amt = float(raw_amt.replace(",", "").rstrip("."))
            if "temporary monthly pay" in pat:
                result["temporary_amount"] = (cur.upper(), amt)
            else:
                result["salary_amount"] = (cur.upper(), amt)
                eff = re.search(r"berlaku mulai\s+(\d{4}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
                result["salary_effective_date"] = parse_date(eff.group(1)) if eff else None
        payday = re.search(r"confirmed salary is now expected on\s+(\d{4}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
        if payday:
            result["salary_day"] = parse_date(payday.group(1)).day
    return result


def _cadence(events: list[dict[str, Any]]) -> tuple[str, int | None] | None:
    dates = sorted(parse_date(e.get("settlement_date") or e["event_date"]) for e in events)
    if len(dates) < 3:
        return None
    gaps = sorted((b - a).days for a, b in zip(dates, dates[1:]))
    med = gaps[len(gaps) // 2]
    if 26 <= med <= 35:
        return ("monthly", None)
    steps = (5, 7, 10, 14, 21)
    step = min(steps, key=lambda n: abs(n - med))
    if abs(step - med) <= 1:
        return ("interval", step)
    return None


def _recurrence_dates(events: list[dict[str, Any]], start: date, end: date) -> list[date]:
    cad = _cadence(events)
    if not cad:
        return []
    kind, step = cad
    edates = sorted(parse_date(e.get("settlement_date") or e["event_date"]) for e in events)
    latest = edates[-1]
    if kind == "interval":
        assert step is not None
        nxt = latest + timedelta(days=step)
        while nxt < start:
            nxt += timedelta(days=step)
        out = []
        while nxt <= end:
            out.append(nxt)
            nxt += timedelta(days=step)
        return out

    # Monthly: use modal day; detect month-end patterns.
    me_count = sum(d.day == calendar.monthrange(d.year, d.month)[1] for d in edates)
    use_me = me_count >= max(3, (len(edates) + 1) // 2)
    dcounts: dict[int, int] = defaultdict(int)
    for d in edates:
        dcounts[d.day] += 1
    anchor = max(dcounts, key=lambda n: (dcounts[n], n))
    mstart = date(latest.year, latest.month, 1)
    out: list[date] = []
    off = 1
    while True:
        cm = add_months(mstart, off, 1)
        if use_me:
            cand = date(cm.year, cm.month, calendar.monthrange(cm.year, cm.month)[1])
        else:
            cand = add_months(cm, 0, anchor)
        if cand > end:
            break
        if cand >= start and cand > latest:
            out.append(cand)
        off += 1
    return out


def _forecast(task_data: dict[str, Any], apply_msgs: bool, allow_cuts: bool, filter_nc: bool):
    inp = task_data["input_data"]
    profile, request = inp["user_profile"], inp["request"]
    home = profile["home_currency"]
    start = parse_date(request["request_date"])
    end = start + timedelta(days=90)
    convert = _fx_converter(inp, home)
    overrides = parse_message_overrides(inp.get("messages", [])) if apply_msgs else parse_message_overrides([])

    flows: dict[date, list[tuple[str, str, float, str, bool]]] = defaultdict(list)
    expenses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    salaries: list[dict[str, Any]] = []
    commissions: list[dict[str, Any]] = []
    sched_sal: dict[date, tuple[float, str]] = {}
    nc_adj = 0.0

    for ev in inp.get("financial_events", []):
        st = ev.get("status")
        dirn = ev.get("direction")
        cat = ev.get("category", "")
        ed_raw = ev.get("settlement_date") or ev.get("event_date")
        ed = parse_date(ed_raw) if ed_raw else None
        if st in {"cancelled", "failed"}:
            continue
        if st == "unrealized" or dirn == "non_cash":
            if not filter_nc and ev.get("amount"):
                ed = parse_date(ev.get("event_date") or request["request_date"])
                val = convert(float(ev["amount"]), ev.get("currency", home), ed)
                nc_adj += val
            continue
        if not ev.get("amount") or ed is None:
            continue
        amt = float(ev["amount"])

        if st == "settled" and ed < start:
            if dirn == "debit":
                if ev.get("linked_event_id") or ev.get("event_type") == "investment_purchase":
                    continue
                expenses[cat].append(ev)
            elif dirn == "credit" and cat == "salary":
                desc = ev.get("description", "").lower()
                if "commission" in desc or "bonus" in desc:
                    commissions.append(ev)
                else:
                    salaries.append(ev)
            continue

        if ed < start or ed > end:
            continue
        if st == "pending" and dirn == "credit":
            continue
        if st in {"pending", "scheduled"}:
            ha = convert(amt, ev.get("currency", home), ed)
            kind = "income" if dirn == "credit" else "expense"
            if dirn == "credit" and cat == "salary" and st == "scheduled":
                sched_sal[ed] = (ha, ev.get("event_id", "scheduled_salary"))
            elif dirn in {"credit", "debit"}:
                flows[ed].append((kind, cat, ha, ev.get("event_id", ""), False))
        elif st == "settled" and ed >= start:
            ha = convert(amt, ev.get("currency", home), ed)
            kind = "income" if dirn == "credit" else "expense"
            flows[ed].append((kind, cat, ha, ev.get("event_id", ""), False))

    recurring: dict[str, dict[str, Any]] = {}
    for cat, evts in expenses.items():
        recur = _recurrence_dates(evts, start, end)
        if not recur:
            continue
        dated_amts = []
        for ev in evts:
            ed = parse_date(ev.get("settlement_date") or ev["event_date"])
            dated_amts.append(convert(float(ev["amount"]), ev.get("currency", home), ed))
        mean_amt = round(sum(dated_amts) / len(dated_amts), 2)
        lat = max(evts, key=lambda e: parse_date(e.get("settlement_date") or e["event_date"]))
        min_ok = float(lat["minimum_allowed_amount"]) if lat.get("minimum_allowed_amount") else 0.0
        recurring[cat] = {
            "amount": mean_amt,
            "latest_event_id": lat.get("event_id", ""),
            "flexibility": lat.get("flexibility", "fixed"),
            "min_allowed": min_ok,
            "occurrences": recur,
        }
        for occ in recur:
            flows[occ].append(("expense", cat, mean_amt, lat.get("event_id", ""), True))

    if not overrides["income_terminated"]:
        salaries.sort(key=lambda e: parse_date(e.get("settlement_date") or e["event_date"]))
        base_ev = salaries[-1] if salaries else None
        if sched_sal:
            first_sd = min(sched_sal)
            base_amt = sched_sal[first_sd][0]
            sal_day = first_sd.day
        elif base_ev:
            last_sd = parse_date(base_ev.get("settlement_date") or base_ev["event_date"])
            base_amt = convert(float(base_ev["amount"]), base_ev.get("currency", home), last_sd)
            sal_day = last_sd.day
            first_sd = date(start.year, start.month, sal_day)
            if first_sd < start:
                first_sd = add_months(first_sd, 1, sal_day)
        else:
            first_sd = None
            base_amt = 0.0
            sal_day = 15

        if overrides["salary_day"]:
            sal_day = overrides["salary_day"]
            if sched_sal:
                first_sd = date(start.year, start.month, sal_day)
                if first_sd < start:
                    first_sd = add_months(first_sd, 1, sal_day)

        if first_sd is not None:
            # An official date in a payroll notice anchors the revised schedule.
            for msg in (inp.get("messages", []) if apply_msgs else []):
                m = re.search(r"(\d{4}-\d{2}-\d{2})", msg.get("message_text", ""))
                txt = msg.get("message_text", "").lower()
                if m and "salary is now expected on" in txt:
                    first_sd = parse_date(m.group(1))
                    break

            paydays: list[date] = []
            off = 0
            while True:
                pd = add_months(first_sd, off, sal_day)
                if pd > end:
                    break
                if pd >= start:
                    paydays.append(pd)
                off += 1

            sal_chg = overrides["salary_amount"]
            sal_chg_home = None
            sal_eff = overrides["salary_effective_date"]
            if sal_chg:
                cur, raw_amt = sal_chg
                sal_eff = sal_eff or (paydays[0] if paydays else start)
                sal_chg_home = convert(raw_amt, cur, sal_eff)
            tmp = overrides["temporary_amount"]
            tmp_home = convert(tmp[1], tmp[0], paydays[0]) if tmp and paydays else None

            for idx, pd in enumerate(paydays):
                if pd in sched_sal:
                    a, eid = sched_sal[pd]
                else:
                    a, eid = base_amt, "projected_salary"
                if sal_chg_home is not None and pd >= sal_eff:
                    a, eid = sal_chg_home, "message_salary_override"
                if idx == 0 and tmp_home is not None:
                    a, eid = tmp_home, "temporary_pay_override"
                flows[pd].append(("income", "salary", a, eid, True))

    if not apply_msgs and commissions:
        commissions.sort(key=lambda e: parse_date(e.get("settlement_date") or e["event_date"]))
        comm_dates = _recurrence_dates(commissions, start, end)
        if comm_dates:
            vals = []
            for ev in commissions:
                d = parse_date(ev.get("settlement_date") or ev["event_date"])
                vals.append(convert(float(ev["amount"]), ev.get("currency", home), d))
            mean_comm = round(sum(vals) / len(vals), 2)
            for d in comm_dates:
                flows[d].append(("income", "salary_commission", mean_comm, "projected_commission", True))

    candidates: list[dict[str, Any]] = []
    if allow_cuts:
        for cat, info in recurring.items():
            if not info["occurrences"]:
                continue
            flex = info["flexibility"]
            if cat in profile.get("expense_categories_user_is_willing_to_stop", "").split("|") and flex == "stoppable":
                candidates.append({"category": cat, "kind": "stop", "event_id": info["latest_event_id"], "minimum": 0.0})
            if cat in profile.get("expense_categories_user_is_willing_to_reduce", "").split("|") and "reducible" in flex:
                candidates.append({"category": cat, "kind": "reduce_to", "event_id": info["latest_event_id"], "minimum": info["min_allowed"]})

    return {
        "input": inp,
        "profile": profile,
        "request": inp["request"],
        "start": start,
        "end": end,
        "home": home,
        "min_balance": float(profile["minimum_balance_to_keep"]),
        "available": float(profile["current_available_balance"]) + nc_adj,
        "requested": float(inp["request"]["requested_amount"]),
        "deadline": parse_date(inp["request"]["desired_completion_date"]),
        "flows": flows,
        "recurring_meta": recurring,
        "candidates": candidates,
    }


def _min_balance(state: dict[str, Any], payments: dict[date, float] | None = None, actions: tuple[dict[str, Any], ...] = ()) -> float:
    payments = payments or {}
    amap = {(a["category"], a["kind"]): a["minimum"] for a in actions}
    bal = state["available"]
    lowest = bal
    for off in range(91):
        day = state["start"] + timedelta(days=off)
        recs = state["flows"].get(day, [])
        delta = 0.0
        for kind, cat, amt, _eid, recurrent in recs:
            if kind == "income":
                delta += amt
                continue
            if recurrent and kind == "expense":
                if (cat, "stop") in amap:
                    continue
                if (cat, "reduce_to") in amap:
                    amt = min(amt, amap[(cat, "reduce_to")])
            delta -= amt
        bal += delta
        bal -= payments.get(day, 0.0)
        lowest = min(lowest, bal)
    return round(lowest, 2)


def _pay_is_safe(state: dict[str, Any], schedule: dict[date, float], actions: tuple[dict[str, Any], ...] = ()) -> bool:
    return _min_balance(state, schedule, actions) >= state["min_balance"] - 0.01


def simulate_decision(
    task_data: dict[str, Any],
    apply_messages: bool = True,
    allow_spending_cuts: bool = True,
    allow_partial_payment: bool = True,
    filter_non_cash: bool = True,
) -> dict[str, Any]:
    state = _forecast(task_data, apply_messages, allow_spending_cuts, filter_non_cash)
    req, start, end = state["request"], state["start"], state["end"]
    requested, floor = state["requested"], state["min_balance"]
    profile = state["profile"]
    allowed = {v.strip() for v in profile["payment_methods_user_will_consider"].split("|") if v.strip()}
    partial_on = str(req.get("allows_partial_payment", "false")).lower() == "true" and allow_partial_payment
    base_min = _min_balance(state)
    safe_today = round(min(requested, max(0.0, base_min - floor)), 2)

    earliest_full = ""
    for off in range(91):
        day = start + timedelta(days=off)
        if _pay_is_safe(state, {day: requested}):
            earliest_full = day.isoformat()
            break

    def result(status: str, method: str, plan: str, changes: str = "none", explanation: str = ""):
        return {
            "request_id": req["request_id"],
            "amount_safe_to_pay": safe_today,
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": plan,
            "earliest_date_for_full_payment": earliest_full,
            "spending_changes_needed": changes,
            "decision_explanation": explanation,
        }

    if safe_today >= requested and "full_payment" in allowed:
        return result("affordable_now", "full_payment", f"{start}:{fmt_num(requested)}", explanation=(
            f"Pay {state['home']} {requested:,.2f} today; the daily 90-day projection keeps the balance at or above {state['home']} {floor:,.2f}."
        ))

    # Try smallest set of spending cuts that makes full payment safe.
    candidates = state["candidates"] if "full_payment" in allowed else []
    feasible_cut: tuple[dict[str, Any], ...] | None = None
    if candidates:
        options = []
        for count in range(1, len(candidates) + 1):
            for combo in itertools.combinations(candidates, count):
                payments = {start: requested}
                if _pay_is_safe(state, payments, combo):
                    savings = 0.0
                    for action in combo:
                        info = state["recurring_meta"][action["category"]]
                        if action["kind"] == "stop":
                            savings += info["amount"] * len(info["occurrences"])
                        else:
                            savings += sum(max(0.0, info["amount"] - action["minimum"]) for _ in info["occurrences"])
                    options.append((savings, count, combo))
            if options:
                break
        if options:
            feasible_cut = min(options, key=lambda item: (item[0], item[1], tuple(a["event_id"] for a in item[2])))[2]
    if feasible_cut:
        tokens = []
        for action in feasible_cut:
            if action["kind"] == "stop":
                tokens.append(f"stop:{action['event_id']}")
            else:
                tokens.append(f"reduce_to:{action['event_id']}:{fmt_num(action['minimum'])}")
        return result("affordable_with_plan", "full_payment", f"{start}:{fmt_num(requested)}", "|".join(tokens), (
            f"Apply {' and '.join(tokens)}, then pay {state['home']} {requested:,.2f}; the daily balance stays above {state['home']} {floor:,.2f}."
        ))

    if partial_on and "partial_payment" in allowed and 0 < safe_today < requested:
        inflow_days = sorted(
            day for day, recs in state["flows"].items()
            if day > start and day <= state["deadline"] and any(r[0] == "income" for r in recs)
        )
        for pay_day in inflow_days:
            remainder = round(requested - safe_today, 2)
            sched = {start: safe_today, pay_day: remainder}
            if _pay_is_safe(state, sched):
                plan = f"{start}:{fmt_num(safe_today)}|{pay_day}:{fmt_num(remainder)}"
                return result("affordable_with_plan", "partial_payment", plan, explanation=(
                    f"Pay {state['home']} {safe_today:,.2f} today and the remaining {state['home']} {remainder:,.2f} on {pay_day}; the daily balance stays above {state['home']} {floor:,.2f}."
                ))

    if "installments" in allowed:
        feasible = []
        for opt in state["input"].get("payment_options", []):
            if opt.get("payment_method") != "installments":
                continue
            count = int(opt["number_of_payments"])
            cap = int(profile["max_installment_months"]) if profile.get("max_installment_months") else 10**6
            if count > cap:
                continue
            first = parse_date(opt["first_payment_date"])
            freq = int(opt.get("payment_frequency_days") or 0)
            amt = float(opt["payment_amount"])
            dates = [first + timedelta(days=i * freq) for i in range(count)]
            if dates[-1] > state["deadline"] or dates[-1] > end:
                continue
            if any(day < start for day in dates):
                continue
            sched: dict[date, float] = defaultdict(float)
            for day in dates:
                sched[day] += amt
            if _pay_is_safe(state, dict(sched)):
                feasible.append((float(opt["total_payable_amount"]), dates[-1], count, amt, dates))
        if feasible:
            _total, _complete, count, amt, dates = min(feasible, key=lambda item: (item[0], item[1], item[2]))
            plan = "|".join(f"{day}:{fmt_num(amt)}" for day in dates)
            return result("affordable_with_plan", "installments", plan, explanation=(
                f"Use {count} installments of {state['home']} {amt:,.2f}; the daily balance stays above {state['home']} {floor:,.2f}."
            ))

    if earliest_full and "full_payment" in allowed:
        return result("affordable_later", "wait", f"{earliest_full}:{fmt_num(requested)}", explanation=(
            f"Wait until {earliest_full} to pay {state['home']} {requested:,.2f}; the daily balance then stays above {state['home']} {floor:,.2f}."
        ))

    return result("not_affordable", "not_recommended", "none", explanation=(
        f"None of the allowed payment choices keeps the daily balance above {state['home']} {floor:,.2f} over the 90-day window."
    ))


def solve_task(task_data: dict[str, Any]) -> dict[str, Any]:
    return simulate_decision(task_data)


def solve_all_tasks(tasks_dir: str) -> dict[str, dict[str, Any]]:
    import json
    from pathlib import Path

    results = {}
    for path in sorted(Path(tasks_dir).glob("task_*.json")):
        if path.name == "tasks_overview.json":
            continue
        task = json.loads(path.read_text(encoding="utf-8"))
        results[task["task_id"]] = solve_task(task)
    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    print(json.dumps(solve_all_tasks(str(root / "tasks")), indent=2))
