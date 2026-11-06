import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

def parse_date(date_str):
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()

def format_amount(amt):
    try:
        val = float(amt)
        if val.is_integer():
            return str(int(val))
        return f"{val:.2f}"
    except:
        return str(amt)

def load_exchange_rates(filepath):
    rates = {}
    if not Path(filepath).exists():
        return rates
    with open(filepath, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["rate_date"].strip(), row["from_currency"].strip(), row["to_currency"].strip())
            rates[key] = float(row["rate"])
    return rates

def load_profiles(filepath):
    profs = {}
    if not Path(filepath).exists():
        return profs
    with open(filepath, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["user_id"].strip()
            methods = [m.strip() for m in row.get("payment_methods_user_will_consider", "").split("|") if m.strip()]
            max_inst = int(row["max_installment_months"]) if row.get("max_installment_months", "").strip().isdigit() else None
            profs[uid] = {
                "home_currency": row["home_currency"].strip(),
                "available_balance": float(row["current_available_balance"]),
                "min_balance": float(row["minimum_balance_to_keep"]),
                "max_installment_months": max_inst,
                "considered_methods": set(methods),
                "willing_to_reduce": set([c.strip() for c in row.get("expense_categories_user_is_willing_to_reduce", "").split("|") if c.strip()]),
                "willing_to_stop": set([c.strip() for c in row.get("expense_categories_user_is_willing_to_stop", "").split("|") if c.strip()]),
                "protected_categories": set([c.strip() for c in row.get("expense_categories_to_protect", "").split("|") if c.strip()]),
            }
    return profs

def load_options(filepath):
    opts = {}
    if not Path(filepath).exists():
        return opts
    with open(filepath, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row["request_id"].strip()
            opts.setdefault(rid, []).append(row)
    return opts

def load_messages(filepath):
    msgs = {}
    if not Path(filepath).exists():
        return msgs
    with open(filepath, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("request_id", "").strip()
            if rid:
                msgs[rid] = row
    return msgs

def process_requests(dataset_dir="dataset", output_file="output.csv"):
    base = Path(dataset_dir)
    rates = load_exchange_rates(base / "exchange_rates.csv")
    profs = load_profiles(base / "financial_profiles.csv")
    opts_by_req = load_options(base / "request_payment_options.csv")
    msgs_by_req = load_messages(base / "messages.csv")

    req_path = base / "requests.csv"
    if not req_path.exists():
        print(f"Requests file not found at {req_path}")
        return

    results = []
    with open(req_path, mode="r", encoding="utf-8") as f:
        for req in csv.DictReader(f):
            rid = req["request_id"].strip()
            uid = req["user_id"].strip()
            rdate_str = req["request_date"].strip()
            rdate = parse_date(rdate_str)
            ramt = float(req["requested_amount"])
            dl_str = req["desired_completion_date"].strip()
            dl = parse_date(dl_str)
            partial_ok = req.get("allows_partial_payment", "false").lower() in ["true", "1", "yes"]

            prof = profs.get(uid, {
                "home_currency": "USD",
                "available_balance": 0.0,
                "min_balance": 0.0,
                "max_installment_months": None,
                "considered_methods": {"full_payment"},
                "willing_to_reduce": set(),
                "willing_to_stop": set(),
            })

            curr = prof["home_currency"]
            avail = prof["available_balance"]
            min_b = prof["min_balance"]
            headroom = max(0.0, avail - min_b)
            considered = prof["considered_methods"]
            max_inst = prof["max_installment_months"]

            safe = min(ramt, headroom)

            req_opts = opts_by_req.get(rid, [])

            # Find valid installment plans that fit within deadline and max months
            valid_plans = []
            for opt in req_opts:
                if opt["payment_method"].strip() == "installments":
                    npay = int(opt["number_of_payments"])
                    if max_inst is not None and npay > max_inst:
                        continue
                    first = parse_date(opt["first_payment_date"])
                    freq = int(opt["payment_frequency_days"]) if opt.get("payment_frequency_days", "").strip().isdigit() else 30
                    last = first + timedelta(days=freq * (npay - 1))
                    if last <= dl:
                        valid_plans.append((opt, first, last, npay, float(opt["payment_amount"]), float(opt["total_payable_amount"])))

            if safe >= ramt and "full_payment" in considered:
                status = "affordable_now"
                method = "full_payment"
                plan = f"{rdate_str}:{format_amount(ramt)}"
                earliest = rdate_str
                changes = "none"
                explanation = f"Pay {curr} {format_amount(ramt)} today. This leaves at least {curr} {format_amount(min_b)} available over the next 90 days."

            elif valid_plans and "installments" in considered:
                valid_plans.sort(key=lambda x: (x[5], x[1], x[3], x[0]["payment_option_id"]))
                best = valid_plans[0]
                opt, first, last, npay, pamt, total = best
                freq = int(opt["payment_frequency_days"]) if opt.get("payment_frequency_days", "").strip().isdigit() else 30

                parts = []
                for i in range(npay):
                    dt = first + timedelta(days=freq * i)
                    parts.append(f"{dt.strftime('%Y-%m-%d')}:{pamt:.2f}")
                plan = "|".join(parts)

                status = "affordable_with_plan"
                method = "installments"
                earliest = (first + timedelta(days=freq)).strftime("%Y-%m-%d")
                changes = "none"
                first_fmt = first.strftime("%d %B %Y").lstrip("0")
                explanation = f"Use {npay} installments of {curr} {pamt:,.2f}, starting {first_fmt}. This leaves at least {curr} {format_amount(min_b)} available."

            elif partial_ok and safe > 0 and "partial_payment" in considered:
                rem = ramt - safe
                payday = rdate.replace(day=15)
                if payday <= rdate:
                    if rdate.month == 12:
                        payday = rdate.replace(year=rdate.year+1, month=1, day=15)
                    else:
                        payday = rdate.replace(month=rdate.month+1, day=15)

                if payday <= dl:
                    earliest = payday.strftime("%Y-%m-%d")
                    status = "affordable_with_plan"
                    method = "partial_payment"
                    plan = f"{rdate_str}:{format_amount(safe)}|{earliest}:{format_amount(rem)}"
                    changes = "none"
                    explanation = f"Pay {curr} {format_amount(safe)} today and the remaining {curr} {format_amount(rem)} on {earliest}. This completes the full request and keeps the {curr} {format_amount(min_b)} minimum protected."
                else:
                    status = "not_affordable"
                    method = "not_recommended"
                    plan = "none"
                    earliest = ""
                    changes = "none"
                    explanation = f"Do not make this payment by {dl.strftime('%d %B %Y').lstrip('0')}. None of the available options keeps the {curr} {format_amount(min_b)} minimum protected."

            elif "full_payment" in considered:
                payday = rdate.replace(day=15)
                if payday <= rdate:
                    if rdate.month == 12:
                        payday = rdate.replace(year=rdate.year+1, month=1, day=15)
                    else:
                        payday = rdate.replace(month=rdate.month+1, day=15)

                if payday <= dl:
                    status = "affordable_later"
                    method = "wait"
                    earliest = payday.strftime("%Y-%m-%d")
                    plan = f"{earliest}:{format_amount(ramt)}"
                    changes = "none"
                    explanation = f"Pay {curr} {format_amount(ramt)} in full on {earliest}. Paying earlier would take the balance below the {curr} {format_amount(min_b)} minimum."
                else:
                    status = "not_affordable"
                    method = "not_recommended"
                    plan = "none"
                    earliest = ""
                    changes = "none"
                    explanation = f"Do not make this payment by {dl.strftime('%d %B %Y').lstrip('0')}. None of the available options keeps the {curr} {format_amount(min_b)} minimum protected."

            else:
                status = "not_affordable"
                method = "not_recommended"
                plan = "none"
                earliest = ""
                changes = "none"
                explanation = f"Do not make this payment by {dl.strftime('%d %B %Y').lstrip('0')}. None of the available options keeps the {curr} {format_amount(min_b)} minimum protected."

            results.append({
                "request_id": rid,
                "amount_safe_to_pay": format_amount(safe),
                "affordability_status": status,
                "recommended_payment_method": method,
                "payment_plan": plan,
                "earliest_date_for_full_payment": earliest,
                "spending_changes_needed": changes,
                "decision_explanation": explanation,
            })

    fieldnames = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Successfully processed {len(results)} requests and wrote to {output_file}")

if __name__ == "__main__":
    process_requests(dataset_dir="dataset", output_file="output.csv")
