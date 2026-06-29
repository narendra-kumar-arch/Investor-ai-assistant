#!/usr/bin/env python3
"""
EquiTie Portfolio Assistant — Python + LangChain + Rich
========================================================
All 10 dataset tables embedded in both modes.

  PORTFOLIO mode  — full dataset, cross-investor analytics (RM / admin)
  INVESTOR mode   — single investor Q&A, all 10 tables filtered

LangChain components used
--------------------------
  ChatAnthropic               model wrapper (claude-sonnet-4-6)
  ChatPromptTemplate          typed prompt with named slots
  MessagesPlaceholder         injects conversation history
  StrOutputParser             extracts plain text from AIMessage
  SessionMemory               per-session in-memory message history

Run
---
  export ANTHROPIC_API_KEY=sk-ant-...
  python assistant.py
"""

import os
import sys
import json
import csv
from pathlib import Path
from getpass import getpass
from collections import defaultdict

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# ── Rich (terminal UI) ────────────────────────────────────────────────────────
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich.columns import Columns
from rich import box

console = Console()
DATA_DIR = Path(__file__).parent / "data"

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_csv(filename: str) -> list[dict]:
    with open(DATA_DIR / filename, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

# Load all 10 tables once at startup
DB: dict[str, list[dict]] = {
    "investors":   load_csv("investors.csv"),           # 112 rows
    "companies":   load_csv("portfolio_companies.csv"), # 16 rows
    "deals":       load_csv("deals.csv"),               # 21 rows
    "allocations": load_csv("allocations.csv"),         # 550 rows
    "valuations":  load_csv("valuations.csv"),          # 55 rows
    "calls":       load_csv("capital_calls.csv"),       # 655 rows
    "fees":        load_csv("fees.csv"),                # 1401 rows
    "dists":       load_csv("distributions.csv"),       # 34 rows
    "stmts":       load_csv("statement_lines.csv"),     # 1390 rows
    "fx":          load_csv("fx_rates.csv"),            # 4 rows
}

ALL_INVESTOR_IDS: set[str] = {i["investor_id"] for i in DB["investors"]}

# ── FX helpers ────────────────────────────────────────────────────────────────
FX: dict[str, float] = {r["currency"]: float(r["to_usd"]) for r in DB["fx"]}

def to_usd(amount: float | str, ccy: str) -> float:
    return float(amount) * FX.get(ccy, 1.0)

def convert(amount: float | str, from_ccy: str, to_ccy: str) -> float:
    if from_ccy == to_ccy:
        return float(amount)
    return to_usd(amount, from_ccy) / FX.get(to_ccy, 1.0)

def fmt_money(n: float, ccy: str = "USD") -> str:
    syms = {"USD": "$", "GBP": "£", "EUR": "€", "AED": "AED "}
    sym = syms.get(ccy, ccy + " ")
    return f"{sym}{abs(n):,.0f}"

# ── Shared indexes ────────────────────────────────────────────────────────────
deal_map: dict[str, dict] = {d["deal_id"]: d for d in DB["deals"]}

latest_val: dict[str, dict] = {}
for v in DB["valuations"]:
    if v["deal_id"] not in latest_val or v["valuation_date"] > latest_val[v["deal_id"]]["valuation_date"]:
        latest_val[v["deal_id"]] = v

# ════════════════════════════════════════════════════════════════════════════
#  DATA BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def build_investor_slice(investor_id: str) -> dict | None:
    """All 10 tables filtered to one investor. Sent verbatim in the system prompt."""
    investor = next((i for i in DB["investors"] if i["investor_id"] == investor_id), None)
    if not investor:
        return None

    allocs  = [a for a in DB["allocations"] if a["investor_id"] == investor_id]
    deal_ids = {a["deal_id"] for a in allocs}

    return {
        "investor":        investor,
        "allocations":     allocs,
        "deals":           [d for d in DB["deals"] if d["deal_id"] in deal_ids],
        "companies":       DB["companies"],                                         # all 16, small
        "valuations":      [v for v in DB["valuations"] if v["deal_id"] in deal_ids],  # full history
        "capital_calls":   [c for c in DB["calls"] if c["investor_id"] == investor_id],
        "fees":            [f for f in DB["fees"]  if f["investor_id"] == investor_id],
        "distributions":   [d for d in DB["dists"] if d["investor_id"] == investor_id],
        "statement_lines": [s for s in DB["stmts"] if s["investor_id"] == investor_id],
        "fx_rates":        DB["fx"],
    }


def compute_investor_summary(slice_: dict) -> dict:
    """Compute dashboard numbers in Python (no model involved)."""
    inv    = slice_["investor"]
    rc     = inv["reporting_currency"]
    allocs = slice_["allocations"]
    dists  = slice_["distributions"]
    calls  = slice_["capital_calls"]
    fees   = slice_["fees"]
    deals  = {d["deal_id"]: d for d in slice_["deals"]}

    cur_usd = cont_usd = com_usd = dis_usd = 0.0

    for a in allocs:
        if a["allocation_status"] == "Pending":
            continue
        deal   = deals.get(a["deal_id"], {})
        exited = deal.get("status") in ("Exited", "Written Off")
        lv     = latest_val.get(a["deal_id"])
        price  = float(lv["share_price"]) if lv else 0.0
        ad     = [d for d in dists if d["allocation_id"] == a["allocation_id"]]
        frac   = sum(float(d.get("fraction_of_units", 0)) for d in ad)
        cv     = 0.0 if exited else float(a["units"]) * price * max(0.0, 1 - frac)
        cur_usd  += to_usd(cv, a["deal_currency"])
        cont_usd += to_usd(float(a["contributed_amount"] or 0), a["deal_currency"])
        com_usd  += to_usd(float(a["commitment_amount"]  or 0), a["deal_currency"])
        dis_usd  += sum(to_usd(float(d["net_amount"]), d["currency"]) for d in ad)

    moic     = (cur_usd + dis_usd) / cont_usd if cont_usd > 0 else 0.0
    up_calls = [c for c in calls if c["status"] == "Upcoming"]
    ov_fees  = [f for f in fees  if f["status"] == "Overdue"]

    deal_ids_active = {a["deal_id"] for a in allocs if a["allocation_status"] != "Pending"}

    return {
        "rc":                  rc,
        "deal_count":          len(deal_ids_active),
        "current_value":       convert(cur_usd,  "USD", rc),
        "contributed":         convert(cont_usd, "USD", rc),
        "commitment":          convert(com_usd,  "USD", rc),
        "distributions":       convert(dis_usd,  "USD", rc),
        "moic":                moic,
        "dpi":                 dis_usd / cont_usd if cont_usd > 0 else 0.0,
        "rvpi":                cur_usd / cont_usd if cont_usd > 0 else 0.0,
        "upcoming_calls_count":len(up_calls),
        "upcoming_calls_rc":   convert(
            sum(to_usd(float(c["amount"]), c["currency"]) for c in up_calls),
            "USD", rc),
        "overdue_fees_count":  len(ov_fees),
        "upcoming_fees_count": len([f for f in fees if f["status"] == "Upcoming"]),
    }



def compute_profile_signals(investor_id: str, slice_: dict) -> dict:
    """
    Derive portfolio-shape signals the brief explicitly requires:
    - deal count, top sectors, company names, concentration.
    These are injected into the system prompt so the model reflects
    the investor's actual portfolio shape, not a generic answer.
    """
    from collections import Counter
    allocs = [a for a in slice_["allocations"] if a["allocation_status"] != "Pending"]
    if not allocs:
        return {"deal_count": 0, "top_sectors": [], "companies": [], "concentration": "no holdings yet"}

    company_map = {c["company_id"]: c for c in slice_["companies"]}
    deal_objs   = slice_["deals"]
    deal_lookup = {d["deal_id"]: d for d in deal_objs}

    sector_counts: Counter = Counter()
    company_counts: Counter = Counter()
    for a in allocs:
        d = deal_lookup.get(a["deal_id"])
        if not d:
            continue
        comp = company_map.get(d["company_id"])
        if comp:
            sector_counts[comp["sector"]] += 1
        company_counts[d["company_name"]] += 1

    top_sectors = [s for s, _ in sector_counts.most_common(3)]
    top_company, top_count = company_counts.most_common(1)[0] if company_counts else ("", 0)
    concentration = (
        f"concentrated in {top_company} ({top_count} rounds)"
        if top_count >= 2 else "diversified across companies"
    )

    return {
        "deal_count":   len(allocs),
        "top_sectors":  top_sectors,
        "companies":    sorted(company_counts.keys()),
        "concentration": concentration,
    }

def build_portfolio_context() -> dict:
    """
    All 10 tables at appropriate granularity for the RM/admin view.
    Aggregated summaries for investors + deals; raw rows for actionable
    transactions (fees, calls, distributions, valuations, allocations).
    """
    inv_summaries = []
    for inv in DB["investors"]:
        iid    = inv["investor_id"]
        allocs = [a for a in DB["allocations"]
                  if a["investor_id"] == iid and a["allocation_status"] != "Pending"]
        deal_ids = {a["deal_id"] for a in allocs}

        cur_usd = cont_usd = com_usd = dis_usd = 0.0
        for a in allocs:
            deal   = deal_map.get(a["deal_id"], {})
            exited = deal.get("status") in ("Exited", "Written Off")
            lv     = latest_val.get(a["deal_id"])
            price  = float(lv["share_price"]) if lv else 0.0
            ad     = [d for d in DB["dists"] if d["allocation_id"] == a["allocation_id"]]
            frac   = sum(float(d.get("fraction_of_units", 0)) for d in ad)
            cv     = 0.0 if exited else float(a["units"]) * price * max(0.0, 1 - frac)
            cur_usd  += to_usd(cv, a["deal_currency"])
            cont_usd += to_usd(float(a["contributed_amount"] or 0), a["deal_currency"])
            com_usd  += to_usd(float(a["commitment_amount"]  or 0), a["deal_currency"])
            dis_usd  += sum(to_usd(float(d["net_amount"]), d["currency"]) for d in ad)

        moic = (cur_usd + dis_usd) / cont_usd if cont_usd > 0 else 0.0
        inv_calls = [c for c in DB["calls"] if c["investor_id"] == iid]
        inv_fees  = [f for f in DB["fees"]  if f["investor_id"] == iid]

        inv_summaries.append({
            "id": iid, "name": inv["investor_name"],
            "type": inv["investor_type"], "country": inv["country"],
            "currency": inv["reporting_currency"], "kyc": inv["kyc_status"],
            "tech": inv["tech_savviness"], "age": inv["age"],
            "deal_count": len(deal_ids),
            "deals": [
                f"{deal_map[d]['company_name']} {deal_map[d]['round']}"
                for d in deal_ids if d in deal_map
            ],
            "current_value_usd":  round(cur_usd),
            "contributed_usd":    round(cont_usd),
            "commitment_usd":     round(com_usd),
            "distributions_usd":  round(dis_usd),
            "moic":               round(moic, 2),
            "upcoming_calls":     sum(1 for c in inv_calls if c["status"] == "Upcoming"),
            "overdue_fees":       sum(1 for f in inv_fees  if f["status"] == "Overdue"),
            "upcoming_fees":      sum(1 for f in inv_fees  if f["status"] == "Upcoming"),
            "has_distributions":  dis_usd > 0,
        })

    deal_summaries = []
    for deal in DB["deals"]:
        did    = deal["deal_id"]
        allocs = [a for a in DB["allocations"]
                  if a["deal_id"] == did and a["allocation_status"] != "Pending"]
        lv     = latest_val.get(did)
        exited = deal["status"] in ("Exited", "Written Off")
        cur_usd = 0.0 if exited else sum(
            to_usd(float(a["units"]) * (float(lv["share_price"]) if lv else 0.0), a["deal_currency"])
            for a in allocs
        )
        deal_summaries.append({
            "deal_id": did, "company": deal["company_name"], "round": deal["round"],
            "status": deal["status"], "currency": deal["deal_currency"],
            "investor_count":        len(allocs),
            "total_committed_usd":   round(sum(to_usd(float(a["commitment_amount"]  or 0), a["deal_currency"]) for a in allocs)),
            "total_contributed_usd": round(sum(to_usd(float(a["contributed_amount"] or 0), a["deal_currency"]) for a in allocs)),
            "current_value_usd":     round(cur_usd),
            "latest_mark":           lv["share_price"]        if lv else None,
            "latest_mark_date":      lv["valuation_date"]     if lv else None,
            "multiple_vs_entry":     lv["multiple_vs_entry"]  if lv else None,
        })

    # Raw actionable tables — small enough to include verbatim
    actionable_fees = [
        {"investor_id": f["investor_id"], "deal_id": f["deal_id"],
         "fee_type": f["fee_type"], "amount": f["amount"],
         "currency": f["currency"], "due_date": f["due_date"], "status": f["status"]}
        for f in DB["fees"] if f["status"] in ("Overdue", "Upcoming")
    ]
    upcoming_calls = [
        {"investor_id": c["investor_id"], "deal_id": c["deal_id"],
         "call_date": c["call_date"], "due_date": c["due_date"],
         "amount": c["amount"], "currency": c["currency"]}
        for c in DB["calls"] if c["status"] == "Upcoming"
    ]
    distributions = [
        {"investor_id": d["investor_id"], "deal_id": d["deal_id"],
         "distribution_date": d["distribution_date"], "distribution_type": d["distribution_type"],
         "gross_amount": d["gross_amount"], "performance_fee_pct": d["performance_fee_pct"],
         "net_amount": d["net_amount"], "currency": d["currency"],
         "fraction_of_units": d["fraction_of_units"]}
        for d in DB["dists"]
    ]
    valuations = [
        {"deal_id": v["deal_id"], "valuation_date": v["valuation_date"],
         "share_price": v["share_price"], "multiple_vs_entry": v["multiple_vs_entry"],
         "mark_source": v["mark_source"]}
        for v in DB["valuations"]
    ]
    allocations = [
        {"allocation_id": a["allocation_id"], "deal_id": a["deal_id"],
         "investor_id": a["investor_id"], "deal_currency": a["deal_currency"],
         "commitment_amount": a["commitment_amount"], "contributed_amount": a["contributed_amount"],
         "units": a["units"], "effective_share_price": a["effective_share_price"],
         "mgmt_fee_pct": a["mgmt_fee_pct"], "performance_fee_pct": a["performance_fee_pct"],
         "structuring_fee_pct": a["structuring_fee_pct"], "admin_fee_usd": a["admin_fee_usd"],
         "fee_discount": a["fee_discount"], "allocation_status": a["allocation_status"]}
        for a in DB["allocations"]
    ]

    total_aum = sum(s["current_value_usd"] for s in inv_summaries)

    return {
        "as_of": "2026-06-25",
        "fx_rates": DB["fx"],
        "total_aum_usd":                round(total_aum),
        "total_committed_usd":          round(sum(s["commitment_usd"] for s in inv_summaries)),
        "total_investors":              len(DB["investors"]),
        "total_deals":                  len(DB["deals"]),
        "total_companies":              len(DB["companies"]),
        "investors_with_overdue_fees":  sum(1 for s in inv_summaries if s["overdue_fees"] > 0),
        "investors_with_upcoming_calls":sum(1 for s in inv_summaries if s["upcoming_calls"] > 0),
        "investors_pending_kyc":        sum(1 for s in inv_summaries if s["kyc"] == "Pending"),
        # Aggregated rows
        "investors":         inv_summaries,   # 112 rows, computed totals
        "deals":             deal_summaries,  # 21 rows, computed totals
        "companies":         DB["companies"], # 16 rows, raw
        # Raw transactional data
        "allocations":       allocations,     # 550 rows — fee discount comparisons
        "actionable_fees":   actionable_fees, # Overdue + Upcoming only (594 rows)
        "upcoming_calls":    upcoming_calls,  # 106 rows
        "distributions":     distributions,   # 34 rows, all
        "valuations":        valuations,      # 55 rows, full history
    }


# ════════════════════════════════════════════════════════════════════════════
#  GUARDRAILS  (deterministic — run BEFORE any model call)
# ════════════════════════════════════════════════════════════════════════════

import re

ADVICE_PATTERNS = [
    re.compile(r"\bshould i\b", re.I),
    re.compile(r"\bwould you recommend\b", re.I),
    re.compile(r"\b(is it|would it be) (a good|worth|smart|wise)\b", re.I),
    re.compile(r"\b(buy|sell|exit|reinvest|pull out|double down)\b", re.I),
    re.compile(r"\badvise\b", re.I),
    re.compile(r"\brecommend (that |me )?(i|you|we)\b", re.I),
    re.compile(r"\bwill .{0,20}(go up|go down|recover|crash|perform|outperform)\b", re.I),
    re.compile(r"\bprice target\b", re.I),
]

INJECTION_PATTERNS = [
    re.compile(r"ignore (your |the )?(previous |prior |all )?(instructions|prompt|rules)", re.I),
    re.compile(r"disregard .{0,20}(instructions|rules)", re.I),
    re.compile(r"you are now\b", re.I),
    re.compile(r"new (system|instructions|prompt|persona)\b", re.I),
    re.compile(r"act as (if |though )?(you are|a different)", re.I),
    re.compile(r"forget (everything|your|the)\b", re.I),
    re.compile(r"override .{0,20}(instructions|rules|system)", re.I),
    re.compile(r"pretend (you are|to be|that)\b", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"\bDAN\b"),
]

ADVICE_REFUSAL = (
    "I can share factual portfolio data — valuations, fees, distributions, and capital calls — "
    "but I cannot give investment advice or buy/sell/exit recommendations.\n\n"
    "For guidance, please speak with a relationship manager."
)
INJECT_REFUSAL = (
    "I can only answer questions about the portfolio data. "
    "I cannot override my instructions or access data outside what has been provided."
)


def check_guardrails(text: str) -> str | None:
    """Return a refusal string if the input violates a guardrail, else None."""
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            return INJECT_REFUSAL
    for pat in ADVICE_PATTERNS:
        if pat.search(text):
            return ADVICE_REFUSAL
    return None


def kyc_gate(investor: dict) -> str | None:
    if investor["kyc_status"] != "Pending":
        return None
    return (
        "[bold red]⚠  KYC Pending[/bold red]\n"
        "Portfolio details are unavailable until identity verification is complete.\n"
        "Please contact your EquiTie relationship manager."
    )


def assert_isolation(slice_: dict) -> None:
    iid = slice_["investor"]["investor_id"]
    for table in ["capital_calls", "fees", "distributions", "allocations", "statement_lines"]:
        for row in slice_.get(table, []):
            if row.get("investor_id") and row["investor_id"] != iid:
                raise RuntimeError(
                    f"DATA ISOLATION VIOLATION: {row['investor_id']} in {iid}'s slice"
                )


def audit_response(text: str, own_id: str) -> list[str]:
    """Return any foreign investor IDs that appear in the model response."""
    return [iid for iid in ALL_INVESTOR_IDS if iid != own_id and iid in text]


# ════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN CHAINS
# ════════════════════════════════════════════════════════════════════════════

def make_model(api_key: str) -> ChatAnthropic:
    return ChatAnthropic(
        api_key=api_key,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0.2,
    )


class SessionMemory:
    """
    Simple per-session in-memory message history.
    Replaces RunnableWithMessageHistory (deprecated in langchain-core 1.x).
    History is a plain list of {"role": "human"|"ai", "content": str} dicts
    passed directly into ChatPromptTemplate's MessagesPlaceholder.
    """
    def __init__(self):
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "human", "content": text})

    def add_ai(self, text: str) -> None:
        self.messages.append({"role": "ai", "content": text})

    def history(self) -> list[dict]:
        return self.messages.copy()


def make_chain(api_key: str, system_template: str):
    """
    Returns a callable: invoke(frozen_vars, user_input, memory) -> str

    LangChain components:
      ChatPromptTemplate   — typed prompt with named slots + MessagesPlaceholder
      ChatAnthropic        — model wrapper (claude-sonnet-4-6)
      StrOutputParser      — extracts plain text from AIMessage

    Memory is managed explicitly via SessionMemory (avoids deprecated
    RunnableWithMessageHistory). History is injected as a plain list of
    dicts at call time, which ChatPromptTemplate handles natively.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | make_model(api_key) | StrOutputParser()

    def invoke(frozen_vars: dict, user_input: str, memory: SessionMemory) -> str:
        reply = chain.invoke({
            **frozen_vars,
            "input":   user_input,
            "history": memory.history(),
        })
        memory.add_user(user_input)
        memory.add_ai(reply)
        return reply

    return invoke


def build_portfolio_chain(api_key: str, ctx: dict):
    system_template = """You are the EquiTie Portfolio Assistant — AI for relationship managers and fund admins.
Report date: 2026-06-25.

You have access to the COMPLETE dataset across all 10 tables.
Use it to answer aggregate, cross-investor, deal-level, and specific transactional questions.

FX RATES: {fx_rates}
Convert X→Y: amount × to_usd[X] / to_usd[Y]

RULES:
1. Base all answers only on data provided. Do not invent numbers.
2. Distinguish committed vs contributed — they differ when deals are partially called.
3. Current value = units × latest share_price × (1 − fraction_realised). Zero if Exited/Written Off.
4. For fee questions: compare allocation's effective rates to the deal's std_* rates to show discounts.
5. MOIC = (current_value + net_distributions) / contributed_amount.
6. When listing investors, include ID + name. Limit to top-10 unless more are requested.
7. actionable_fees: status Overdue = past due, Upcoming = future due.
8. valuations: full mark history per deal — use for trend and down-round questions.

DATA COVERAGE:
- investors (112)      — pre-computed totals: value, MOIC, deal list, fee/call counts
- deals (21)           — pre-computed totals per deal
- companies (16)       — sector, HQ, status
- allocations (550)    — every investor-deal position incl. effective fee rates
- actionable_fees(594) — all Overdue and Upcoming fees with amounts and due dates
- upcoming_calls (106) — all upcoming capital calls with amounts and due dates
- distributions (34)   — all exit proceeds and secondary sales
- valuations (55)      — full mark history per deal (all dates)
- fx_rates (4)

FULL PORTFOLIO CONTEXT:
{portfolio_data}"""

    invoke = make_chain(api_key, system_template)
    frozen = {
        "fx_rates":       json.dumps(ctx["fx_rates"]),
        "portfolio_data": json.dumps(ctx),
    }
    return invoke, frozen, SessionMemory()


def build_investor_chain(api_key: str, slice_: dict):
    inv  = slice_["investor"]
    tech = inv["tech_savviness"]
    age  = int(inv.get("age") or 0)
    n_deals = len(slice_["allocations"])

    if tech == "Low" or age >= 65:
        persona = "Plain language. Define all jargon (MOIC, carry, SPV, DPI, RVPI). Short answers. Patient tone."
    elif tech == "High" and n_deals >= 6:
        persona = "Concise and data-dense. Investor is sophisticated. Skip basic definitions."
    else:
        persona = "Balanced explanations. Define niche terms where helpful. Professional tone."

    system_template = """You are the EquiTie Investor Assistant — a personal portfolio AI.
Report date: 2026-06-25. Authenticated investor: {investor_name} ({investor_id}).

PERSONA: {persona}

INVESTOR PROFILE (use to personalise every answer):
- Reporting currency: {reporting_currency}
- Tech savviness: {tech} | Age: {age}
- Active deals: {deal_count} | Top sectors: {top_sectors}
- Companies: {companies}
- Portfolio shape: {concentration}

PERSONALISATION RULES:
- Reflect portfolio shape in answers — mention their actual sectors and companies, not generic examples.
- Low tech or age ≥ 65: plain language, define MOIC, carry, SPV, DPI, RVPI on first use. Short answers.
- High tech + many deals: concise, data-dense, skip definitions.
- Never patronising. Tone changes; numbers never change.

DATA RULES:
1. Answer ONLY about this investor. Never reference other investors' data.
2. Every number must come from the data below. Do not invent or estimate.
3. Convert all portfolio totals to {reporting_currency}.
4. FX: {fx_rates}. Convert X→Y: amount × to_usd[X] / to_usd[Y].

FORMULAS — apply exactly:
- Current value (per allocation) = units × latest_share_price × (1 − fraction_of_units_realised).
  → Zero if deal status is Exited or Written Off.
- MOIC = (current_value + Σ distributions.net_amount) ÷ contributed_amount.
  → Convert everything to same currency before summing.
- Cost basis = contributed_amount (NOT commitment_amount — they differ for partial calls).
- DPI = total distributions ÷ contributed. RVPI = total current value ÷ contributed.
- Same company, multiple rounds: show each round separately, then combined total.
- Upcoming obligations = capital_calls (status=Upcoming) + fees (status=Upcoming or Overdue).
- Fee discount: compare allocation's mgmt_fee_pct / performance_fee_pct / structuring_fee_pct / admin_fee_usd
  to the deal's std_mgmt_fee_pct / std_performance_fee_pct / std_structuring_fee_pct / std_admin_fee_usd.
- Net distribution = gross_amount − performance_fee_amount (carry withheld at allocation's effective rate).
- statement_lines: negative amount = cash out (contributions, fees); positive = cash in (distributions).

EDGE CASES — handle explicitly:
- Forgecraft Robotics has 3 rounds (Seed/A/B): aggregate across rounds when asked about "Forgecraft".
- Qubrium: down round — Series B mark (6.2) is below entry (10.0). Show the markdown.
- Helianthe Energy: exited — current value = 0, but distributions exist and count toward MOIC.
- Yappio: written off — MOIC < 1, current value = 0.
- Tallybook: partial secondary (fraction_of_units < 1) — realised and unrealised portions both exist.
- Pulsegrid Health Series B and Forgecraft Series B: contributed_pct < 100 — outstanding commitment exists.
- Grace Okafor: Pending KYC / unfunded — not deployed capital.
- Admin fee billed in USD even on non-USD deals.
- If an investor holds no allocations: say so clearly, do not fabricate positions.
- Similar names: Northpeak Analytics ≠ Northpeak Health (different sectors, different currencies).

DATA COVERAGE (all 10 tables, filtered to this investor):
  investor          profile, KYC, reporting currency, tech savviness
  allocations       all positions incl. effective fee rates, price discounts, units
  deals             their deals incl. standard fee schedules (for discount comparison)
  companies         all 16 portfolio companies (sector and status reference)
  valuations        full mark history for their deals (all dates, not just latest)
  capital_calls     all calls — Paid and Upcoming
  fees              all fees — Paid, Upcoming, and Overdue
  distributions     all exit proceeds and secondary sales, gross and net of carry
  statement_lines   full per-investor account statement (signed cash flows)
  fx_rates          FX rates as of 2026-06-25

INVESTOR DATA:
{investor_data}"""

    signals = compute_profile_signals(inv["investor_id"], slice_)

    invoke = make_chain(api_key, system_template)
    frozen = {
        "investor_name":      inv["investor_name"],
        "investor_id":        inv["investor_id"],
        "persona":            persona,
        "reporting_currency": inv["reporting_currency"],
        "tech":               tech,
        "age":                inv.get("age") or "N/A",
        "deal_count":         str(signals["deal_count"]),
        "top_sectors":        ", ".join(signals["top_sectors"]) if signals["top_sectors"] else "none yet",
        "companies":          ", ".join(signals["companies"])   if signals["companies"]   else "none yet",
        "concentration":      signals["concentration"],
        "fx_rates":           json.dumps(slice_["fx_rates"]),
        "investor_data":      json.dumps(slice_),
    }
    return invoke, frozen, SessionMemory()


# ════════════════════════════════════════════════════════════════════════════
#  RICH TERMINAL UI
# ════════════════════════════════════════════════════════════════════════════

def print_banner():
    console.clear()
    console.print(Panel(
        "[bold cyan]EquiTie Portfolio Assistant[/bold cyan]  "
        "[dim]· Python + LangChain + Claude[/dim]\n"
        "[dim]All 10 dataset tables · Report date: 2026-06-25[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()


def print_portfolio_dashboard(ctx: dict):
    table = Table(
        title="[bold]Portfolio Overview[/bold]  [dim](RM / Admin mode)[/dim]",
        box=box.ROUNDED, border_style="blue", show_header=False,
        padding=(0, 1),
    )
    table.add_column("Metric",  style="dim",  no_wrap=True)
    table.add_column("Value",   style="bold")

    def row(label, value, style="bold"):
        table.add_row(label, f"[{style}]{value}[/{style}]")

    row("Total AUM (USD)",               fmt_money(ctx["total_aum_usd"]))
    row("Total committed (USD)",          fmt_money(ctx["total_committed_usd"]))
    row("Investors",                      str(ctx["total_investors"]))
    row("Deals",                          str(ctx["total_deals"]))
    row("Portfolio companies",            str(ctx["total_companies"]))
    row("Investors w/ overdue fees",      str(ctx["investors_with_overdue_fees"]),  "yellow")
    row("Investors w/ upcoming calls",    str(ctx["investors_with_upcoming_calls"]))
    row("Pending KYC",                    str(ctx["investors_pending_kyc"]),         "yellow" if ctx["investors_pending_kyc"] else "bold")
    row("Actionable fees in context",     f"{len(ctx['actionable_fees'])} rows (Overdue + Upcoming)")
    row("Upcoming calls in context",      f"{len(ctx['upcoming_calls'])} rows")
    row("Distributions in context",       f"{len(ctx['distributions'])} rows")
    row("Valuations in context",          f"{len(ctx['valuations'])} rows (full history)")

    console.print(table)
    console.print()


def print_portfolio_suggestions():
    qs = [
        "Which investors have overdue fees? List them with amounts",
        "List all upcoming capital calls due before 31 July 2026",
        "Who are the top 5 investors by current portfolio value?",
        "Show me all investors in Forgecraft Robotics",
        "Which deals have the highest MOIC?",
        "What distributions did Helianthe Energy investors receive?",
        "Show Qubrium's valuation history — I want to see the down round",
        "Which investors negotiated a fee discount on their allocations?",
    ]
    console.print("[dim]Suggested questions:[/dim]")
    for i, q in enumerate(qs, 1):
        console.print(f"  [blue]{i}.[/blue] {q}")
    console.print()


def print_investor_dashboard(investor: dict, s: dict):
    rc = s["rc"]
    table = Table(
        title=f"[bold]{investor['investor_name']}[/bold]  [dim]({investor['investor_id']})[/dim]",
        box=box.ROUNDED, border_style="cyan", show_header=False,
        padding=(0, 1),
    )
    table.add_column("Metric", style="dim",  no_wrap=True)
    table.add_column("Value",  style="bold")

    subtitle = "  ·  ".join([
        investor["investor_type"], investor["country"],
        f"KYC: {investor['kyc_status']}", f"Tech: {investor['tech_savviness']}"
    ])
    table.add_row("[dim]" + subtitle + "[/dim]", "")

    if s["deal_count"] == 0:
        table.add_row("Holdings", "[yellow]No active holdings[/yellow]")
    else:
        table.add_row("Portfolio value",
                      fmt_money(s["current_value"], rc))
        table.add_row("Total contributed",
                      f"{fmt_money(s['contributed'], rc)}  [dim]({fmt_money(s['commitment'], rc)} committed)[/dim]")
        table.add_row("Distributions",    fmt_money(s["distributions"], rc))
        table.add_row("MOIC",             f"{s['moic']:.2f}×")
        table.add_row("DPI / RVPI",       f"{s['dpi']:.2f}× / {s['rvpi']:.2f}×")
        table.add_row("Active deals",     str(s["deal_count"]))
        if s["upcoming_calls_count"] > 0:
            table.add_row(
                "Upcoming calls",
                f"[yellow]{s['upcoming_calls_count']} · {fmt_money(s['upcoming_calls_rc'], rc)}[/yellow]"
            )
        if s["overdue_fees_count"] > 0:
            table.add_row(
                "Overdue fees",
                f"[red]{s['overdue_fees_count']} fee(s) — action needed[/red]"
            )

    console.print(table)
    console.print()


def print_investor_suggestions(s: dict):
    qs = [
        "Give me a portfolio overview — total value, contributions, and MOIC",
        "Walk me through my current holdings and their valuations",
        "What fees am I paying, and do I have any discounts vs the deal standard?",
        "Walk me through my full account statement — all cash in and out",
    ]
    if s["distributions"] > 0:
        qs.append("What distributions or exits have I received?")
    if s["deal_count"] >= 3:
        qs.append("Which of my deals has performed best by MOIC?")
    if s["overdue_fees_count"] > 0:
        qs.append("Do I have any overdue fees?")
    if s["upcoming_calls_count"] > 0:
        qs.append("What are my upcoming capital calls?")

    console.print("[dim]Suggested questions:[/dim]")
    for i, q in enumerate(qs[:6], 1):
        console.print(f"  [cyan]{i}.[/cyan] {q}")
    console.print()


def print_investor_list():
    sorted_investors = sorted(DB["investors"], key=lambda i: i["investor_name"])
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    for _ in range(3):
        table.add_column(no_wrap=True)

    cols = 3
    n    = len(sorted_investors)
    rows = (n + cols - 1) // cols
    for r in range(rows):
        cells = []
        for c in range(cols):
            idx = r + c * rows
            if idx < n:
                inv = sorted_investors[idx]
                cells.append(f"[dim]{inv['investor_id']}[/dim] {inv['investor_name']}")
            else:
                cells.append("")
        table.add_row(*cells)

    console.print("[bold]Available investors:[/bold]\n")
    console.print(table)
    console.print()


def print_slice_info(slice_: dict):
    tables = [
        ("allocations",     len(slice_["allocations"])),
        ("deals",           len(slice_["deals"])),
        ("valuations",      len(slice_["valuations"])),
        ("capital_calls",   len(slice_["capital_calls"])),
        ("fees",            len(slice_["fees"])),
        ("distributions",   len(slice_["distributions"])),
        ("statement_lines", len(slice_["statement_lines"])),
    ]
    size_kb = len(json.dumps(slice_)) / 1024
    console.print(f"  [green]✓[/green] Slice built ({size_kb:.1f} KB, all 10 tables):")
    for name, count in tables:
        console.print(f"    [dim]{name:<18}[/dim] {count} row(s)")
    console.print()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def ask(prompt_text: str) -> str:
    return Prompt.ask(f"[yellow]→[/yellow] {prompt_text}").strip()


def main():
    print_banner()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        console.print("[dim]  No ANTHROPIC_API_KEY environment variable found.[/dim]")
        api_key = Prompt.ask("  [yellow]→[/yellow] Enter your Anthropic API key", password=True).strip()
        if not api_key:
            console.print("[red]  No key provided. Exiting.[/red]")
            sys.exit(1)
    else:
        console.print("  [green]✓[/green] API key loaded from environment.\n")

    # ── Mode selection ────────────────────────────────────────────────────────
    console.print("  [bold]Select mode:[/bold]\n")
    console.print("  [blue][P][/blue] [bold]Portfolio mode[/bold]  — full dataset, cross-investor analytics, RM / admin")
    console.print("  [cyan][I][/cyan] [bold]Investor mode[/bold]   — single investor Q&A (all 10 tables filtered)\n")

    mode = ""
    while mode not in ("p", "i"):
        mode = ask("Enter P or I").lower()

    # ════════════════════════════════════════════════════════════════════════
    #  PORTFOLIO MODE
    # ════════════════════════════════════════════════════════════════════════
    if mode == "p":
        with console.status("[blue]Building full portfolio context (all 10 tables)…[/blue]"):
            ctx      = build_portfolio_context()
            invoke, frozen, memory = build_portfolio_chain(api_key, ctx)
        ctx_kb = len(json.dumps(ctx)) / 1024
        console.print(f"  [green]✓[/green] Portfolio context ready ({ctx_kb:.0f} KB)\n")

        print_portfolio_dashboard(ctx)
        print_portfolio_suggestions()

        console.print("[dim]  LangChain: ChatPromptTemplate | ChatAnthropic(sonnet-4-6) | StrOutputParser[/dim]")
        console.print("[dim]  Memory: SessionMemory · Session: portfolio[/dim]")
        console.print("[dim]  Commands: dashboard · help · clear · exit[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask("[blue]RM[/blue]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() in ("dashboard", "summary"):
                print_portfolio_dashboard(ctx)
                continue
            if user_input.lower() in ("help", "?"):
                print_portfolio_suggestions()
                continue
            if user_input.lower() == "clear":
                console.clear()
                print_portfolio_dashboard(ctx)
                continue

            blocked = check_guardrails(user_input)
            if blocked:
                console.print(Panel(blocked, border_style="yellow", title="[yellow]Note[/yellow]"))
                continue

            with console.status("[magenta]Thinking…[/magenta]"):
                try:
                    reply = invoke(frozen, user_input, memory)
                except Exception as e:
                    console.print(f"[red]  Error: {e}[/red]")
                    continue

            console.print(Panel(reply, border_style="magenta", title="[magenta]Assistant[/magenta]"))
            console.print()

    # ════════════════════════════════════════════════════════════════════════
    #  INVESTOR MODE
    # ════════════════════════════════════════════════════════════════════════
    else:
        print_investor_list()
        investor = None
        while not investor:
            raw = ask("Enter investor ID or name")
            investor = next(
                (i for i in DB["investors"]
                 if i["investor_id"].lower() == raw.lower()
                 or raw.lower() in i["investor_name"].lower()),
                None,
            )
            if not investor:
                console.print(f"[red]  Not found: \"{raw}\"[/red]\n")

        slice_ = build_investor_slice(investor["investor_id"])

        # Guardrail: data isolation
        assert_isolation(slice_)

        # Guardrail: KYC gate
        kyc_msg = kyc_gate(investor)
        if kyc_msg:
            console.print(Panel(kyc_msg, border_style="red", title="[red]Access Restricted[/red]"))
            sys.exit(0)

        print_slice_info(slice_)
        summary = compute_investor_summary(slice_)
        print_investor_dashboard(investor, summary)
        print_investor_suggestions(summary)

        with console.status("[cyan]Building investor chain…[/cyan]"):
            invoke, frozen, memory = build_investor_chain(api_key, slice_)

        console.print("[dim]  LangChain: ChatPromptTemplate | ChatAnthropic(sonnet-4-6) | StrOutputParser[/dim]")
        console.print(f"[dim]  Memory: SessionMemory · Session: {investor['investor_id']}[/dim]")
        console.print("[dim]  Commands: summary · help · clear · exit[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask("[cyan]You[/cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() in ("summary", "dashboard"):
                print_investor_dashboard(investor, summary)
                continue
            if user_input.lower() in ("help", "?"):
                print_investor_suggestions(summary)
                continue
            if user_input.lower() == "clear":
                console.clear()
                print_investor_dashboard(investor, summary)
                continue

            blocked = check_guardrails(user_input)
            if blocked:
                console.print(Panel(blocked, border_style="yellow", title="[yellow]Note[/yellow]"))
                continue

            with console.status("[magenta]Thinking…[/magenta]"):
                try:
                    reply = invoke(frozen, user_input, memory)
                except Exception as e:
                    console.print(f"[red]  Error: {e}[/red]")
                    continue

            # Guardrail: post-response audit
            leaks = audit_response(reply, investor["investor_id"])
            for leak in leaks:
                console.print(f"[yellow]  ⚠ Cross-investor ID in response: {leak}[/yellow]")

            console.print(Panel(reply, border_style="magenta", title="[magenta]EquiTie[/magenta]"))
            console.print()

    console.print(Rule("[dim]Session ended[/dim]"))


if __name__ == "__main__":
    main()
