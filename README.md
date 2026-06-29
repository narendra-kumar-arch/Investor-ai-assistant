# Investor Assistant — Python Prototype

Python + LangChain + Rich terminal chatbot. All 10 dataset tables. Two modes.
Covers every question type in the brief: portfolio overview, single positions,
obligations, distributions, fees, valuations, account statements, and personalisation.

## Requirements

- Python 3.10+
- An Anthropic API key — https://console.anthropic.com

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python assistant.py
```

The app prompts for the key if the env var is not set (input is hidden).

## Two modes

**[P] Portfolio mode** — RM / admin view, full shared dataset:

| Ask | What it shows |
|---|---|
| "Which investors have overdue fees? List with amounts" | 60 investors, specific amounts and due dates |
| "List all upcoming capital calls due before 31 July" | 106 calls, amounts, investor IDs |
| "Top 5 investors by current portfolio value" | Ranked from pre-computed totals |
| "Show all investors in Forgecraft Robotics" | Multi-round company filter |
| "What distributions did Helianthe Energy investors receive?" | Exit proceeds, net of carry |
| "Show Qubrium's valuation history — the down round" | Entry 10.0 → current 6.2 |
| "Which investors got a fee discount on their allocations?" | Reads fee_discount flag + effective vs std rates |

**[I] Investor mode** — single investor Q&A, all 10 tables:

| Ask | What it shows |
|---|---|
| "Portfolio overview" | Value, MOIC, DPI/RVPI, contributions, obligations |
| "Walk me through my holdings" | Each position: value, cost basis, mark, MOIC |
| "What fees am I paying, and do I have discounts?" | Effective vs deal standard, per-deal |
| "Walk me through my full account statement" | Signed cash flows from statement_lines |
| "What distributions have I received?" | Gross, carry withheld, net received |
| "How has Forgecraft's valuation moved?" | Full mark history from valuations table |
| "Do I have any upcoming capital calls?" | Amount, due date, deal |

## Personalisation

The system prompt is built with computed investor signals:
- `deal_count`, `top_sectors`, `companies`, `concentration` derived from allocations + deals + companies
- `tech_savviness`, `age` from the investor profile

Three persona branches:
- **Low tech or age ≥ 65** — plain language, jargon defined on first use (MOIC, carry, SPV, DPI), short answers, patient tone
- **High tech + many deals** — concise, data-dense, skip definitions, assume fluency
- **Otherwise** — balanced, professional, define niche terms

## Data coverage

All 10 tables are loaded. Per mode:

| Table | Investor mode | Portfolio mode |
|---|---|---|
| investors | 1 row (this investor) | 112 rows (aggregated totals) |
| allocations | filtered to investor | 550 rows (raw, for fee discount Qs) |
| deals | filtered to investor's deals | 21 rows (aggregated totals) |
| companies | all 16 | all 16 |
| valuations | full history, filtered | 55 rows, full history |
| capital_calls | filtered | 106 upcoming (raw) |
| fees | filtered (all statuses) | 594 actionable: Overdue + Upcoming (raw) |
| distributions | filtered | 34 rows (raw) |
| statement_lines | filtered | not included (per-investor only) |
| fx_rates | all 4 | all 4 |

## Guardrails

All run deterministically **before** any API call:

| Layer | What it catches |
|---|---|
| Investment advice | "should I sell?", "will this recover?", "recommend exiting", "price target" |
| Prompt injection | "ignore your instructions", "jailbreak", "you are now", "pretend you are" |
| KYC gate | Blocks portfolio detail for Pending KYC investors (INV021 Grace Okafor, INV023 Lara Greco) |
| Data isolation | `assert_isolation()` raises if a foreign investor's row appears in the slice |
| Post-response audit | `audit_response()` flags any cross-investor ID appearing in model output |

## LangChain components

| Component | Role |
|---|---|
| `ChatAnthropic` | Model wrapper — claude-sonnet-4-6, temp 0.2 |
| `ChatPromptTemplate` | Typed prompt: system template with named slots + MessagesPlaceholder |
| `MessagesPlaceholder` | Injects conversation history into the prompt at call time |
| `StrOutputParser` | Extracts plain text from the AIMessage response |
| `SessionMemory` | Custom per-session message store; replaces deprecated RunnableWithMessageHistory |

## In-session commands

| Type | Effect |
|---|---|
| Any question | Answered by Claude Sonnet 4.6 |
| `summary` / `dashboard` | Re-prints the dashboard |
| `help` / `?` | Shows suggested questions |
| `clear` | Clears screen |
| `exit` / `quit` or Ctrl+C | Ends session |

## Good demo sequence

1. Run in **[P] Portfolio mode**
   - "which investors have overdue fees?" → 60 investors with amounts
   - "show Qubrium's valuation history" → down-round: entry 10.0, current 6.2
   - "what distributions did Helianthe investors receive?" → exit proceeds + carry

2. Run in **[I] Investor mode**, select `INV001` (Idris Olawale, High tech)
   - "give me a portfolio overview" → £438K value, 2.60× MOIC
   - "what fees am I paying on Inferna AI and do I have a discount?" → 1% vs 2% std

3. Run again, select `INV017` (Elena Petrova, Low tech, age 67)
   - Same question → plain language, jargon defined, shorter answer

4. Select `INV021` (Grace Okafor) → KYC gate blocks immediately

## Known limitations

- Arithmetic delegated to the model when complex (e.g. multi-round MOIC aggregation). The dashboard cards are computed deterministically in Python as a cross-check.
- No streaming — full response returned before display.
- In-memory session only — history lost on exit.
- statement_lines excluded from portfolio mode (1,390 rows, per-investor only; no cross-portfolio question requires it).
