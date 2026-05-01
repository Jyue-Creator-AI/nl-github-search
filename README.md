# GitHub Repo Search — GoFreight Take-Home Assessment

A CLI tool that converts natural language queries into GitHub repository searches using LLMs, with a multi-model eval pipeline measuring structured query generation accuracy.

---

## Why GitHub API

GitHub's search syntax is explicit and unambiguous (`language:python stars:>1000 created:>2024-01-01`), which makes ground truth verifiable without running queries against a live database. The domain is also directly relevant to GoFreight's context — AI agents operating in software development workflows.

Alternatives considered:
- **Wikidata**: SPARQL syntax too complex for a clean baseline; hallucination patterns harder to categorize
- **Data Commons**: Less developer-familiar; harder to design adversarial cases with clear ground truth

---

## Architecture

```
User input (natural language)
        ↓
  LLM (Claude / GPT-4o mini / Llama)
        ↓
  GitHub search query string
  e.g. language:python stars:>1000 topic:machine-learning
        ↓
  GitHub Search REST API
  GET /search/repositories?q=...
        ↓
  Formatted results (name, stars, description, URL)
```

**Key design decisions:**
- Single-turn prompt: query generation does not require multi-turn context
- Output-only constraint: LLM instructed to return query string only, no explanation
- Qualifier whitelist in system prompt: prevents hallucination of non-existent GitHub qualifiers
- Dynamic date injection: system prompt receives today's date and pre-calculated anchors at runtime

---

## How to Run

```bash
# Install dependencies
pip install anthropic openai groq requests python-dotenv

# Set up environment variables
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, GITHUB_TOKEN

# Run the CLI
python3 search.py "find popular Python machine learning repos"
python3 search.py "最近更新的 JavaScript 框架" --results 10

# Run the eval pipeline
python3 eval.py --prompt v5 --model all
python3 eval.py --prompt v5 --model claude
```

---

## Part 1: Baseline → Break → Harden

### Baseline

The baseline CLI (`prompts/v1_system_prompt.txt`) accepts natural language input, calls Claude Haiku to generate a GitHub search query, executes it against the GitHub Search API, and returns formatted results.

### Break It — Failure Case Taxonomy

I tested 6 systematic failure categories rather than random adversarial inputs. The goal was to find failure patterns that recur across different inputs, not one-off bugs.

| Category | Example Input | Failure Mode |
|---|---|---|
| Ambiguous intent | "find good Python repos" | LLM decides threshold non-deterministically (`stars:>100` vs `stars:>1000`) |
| Conflicting constraints | "brand new repos with 100000 stars" | LLM passes both constraints without flagging logical impossibility |
| Implicit filters | "repos I can contribute to" | LLM hallucinated `good-first-issue:>0`, a non-existent qualifier → silent empty result |
| Typos | "pythoon machine leraning repos" | LLM auto-corrects correctly — this category was robust |
| Non-English | "找最多星的機器學習專案" | Translation correct, but temporal references used static 2024 dates |
| Temporal ambiguity | "recent trending repos" | `pushed:>2024-01-01` hardcoded — becomes stale as time passes |

### Harden — What I Fixed and Why

**Fix 1: Qualifier whitelist** (`prompts/v2_system_prompt.txt`)

Root cause: LLM generates plausible-sounding but invalid GitHub qualifiers (e.g. `good-first-issue:>0`, `contributors:>50`). These produce silent empty results with no indication of what went wrong — the worst failure mode for a user-facing tool.

Fix: Added exhaustive list of valid GitHub qualifiers to system prompt. LLM can no longer invent qualifiers outside this set.

**Fix 2: Dynamic date injection**

Root cause: LLM's training cutoff creates a static temporal reference point. "Recent" reliably maps to 2024 dates regardless of when the tool is run.

Fix: Inject today's date and pre-calculated 3mo/6mo anchors at runtime. Date math uses `timedelta(days=90/180)` — not month arithmetic, which crashes on month-end dates (e.g. May 31 → Feb 31 does not exist).

**Fix 3: Empty result explanation**

Root cause: Original output for zero results was `No results found.` — no indication of whether the query was bad, constraints too strict, or input contradictory.

Fix: Empty results now surface the generated query string and suggest relaxing constraints.

### What I Didn't Fix — and Why It's Fundamentally Hard

**Ambiguous intent (threshold non-determinism)**

"Good" has no universal definition. Whether `stars:>100` or `stars:>1000` is "correct" depends on user context. Fixing this requires either: (a) a clarifying question — breaks single-turn design, or (b) hardcoding a convention that is always wrong for some users. I documented the eval convention (`stars:>1000`) rather than pretending it's solved.

**Conflicting constraints**

"Brand new repos with 100,000 stars" is logically impossible. Fixing this requires semantic validation of GitHub's data distribution, which is outside a query generator's scope. Fix 3's empty result message handles the downstream consequence.

---

## Part 2: Multi-Model Eval Pipeline

### Ground Truth Design

**Evaluation method: Normalized exact match**

Qualifiers are sorted alphabetically then lowercased before comparison:
- `Stars:>1000 language:Python` == `language:python stars:>1000` → match ✓

**Why not execution-based comparison?**
GitHub search results change over time. Two structurally equivalent queries evaluated by result overlap on different days would appear different, adding noise that obscures prompt quality.

**Why not LLM-as-judge?**
Introduces a fourth model whose biases affect the score. Harder to reproduce, harder to explain.

**Known limitation of exact match:**
Ambiguous-intent cases (5 of 30) have multiple structurally valid answers. A model outputting `stars:>500` instead of `stars:>1000` for "popular repos" is judged incorrect even if reasonable. This is a documented design tradeoff — the alternative requires another model and can't be independently reproduced.

**Threshold conventions (applied consistently across all 30 cases, eval date: 2026-05-01):**
- `popular` / `good` = `stars:>1000`
- `active` = `archived:false pushed:>2025-11-02` (both required)
- `good` / `useful` = `stars:>1000 archived:false` (both required)
- `recent` / `lately` = `pushed:>2026-01-31` (90 days)
- `new` (6 months) = `pushed:>2025-11-02` (180 days)
- `this year` = `created:>2026-01-01`
- `brand new` (1 month) = `created:>2026-04-01`

### Test Case Design — 30 Cases

| Category | Count | Purpose |
|---|---|---|
| Basic | 5 | Establish floor — single qualifier, unambiguous |
| Time filters | 5 | Validate date injection fix |
| Multi-condition | 8 | Test qualifier composition under explicit constraints |
| Ambiguous intent | 5 | Stress-test threshold conventions |
| Adversarial | 7 | Typos, Chinese input, conflicting constraints, implicit filters |

Full cases with ground truth: `data/test_cases.json`

### Model Selection

| Model | Type | Why capable of reaching threshold |
|---|---|---|
| Claude Haiku 4.5 | Closed-source | Strong instruction-following; primary tool in GoFreight's stated stack |
| GPT-4o mini | Closed-source | Strongest closed-source competitor at low cost; reliable structured output |
| Llama 3.3 70B (Groq) | Open-weight | Largest open-weight model accessible without local GPU; 3.3 generation shows strong instruction following |

All three were selected for their demonstrated ability to follow strict output-format constraints — critical for exact-match eval.

### Prompt Iteration Log

| Version | Key Change | Claude Haiku | GPT-4o mini | Llama 3.3 70B |
|---|---|---|---|---|
| v2 | Qualifier whitelist + date injection | 30.0% | — | — |
| v3 | Strict rules + topic mappings + no-stars rule | 63.3% | — | — |
| v4 | archived:false trigger rules + AI topic fix + updated=pushed | 83.3% | 90.0% | 90.0% |
| v5 | active=pushed enforced + no pushed when explicit created: + framework compound noun fix | **90.0%** | **93.3%** | **96.7%** |

*v1 was used for manual Break It testing only; v3 and earlier were not run on GPT/Llama to conserve API calls during iteration.*

### Performance Comparison (v5, final)

| Model | Overall | Basic | Time | Multi | Ambiguous | Adversarial |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | **90.0%** (27/30) | 5/5 | 4/5 | 7/8 | 5/5 | 6/7 |
| GPT-4o mini | **93.3%** (28/30) | 5/5 | 4/5 | 8/8 | 5/5 | 6/7 |
| Llama 3.3 70B | **96.7%** (29/30) | 5/5 | 5/5 | 8/8 | 5/5 | 6/7 |

**Where each model failed:**
- **Claude**: Case #8 (6-month date off by 1 day), #15 ("created in 2024" over-interpreted as range), #25 (中文 "最多星" added `language:zh`)
- **GPT-4o mini**: Case #8 (date calculation), #28 (typo correction dropped `language:javascript`)
- **Llama**: Case #25 (中文 "最多星" → `stars:>10000` instead of `stars:>1000`)

**Surprising finding:** Llama 3.3 70B outperformed both closed-source models. At v5, the prompt constraints were precise enough that model size and instruction-following quality mattered more than whether the model was open or closed weight.

### Learnings

**1. Eval design determines what you're actually measuring**

The hardest part of this assignment was not writing the query generator — it was defining what "correct" means. Normalized exact match sounds simple but requires committing to specific threshold conventions (`stars:>1000` for "popular") and accepting that ambiguous cases will always have edge-case failures. The ground truth rules I documented are opinionated; a different engineer would make different choices. What matters is consistency and transparency, not perfection.

**2. Prompt iteration needs failure categorization, not trial-and-error**

Going from 30% to 90% across 5 prompt versions was possible because I categorized failure root causes before changing anything. Each version targeted one category of failures (hallucinated qualifiers → qualifier whitelist; temporal drift → date injection; over-triggering → conditional rules). When you change one thing at a time, you can explain exactly why accuracy improved.

**3. Ground truth and prompt must be co-designed**

I initially wrote ground truths on Day 3 and prompts on Day 4. This created drift: the prompt's definition of "active" (archived:false + pushed) didn't match the ground truth for case #17 (archived:false only). The fix required updating both. In production, eval criteria and system behavior need to be defined together, not sequentially.

**4. Date-relative ground truths are fragile**

All time-relative ground truths were calculated at eval time (2026-05-01). Running the same eval the next day would produce different ground truths from the model (since the prompt injects today's date) while the stored ground truths stay fixed — causing false negatives. The production fix is to store relative expressions and calculate at eval time, not fix dates.
