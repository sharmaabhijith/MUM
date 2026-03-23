# MUM: Multi-User Memory Benchmark

A production-quality benchmark generation pipeline for evaluating AI memory systems in multi-user settings. MUM extends the LoCoMo paradigm (long multi-session conversations) into a **multi-user setting**: multiple users independently interact with an AI assistant over shared documents, and the benchmark evaluates whether a system can extract, manage, and reconcile knowledge across these parallel conversation streams.

**What makes this different from LoCoMo**: LoCoMo tests single-thread memory (2 speakers, 1 conversation, ~300 turns, ~19 sessions). MUM tests **cross-user** memory: 3-4 users per scenario, each with their own long conversation history, producing conflicting facts, authority gradients, and temporal dependencies that a memory system must reconcile.

## Key Metrics

| Metric | Value |
|--------|-------|
| Scenarios | 5 (one per relationship type) |
| Source documents | 15 PDFs (~87K tokens total) |
| Unique users | 18 (3-4 per scenario) |
| Sessions per user | 7 |
| Turns per session | 15-20 |
| Total sessions | ~133 |
| Total turns | ~2,340 |
| Extracted memories | ~465+ (~3.5 per session) |
| Conflict annotations | ~30+ |
| Evaluation categories | 12 (8 universal + 2 cluster + 2 specific) |
| Evaluation questions | 1,200 (100 per category) |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [The 5 Scenarios](#the-5-scenarios)
- [4-Phase Generation Pipeline](#4-phase-generation-pipeline)
- [Evaluation Framework](#evaluation-framework)
- [12 Evaluation Categories](#12-evaluation-categories)
- [Data Schemas](#data-schemas)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Prompt Design](#prompt-design)
- [Adding a New Scenario](#adding-a-new-scenario)
- [Testing](#testing)
- [Design Principles](#design-principles)
- [License](#license)

---

## Installation

**Requirements**: Python >= 3.10

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Copy and fill in your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

### Environment Variables

```bash
OPENAI_API_KEY=sk-your-key-here
MUM_MODEL_CONVERSATION=gpt-4o-mini    # Model for conversation generation
MUM_MODEL_ANNOTATION=gpt-4o           # Model for memory/conflict annotation
MUM_MODEL_EVAL=gpt-4o                 # Model for evaluation scoring
```

---

## Quick Start

```bash
# 1. Validate source documents exist and meet token targets
python -m src.cli validate-docs --scenario 1

# 2. Dry run (mock LLM, no API calls — test the pipeline)
python -m src.cli generate --scenario 1 --dry-run

# 3. Estimate API cost before committing
python -m src.cli estimate-cost --all

# 4. Generate benchmark for one scenario
python -m src.cli generate --scenario 1

# 5. Generate all 5 scenarios
python -m src.cli generate --all

# 6. Run evaluation with a baseline
python -m eval.run_eval --scenario 1 --baseline full_context

# 7. Export to HuggingFace datasets format
python -m src.cli export --format huggingface
```

---

## Project Structure

```
mum/
├── README.md
├── IMPLEMENTATION.md              # Detailed implementation specification
├── pyproject.toml                 # Package metadata and dependencies
├── Makefile                       # Development task shortcuts
├── .env.example                   # API key template
│
├── config/
│   ├── models.yaml                # LLM model configs (iteration / final / eval)
│   ├── generation.yaml            # Global generation parameters
│   └── scenarios/
│       ├── scenario_1.yaml        # Study Group (symmetric)
│       ├── scenario_2.yaml        # Exec + Team (hierarchical)
│       ├── scenario_3.yaml        # Contract Review (cross-functional)
│       ├── scenario_4.yaml        # Negotiation (adversarial)
│       └── scenario_5.yaml        # Support Escalation (sequential)
│
├── documents/                     # Source PDFs (user-provided, gitignored)
│   ├── scenario_1/               # textbook_chapter.pdf, lecture_slides.pdf, past_exam.pdf
│   ├── scenario_2/               # quarterly_report.pdf, data_appendix.pdf, board_presentation.pdf
│   ├── scenario_3/               # master_services_agreement.pdf, statement_of_work.pdf, pricing_schedule.pdf
│   ├── scenario_4/               # asset_purchase_agreement.pdf, valuation_report.pdf, comparable_transactions.pdf
│   └── scenario_5/               # customer_ticket.pdf, product_documentation.pdf, system_error_logs.pdf
│
├── src/
│   ├── __init__.py
│   ├── cli.py                     # Click-based CLI entry point
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py        # Main pipeline orchestrator (runs all phases)
│   │   ├── phase1_document_prep.py
│   │   ├── phase2_conversation.py
│   │   ├── phase3_annotation.py
│   │   └── phase4_validation.py
│   │
│   ├── models/
│   │   ├── schemas.py             # All Pydantic data models
│   │   └── enums.py               # RelationshipType, ConflictType, MemoryType, etc.
│   │
│   ├── prompts/
│   │   ├── base.py                # PromptBuilder utilities
│   │   ├── conversation_system.py # System prompt for conversation generation
│   │   ├── conversation_user.py   # Trigger prompt for user turns
│   │   ├── session_summary.py     # Session summary generation
│   │   ├── memory_extraction.py   # Memory extraction instructions
│   │   ├── conflict_detection.py  # Cross-user conflict detection
│   │   ├── eval_question_gen.py   # Per-category eval question generation
│   │   └── quality_check.py       # Validation pass prompts
│   │
│   ├── llm/
│   │   ├── client.py              # LLMClient (OpenAI) + MockLLMClient
│   │   ├── token_counter.py       # Tiktoken-based token counting
│   │   └── cost_tracker.py        # Per-call and cumulative cost tracking
│   │
│   ├── scenarios/
│   │   ├── __init__.py            # Factory functions for scenario loading
│   │   ├── base.py                # BaseScenario ABC
│   │   ├── scenario_1.py          # StudyGroupScenario
│   │   ├── scenario_2.py          # ExecTeamScenario
│   │   ├── scenario_3.py          # ContractReviewScenario
│   │   ├── scenario_4.py          # NegotiationScenario
│   │   └── scenario_5.py          # SupportScenario
│   │
│   └── utils/
│       ├── pdf_reader.py          # PDF extraction (PyMuPDF / fitz)
│       ├── io.py                  # YAML, JSON, JSONL read/write
│       └── logging.py             # Rich-based structured logging
│
├── eval/
│   ├── run_eval.py                # EvalRunner + MemorySystemInterface ABC
│   ├── metrics.py                 # ROUGE, F1, exact match, LLM-as-judge, FactScore
│   └── baselines/
│       ├── full_context.py        # Baseline: all conversations as raw context
│       └── rag_summaries.py       # Baseline: RAG over session summary embeddings
│
├── tests/
│   ├── test_schemas.py
│   ├── test_prompts.py
│   ├── test_scenarios.py
│   └── test_pipeline.py
│
├── scripts/
│   ├── validate_documents.py      # Validate source PDFs
│   ├── dry_run.py                 # Test with mock LLM
│   ├── cost_estimate.py           # Pre-run cost estimation
│   └── export_hf.py               # Export to HuggingFace datasets
│
├── docs/                          # Additional documentation
│   ├── GENERATION_GUIDE.md
│   ├── PROMPT_DESIGN.md
│   ├── SCHEMA_REFERENCE.md
│   └── EVALUATION_GUIDE.md
│
└── output/                        # Generated benchmark data (gitignored)
    ├── scenario_1/
    │   ├── conversations/         # Per-user, per-session JSON files
    │   ├── summaries/
    │   ├── memories/
    │   ├── conflicts/
    │   ├── evaluation/
    │   ├── validation/
    │   └── scenario_1_complete.json
    ├── scenario_2/ ... scenario_5/
    ├── benchmark.json             # Final combined benchmark
    └── generation_report.json     # Costs, tokens, timing, quality metrics
```

---

## The 5 Scenarios

Each scenario models a distinct multi-user relationship type, with different authority structures, conflict dynamics, and resolution rules. Every user is defined with 10+ roleplay fields that control their voice, reading patterns, knowledge gaps, misconceptions, emotional state, and session-by-session evolution across 7 sessions.

### Conversation Scale Summary

| Scenario | Users | Sessions/User | Turns/Session | Turns/User | Total Turns | Doc Tokens | Conflicts | Eval Qs |
|----------|-------|---------------|---------------|------------|-------------|------------|-----------|---------|
| 1 Study Group | 4 | 7 | 18 | 126 | 504 | 15K | 3 | 175 |
| 2 Exec + Team | 4 | 7 | 16 | 112 | 448 | 22K | 3 | 235 |
| 3 Contract Review | 4 | 7 | 18 | 126 | 504 | 18K | 3 | 200 |
| 4 Negotiation | 4 | 7 | 18 | 126 | 504 | 20K | 7 | 315 |
| 5 Support | 3 | 7 | 18 | 126 | 378 | 12K | 2 | 275 |
| **Total** | **18** | | | | **~2,340** | **87K** | **~18** | **1,200** |

### 7-Session Narrative Arc

Each user follows a consistent narrative arc across their 7 sessions:

```
Session 1 — Orientation:     First read of documents. Surface-level questions.
Session 2 — Focused Dive:    Drill into primary area of expertise.
Session 3 — Cross-Topic:     Explore secondary topics. Early conflicts may surface.
Session 4 — Deep Analysis:   Core conflicts emerge. Strong opinions formed.
Session 5 — Revision:        Revisit earlier conclusions. Positions evolve. Corrections happen.
Session 6 — Synthesis:       Consolidate understanding. Ask cross-cutting questions.
Session 7 — Final Review:    Last pass. Summary requests. Lingering doubts addressed.
```

---

### Scenario 1: Study Group (Symmetric)

| | |
|---|---|
| **Relationship** | Symmetric — peers, equal authority |
| **Domain** | Education / Distributed Systems CS |
| **Timeline** | 2 weeks (exam prep period) |
| **Documents** | Textbook chapter (~8K), lecture slides (~4K), past exam (~3K) — total ~15K tokens |
| **Resolution Rule** | Preserve both positions (symmetric peers, no authority override) |
| **Primary Eval Stress** | Equal-weight attribution; corrections between peers; nobody dominates |
| **Applicable Categories** | All 8 universal + Temporal Correction Chain |

#### Users

**Student A** (authority: EQUAL, weight: 1.0)
- **Expertise**: Strong on theory, weak on implementation
- **Focus**: Clock algorithms, ordering proofs, formal correctness
- **Style**: Formal and precise. Uses technical jargon, numbered sub-points, theorem citations. Treats the AI like a teaching assistant.
- **Reading Pattern**: Deep reader of theoretical sections. Reads proofs line by line. Skips pseudocode and implementation examples entirely.
- **Knowledge Gaps**: Cannot read pseudocode fluently; weak on practical system design; doesn't understand why Raft was created if Paxos is provably correct
- **Misconceptions**: Believes vector clocks are the most exam-relevant topic (~30% of exam)
- **Corrections**: Gracefully accepts on theory with curiosity. Defensive if told their theory is wrong without formal proof. Dismissive of implementation corrections.
- **Evolution**: Tentative in Session 1 → deep-dives Lamport/vector clocks in Session 2 → struggles with Paxos implementation in Session 3 → strong opinions on vector clocks in Session 4 → grudgingly acknowledges implementation matters in Session 5 → synthesizes theory + practice in Session 6 → final review, still theory-heavy but more balanced in Session 7
- **Example**: *"Can you walk me through the proof of the vector clock partial ordering property from Theorem 3.4? I want to make sure the antisymmetry condition holds under concurrent events."*

**Student B** (authority: EQUAL, weight: 1.0)
- **Expertise**: Strong on coding, weak on theory
- **Focus**: Consensus implementations, Raft/Paxos code, system design
- **Style**: Casual and direct. Programming metaphors ("it's like a state machine with three transitions"). Short, pointed questions. Impatient with long theory.
- **Reading Pattern**: Skims theory. Jumps straight to pseudocode and implementation examples. Reads past exam solutions backward (starts with answers).
- **Knowledge Gaps**: Cannot follow mathematical proofs; confuses safety vs liveness properties; doesn't understand TLA+
- **Misconceptions**: Believes Raft is strictly superior to Paxos in all scenarios
- **Corrections**: Quick acceptance when shown code that proves the point. Resistant when corrected with pure theory ("But does it actually work in practice?").
- **Evolution**: Overwhelmed by theory in Session 1 → deep-dives Raft implementation in Session 2 → forced to look at Paxos (complains) in Session 3 → crystallized opinion that Raft is the only sane exam choice in Session 4 → grudgingly revisits Paxos in Session 5 → builds mental implementation of both in Session 6 → pragmatic final review in Session 7
- **Example**: *"lol the textbook calls Paxos 'elegant.' Have they tried implementing it?"*

**Student C** (authority: EQUAL, weight: 1.0)
- **Expertise**: Average across all areas
- **Focus**: Exam strategy, high-yield topics, past exam patterns
- **Style**: Casual, efficiency-oriented. Meta-questions ("is this worth studying?", "what's the marks breakdown?"). Short messages.
- **Reading Pattern**: Strategic skimmer. Reads past exam first for patterns, then textbook sections that map to exam questions. Quick, misses details.
- **Knowledge Gaps**: Shallow understanding of all topics; can't derive anything from first principles
- **Misconceptions**: Believes BFT requires 2f+1 nodes (actually 3f+1); thinks vector clocks are minor and worth skipping
- **Corrections**: "Oh wait, really? Thanks for catching that." Accepts easily but may not deeply internalize — might repeat the same mistake later if not reinforced.
- **Evolution**: Scoping/study plan in Session 1 → high-yield topics in Session 2 → encounters BFT, confidently states wrong formula in Session 3 → gets corrected on BFT in Session 4 → double-checks correction stuck in Session 5 → practice problems in Session 6 → final cram ("if I only have 2 hours left...") in Session 7
- **Example**: *"Quick question — for the BFT section, the textbook says you need 2f+1 nodes to tolerate f Byzantine faults, right?"*

**Student D** (authority: EQUAL, weight: 1.0)
- **Expertise**: Took prerequisite course in distributed computing
- **Focus**: Consensus + replication, prior coursework connections, formal verification
- **Style**: Confident and professorial. Frequently starts with "In my distributed computing prerequisite, we..." Draws analogies. Enjoys explaining. Slightly condescending about prior experience.
- **Reading Pattern**: Reads with a comparative lens — constantly checking against prerequisite notes. Notices inconsistencies between sources. May miss things new to this course but not in the prerequisite.
- **Knowledge Gaps**: New topics not in the prerequisite; overestimates how much prerequisite knowledge transfers directly
- **Misconceptions**: Assumes all notation from prerequisite is universal (different textbooks use different conventions)
- **Corrections**: Initially resistant ("But in my prerequisite, we proved that..."). Then thoughtful. Eventually integrates with synthesis ("So the distinction is..."). Corrects others confidently with citations.
- **Evolution**: Notices overlap with prerequisite in Session 1 → compares Paxos treatment to TLA+ proofs in Session 2 → finds areas where prior knowledge doesn't apply in Session 3 → strong opinions on Paxos elegance, corrects C's BFT misconception in Session 4 → reflects on prior knowledge in Session 5 → synthesis across topics in Session 6 → identifies remaining gaps, offers to help in Session 7
- **Example**: *"Actually, BFT requires 3f+1 nodes, not 2f+1. The 2f+1 threshold is for crash faults only — I remember proving this distinction in my prerequisite."*

#### Injected Conflicts

| ID | Users | Topic | Nature | Resolution |
|----|-------|-------|--------|------------|
| C1-1 | Student B vs D | Paxos vs Raft complexity | B: Raft simpler to implement. D: Paxos more elegant theoretically. | Preserve both views |
| C1-2 | Student A vs C | Vector clocks exam relevance | A: heavily tested (~30%). C: not worth deep study. | Preserve both strategies |
| C1-3 | Student D vs C | BFT threshold | C: 2f+1 (wrong). D/AI/Textbook: 3f+1 (correct). | Supersede C's claim (factual correction) |

#### Correction Chains Tested

- Student C states BFT requires 2f+1 → Student D/AI corrects to 3f+1 → Student C accepts → Does C's later session reflect the correction or revert?
- Student A believes vector clocks are 30% of exam → Is this ever challenged? What's the final state?
- Notation inconsistency (increment-then-send vs send-then-increment) → Who discovers it? Is it resolved?

---

### Scenario 2: Exec + Team (Hierarchical)

| | |
|---|---|
| **Relationship** | Hierarchical — clear authority gradient |
| **Domain** | Business / Finance, Board Presentation Prep |
| **Timeline** | ~1.5 weeks (board prep sprint) |
| **Documents** | Quarterly report (~10K), data appendix (~7K), board presentation (~5K) — total ~22K tokens |
| **Resolution Rule** | Authority wins on opinions/narrative tone, but facts are flat regardless of rank |
| **Primary Eval Stress** | Authority weighting; junior facts override senior opinions; multiple correction chains |
| **Applicable Categories** | All 8 universal + Authority & Hierarchy Resolution + Temporal Correction Chain |

#### Users

**VP of Operations** (authority: HIGH, weight: 3.0)
- **Expertise**: Strategic narrative, executive communication, board management
- **Focus**: Executive summary, board messaging, narrative tone, Q4 outlook
- **Style**: Commanding and decisive. "The board needs confidence", "Let's lead with wins". Short, directive. Doesn't ask — tells. Speaks in outcomes and narrative, not data.
- **Reading Pattern**: Reads executive summary and board slides first. Skims data appendix — trusts the team. Focuses on how numbers FEEL to a board member. Reads speaker notes more carefully than slide content.
- **Knowledge Gaps**: Doesn't verify raw data personally; may miss numerical errors buried in appendix tables; over-relies on team for accuracy
- **Misconceptions**: Assumes the $42M figure in the exec summary is correct because it's in the official report
- **Corrections**: Accepts factual corrections on data ("fix the number") but pushes back on narrative ("I decide the tone, not the data"). Will acknowledge a risk flag but explicitly choose not to include it.
- **Example**: *"Don't put risk language in the slides. If the board asks about Q4, I'll handle it verbally."*

**Senior Analyst** (authority: MEDIUM, weight: 2.0)
- **Expertise**: Financial modeling, data accuracy, KPI analysis, risk assessment
- **Focus**: KPI tables, revenue calculations, Q4 pipeline coverage, data appendix accuracy
- **Style**: Precise and data-driven. Cites specific numbers, table references, percentage changes. Hedging language ("the data suggests"). Respectful but firm when pushing back on the VP.
- **Reading Pattern**: Reads data appendix first and thoroughly. Cross-references every number in the exec summary against appendix tables. Reads the board presentation with a "fact-checker" lens.
- **Knowledge Gaps**: Less attuned to board politics and messaging nuance
- **Corrections**: Very receptive — "show me the data." Will NOT back down if their data is correct and the VP wants to ignore it.
- **Example**: *"I'm concerned about the Q4 messaging. The appendix shows pipeline coverage at 2.1x, which is below our 3.0x target."*

**Marketing Lead** (authority: MEDIUM, weight: 2.0)
- **Expertise**: Campaign performance, marketing attribution, demand generation metrics
- **Focus**: Marketing KPIs, campaign ROI, attribution modeling, demand generation
- **Style**: Enthusiastic and promotional. "Campaign-driven demand", "marketing-sourced revenue". Presents the most favorable data interpretation. Asks leading questions that frame marketing positively.
- **Reading Pattern**: Goes straight to marketing sections. Reads campaign data selectively — focuses on wins, skims failures. Checks that the board presentation gives marketing proper credit.
- **Knowledge Gaps**: Doesn't deeply understand financial modeling; weak on correlation vs causation in attribution
- **Misconceptions**: Believes 35% of Q3 revenue was directly "campaign-driven" (actually "campaign-influenced" at best)
- **Corrections**: Initially defensive ("But the data shows..."). Concedes when shown attribution methodology is flawed. Accepts the VP's moderation without resentment.
- **Example**: *"Our campaigns drove 35% of Q3 revenue. The attribution data is clear on this."*

**Junior Associate** (authority: LOW, weight: 1.0)
- **Expertise**: General business, cross-referencing, formatting, detail-oriented review
- **Focus**: Broad review, cross-referencing, formatting, catching errors
- **Style**: Tentative and questioning. "I might be reading this wrong, but...", "Sorry if this is a dumb question, but...". Short messages. Asks for confirmation before asserting.
- **Reading Pattern**: Reads everything cover to cover — the only team member who does. Cross-references obsessively. Catches details others miss because they read the footnotes. Slow and methodical.
- **Knowledge Gaps**: Doesn't understand financial modeling deeply; can't assess strategic implications; doesn't know what matters to the board
- **Corrections**: Grateful and relieved when confirmed correct. If told wrong, immediately defers ("You're probably right, I'll double-check"). Doesn't argue even when they should.
- **Example**: *"I might be reading this wrong, but the exec summary says $42M in enterprise revenue and the appendix shows $4.2M. Is this a typo?"*

#### Injected Conflicts

| ID | Users | Topic | Nature | Resolution |
|----|-------|-------|--------|------------|
| C2-1 | VP vs Senior Analyst | Q4 pipeline risk in board slides | VP: exclude risk language for confidence. Analyst: include pipeline gap data (2.1x vs 3.0x target). | Authority wins on narrative tone; data preserved as backup |
| C2-2 | Marketing Lead vs VP | Marketing attribution language | Marketing: "campaign-driven revenue." VP: "campaign-influenced pipeline." | Authority wins (VP moderates language) |
| C2-3 | Junior Associate vs VP | Revenue figure ($42M vs $4.2M) | Junior: finds the error. VP: hadn't noticed. Factual — junior is correct. | Supersede — factual correction wins regardless of authority |

#### Correction Chains Tested

- Executive summary says $42M → Junior Associate finds it's $4.2M → Senior Analyst validates → VP acknowledges → Is the board presentation corrected?
- Marketing claims 35% "campaign-driven" → VP moderates to "campaign-influenced" → Does Marketing accept in later sessions?
- Analyst flags pipeline at 2.1x vs 3.0x target → VP explicitly excludes from slides → What is the final state of this concern?

---

### Scenario 3: Contract Review (Cross-Functional)

| | |
|---|---|
| **Relationship** | Cross-functional — domain expertise-based authority (authority rotates by topic) |
| **Domain** | Business / Legal, Vendor Contract Review |
| **Timeline** | 3 weeks (review cycle with milestones) |
| **Documents** | Master Services Agreement (~8K), Statement of Work (~5K), Pricing Schedule (~5K) — total ~18K tokens |
| **Resolution Rule** | Domain expert wins on their domain (Legal on legal risk, Engineering on technical feasibility, Finance on cost, Procurement on vendor negotiability) |
| **Primary Eval Stress** | Rotating domain authority; who's expert depends on topic; richest cross-user synthesis |
| **Applicable Categories** | All 8 universal + Authority & Hierarchy Resolution |

#### Users

**Legal Counsel** (authority: MEDIUM, weight: 2.0, domain: legal)
- **Expertise**: Contract law, IP rights, liability, indemnification
- **Focus**: IP clause (Section 4), liability caps (Section 7), indemnification (Section 8)
- **Style**: Formal, precise legal language. "As written", "on its face", "carve-out". Structures analysis as: issue → risk → recommendation. Decisive — calls things "deal-breakers" or "acceptable risk".
- **Reading Pattern**: Reads the MSA clause by clause starting with IP, liability, indemnification. Marks up ambiguous phrases. Cross-references defined terms. Reads the SOW only to assess legal exposure from SLA commitments.
- **Knowledge Gaps**: Cannot assess technical SLA feasibility; doesn't understand engineering trade-offs; may overstate risk for technically standard terms
- **Corrections**: On legal matters: resistant ("the legal risk is real regardless of industry practice"). On technical matters: defers gracefully.
- **Example**: *"Section 4.2(a) assigns all derivative works to the Vendor. This is a deal-breaker as written."*

**Finance Lead** (authority: MEDIUM, weight: 2.0, domain: financial)
- **Expertise**: Pricing analysis, budgeting, payment terms, total cost of ownership
- **Focus**: Payment schedule, total cost analysis, NET-30 vs NET-60, milestone payments
- **Style**: Numbers-oriented. "Cash flow impact", "NPV", "working capital". Presents arguments with spreadsheet-style breakdowns. Frames everything as "what does this cost us?"
- **Reading Pattern**: Goes straight to pricing schedule. Builds mental total cost model. Reads payment terms in the MSA. Ignores legal and technical clauses unless they have financial penalties attached.
- **Knowledge Gaps**: Doesn't understand SLA technical requirements; doesn't assess legal risk; focused on cost, may miss quality/risk trade-offs
- **Corrections**: Readily accepts if shown better numbers. Pushes back on payment terms with financial modeling.
- **Example**: *"NET-60 gives us an extra 30 days of working capital. Under NET-30, we'd need to draw on the credit facility."*

**Engineering Manager** (authority: MEDIUM, weight: 2.0, domain: technical)
- **Expertise**: Technical delivery, SLA feasibility, integration architecture, system design
- **Focus**: SLA metrics (99.9% uptime), deliverable specs, integration requirements, technical feasibility
- **Style**: Direct, practical, engineering-oriented. "Service tier", "connection pooling", "failover architecture". Gets impatient with non-technical concerns about technically standard terms.
- **Reading Pattern**: Reads the SOW first for deliverables and SLAs. Reads the MSA's technical sections. Skims pricing. Reads the IP clause casually — doesn't see the legal risk.
- **Knowledge Gaps**: Doesn't understand legal nuance of IP assignment; may underestimate legal risk of technically standard terms
- **Misconceptions**: Believes the IP clause is standard and harmless for managed services
- **Corrections**: Carefully considers on technical matters. On IP clause: "I see your point, but in managed services this is standard practice."
- **Example**: *"For a managed services engagement, IP assignment to the vendor is pretty standard. They're bringing their proprietary tooling."*

**Procurement Lead** (authority: MEDIUM, weight: 2.0, domain: vendor)
- **Expertise**: Vendor management, contract benchmarking, market intelligence, negotiation tactics
- **Focus**: Termination clauses, renewal terms, industry benchmarks, vendor flexibility
- **Style**: Market-oriented. "Compared to industry benchmarks..." Uses data from comparable deals. Thinks in terms of leverage and BATNA.
- **Reading Pattern**: Reads termination, renewal, and payment sections. Benchmarks each term against comparable contracts. Defers to others on deep technical and legal analysis.
- **Knowledge Gaps**: Doesn't assess legal risk; doesn't evaluate technical feasibility; may accept unfavorable terms if they're "market standard"
- **Corrections**: Accepts readily when shown data that contradicts benchmarks. Pushes back when others suggest terms outside market norms.
- **Example**: *"Industry benchmark data shows 60% of comparable contracts use NET-30. The vendor won't budge on this."*

#### Injected Conflicts

| ID | Users | Topic | Nature | Resolution |
|----|-------|-------|--------|------------|
| C3-1 | Legal vs Engineering | IP clause (Section 4.2(a)) risk | Legal: deal-breaker. Engineering: standard for managed services. | Legal owns legal risk; Engineering owns technical feasibility |
| C3-2 | Finance vs Procurement | Payment terms (NET-30 vs NET-60) | Finance: NET-60 for cash flow. Procurement: vendor always insists on NET-30. | Procurement owns vendor negotiability; Finance owns internal cost impact |
| C3-3 | Legal vs Engineering | SLA penalty structure | Legal: penalties too low, push higher. Engineering: penalties are market-appropriate. | Cross-functional: Legal on contractual risk, Engineering on market feasibility |

---

### Scenario 4: Negotiation (Adversarial)

| | |
|---|---|
| **Relationship** | Adversarial — two opposing sides with internal hierarchies |
| **Domain** | Business / Finance, M&A Negotiation |
| **Timeline** | 4 weeks (negotiation rounds, buyer and seller timelines interleaved) |
| **Documents** | Asset Purchase Agreement (~8K), Valuation Report (~7K), Comparable Transactions (~5K) — total ~20K tokens |
| **Resolution Rule** | Preserve both sides completely — conflict IS the data. **Never leak one side's strategy to the other. Never average or merge opposing positions.** |
| **Primary Eval Stress** | Must preserve both sides; never merge opposing positions; adversarial isolation; dual hierarchies |
| **Applicable Categories** | All 8 universal + Authority & Hierarchy Resolution + Adversarial Side Isolation |

#### Authority Structure

```
BUYER SIDE                          SELLER SIDE
─────────────                       ─────────────
Buyer's CFO (HIGH, 3.0)            Seller's CEO (HIGH, 3.0)
    │                                   │
Buyer's Counsel (MEDIUM, 2.0)      Seller's Counsel (MEDIUM, 2.0)
```

No one on the buyer side outranks anyone on the seller side — they are adversarial peers across the divide.

#### Users

**Buyer's Counsel** (side: BUYER, authority: MEDIUM, weight: 2.0)
- **Expertise**: Buy-side M&A, warranty negotiation, liability caps, due diligence
- **Focus**: Warranty scope and duration, indemnification basket, rep & warranty insurance
- **Style**: Cautious, thorough, adversarial-by-training. "We need to protect against...", "in our experience, sellers tend to...". Marks up every ambiguous phrase.
- **Reading Pattern**: Reads APA warranty and indemnification first. Cross-references against comparable deal terms. Reads valuation report for risk factors that should be warrantied.
- **Corrections**: Accepts data corrections readily. On strategy: firm but willing to trade ("We can concede on X if we get Y").
- **Example**: *"24 months is the standard warranty period for deals of this size. We cite Deal #5 as precedent."*

**Buyer's CFO** (side: BUYER, authority: HIGH, weight: 3.0)
- **Expertise**: Valuation, financial modeling, deal structuring, risk assessment
- **Focus**: Purchase price, earn-out structure, escrow, comparable transaction analysis
- **Style**: Analytical and assertive. Builds arguments with data ("The DCF midpoint is X, and size-matched comps average Y"). Thinks in IRR and downside protection.
- **Reading Pattern**: Reads valuation report first, then comps table. Reads APA for price-related terms. Ignores legal details.
- **Biases**: Anchors low on valuation. Cherry-picks lower comps (Deals #3, #5, #7). Emphasizes market risk.
- **BATNA**: Organic build at $18M over 3 years caps willingness to pay. Fallback price: 5.8x.
- **Example**: *"The DCF midpoint is 5.8x, and applying a 10% market-risk discount brings us to 5.2x. That's our opening."*

**Seller's Counsel** (side: SELLER, authority: MEDIUM, weight: 2.0)
- **Expertise**: Sell-side M&A, representation limits, indemnification caps, post-closing exposure
- **Focus**: Rep & warranty limits, indemnification caps, survival periods, seller exposure
- **Style**: Strategic, measured, focused on limiting exposure. "Market trend is toward...", "seller's standard position is...". Frames concessions as generous.
- **Reading Pattern**: Reads APA for seller exposure (warranties, indemnification, escrow holdback). Assesses worst-case financial exposure. Reads comps for precedent on deal terms.
- **Corrections**: Accepts factual corrections. On strategy: firm but tactical — willing to concede minor points to hold major positions.
- **Example**: *"12 months is the market trend for warranty periods. 24 months creates excessive post-closing exposure."*

**Seller's CEO** (side: SELLER, authority: HIGH, weight: 3.0)
- **Expertise**: Strategic valuation, growth narrative, business vision, deal positioning
- **Focus**: Purchase price justification, growth trajectory narrative, strategic premium, transition terms
- **Style**: Visionary and assertive. Tells a growth story ("Our trajectory from 12% to 18% to projected 25%..."). "Market entry value", "build-vs-buy premium". Passionate — personal stakes.
- **Reading Pattern**: Reads valuation report for high-end scenarios. Reads comps for premium deals. Delegates legal details to Counsel.
- **Biases**: Anchors high. Cherry-picks premium comps (Deals #1, #9). Over-weights growth trajectory. Frames everything as strategic value.
- **Misconceptions**: Believes growth trajectory alone justifies a 7.1x premium despite market conditions
- **BATNA**: IPO in 18 months. Bankers say listing is viable.
- **Example**: *"The buyer gets strategic entry into our vertical. Building organically would take 3-5 years. That's worth a premium."*

#### Injected Conflicts (7 total — most conflict-dense scenario)

| ID | Users | Topic | Nature | Resolution |
|----|-------|-------|--------|------------|
| C4-1 | Buyer CFO vs Seller CEO | Purchase price (5.2x vs 7.1x) | Buyer: 5.2x based on size-matched comps. Seller: 7.1x based on growth premium. | Preserve both |
| C4-2 | Buyer Counsel vs Seller Counsel | Warranty period (24 vs 12 months) | Buyer: 24 months standard. Seller: 12 months market trend. | Preserve both |
| C4-3 | Buyer Counsel vs Seller Counsel | Indemnification basket ($3M vs $5M) | Buyer: $3M for meaningful remedy. Seller: $5M to avoid nuisance claims. | Preserve both |
| C4-4 | Buyer CFO vs Seller CEO | Comparable transaction selection | Buyer: Deals #3,5,7 (lower multiples). Seller: Deals #1,9 (premium multiples). | Preserve both |
| C4-5 | Buyer CFO vs Seller CEO | Earn-out structure | Buyer: aggressive earn-out tied to growth. Seller: minimal earn-out, more upfront. | Preserve both |
| C4-6 | Buyer Counsel vs Seller Counsel | Knowledge qualifier in representations | Buyer: remove "knowledge of" for broader coverage. Seller: keep to limit exposure. | Preserve both |
| C4-7 | Buyer CFO vs Seller CEO | BATNA comparison | Buyer: organic build $18M/3yr. Seller: IPO in 18 months. | Preserve both |

#### Adversarial Side Isolation Rules

The memory system must maintain strict information firewalls:
- A briefing for any buyer user must NEVER contain seller-side strategy or BATNA analysis
- A briefing for any seller user must NEVER contain buyer-side fallback positions
- Opposing positions must be preserved without averaging (e.g., do NOT suggest a midpoint between 5.2x and 7.1x)
- Cross-side reasoning is allowed (e.g., "what is the gap?") but must present both sides' positions faithfully

---

### Scenario 5: Support Escalation (Sequential)

| | |
|---|---|
| **Relationship** | Sequential — temporal handoffs between agents |
| **Domain** | Technical Support, Multi-Agent Incident Resolution |
| **Timeline** | 3 days (multi-day incident with hour-level timestamps) |
| **Documents** | Customer ticket (~3K), product documentation (~5K), system error logs (~4K) — total ~12K tokens |
| **Resolution Rule** | Later findings supersede earlier ones, but full audit trail (including the wrong diagnosis) must be preserved |
| **Primary Eval Stress** | Temporal ordering; later overrides earlier; audit trail; handoff gaps; customer communication accuracy |
| **Applicable Categories** | All 8 universal + Temporal Correction Chain + Sequential Handoff Continuity |

#### Handoff Sequence

```
Agent A (Tier 1, LOW)  →  Agent B (Tier 2, MEDIUM)  →  Agent C (Resolution, MEDIUM)
   Triages ticket            Deep log analysis             Implements fix
   Diagnoses DNS (WRONG)     Finds connection pool          Corrects customer
   Applies KB-2847 fix       exhaustion (CORRECT)           communication
   Tells customer "DNS"      Finds docs error               Closes incident
   Fix fails → escalates     Documents for C
```

#### Users

**Agent A — Tier 1 Triage** (sequence: 1st, authority: LOW, weight: 1.0)
- **Expertise**: Tier 1 support, standard playbook execution, initial triage
- **Focus**: Problem identification, standard troubleshooting, customer communication, escalation
- **Style**: Structured, playbook-oriented. "Initial triage", "applying KB-2847 remediation", "escalating to Tier 2." Follows a checklist mentality. Reports in numbered steps.
- **Reading Pattern**: Reads error logs first — matches symptoms to KB articles. Reads product docs for the matching remediation steps. Stops investigating once a plausible match is found.
- **Knowledge Gaps**: Can't do deep log analysis; doesn't understand database connection pooling; follows playbook but can't reason beyond it
- **Misconceptions**: DNS timeout errors = DNS caching issue (actually coincidental with a scheduled refresh)
- **Corrections**: Accepts escalation feedback professionally. Doesn't take it personally when diagnosis is overridden.
- **Critical Detail**: Told the customer it was DNS caching. If root cause is different, the customer needs an update.
- **Example**: *"KB-2847 matches — DNS caching issue. Applying the standard fix: flush cache, update TTL, verify with dig."*

**Agent B — Tier 2 Deep Analysis** (sequence: 2nd, authority: MEDIUM, weight: 2.0)
- **Expertise**: Deep log analysis, database systems, API troubleshooting, root cause analysis
- **Focus**: Full log analysis, connection pool diagnostics, root cause identification, documentation accuracy
- **Style**: Analytical and detailed. Walks through findings step by step with evidence. "Connection pool exhaustion", "max pool size exceeded", "downstream cascade." Explicitly flags when a prior diagnosis was wrong.
- **Reading Pattern**: Reads full 48-hour log set chronologically. Cross-references against product docs. Notices patterns that span timestamps. Checks Agent A's assumptions, not just conclusions.
- **Knowledge Gaps**: Less focused on customer communication; may produce findings that are hard to explain to customers
- **Key Discovery**: The product docs say the parameter is `max_pool_wait`, but the config dump shows `pool_max_wait` — the docs have it wrong.
- **Example**: *"Agent A diagnosed DNS caching, but the fix didn't work. I'm looking at the full 48-hour log set. Starting around 6 AM, there's a clear pattern of 'max pool size exceeded' errors. The DNS errors are a red herring."*

**Agent C — Resolution & Communication** (sequence: 3rd, authority: MEDIUM, weight: 2.0)
- **Expertise**: Fix implementation, customer communication, incident resolution, diplomacy
- **Focus**: Fix implementation, customer update drafting, incident closure, corrective communication
- **Style**: Diplomatic and customer-focused. "After deeper analysis", "our Tier 2 team identified", "we apologize for the initial misdirection." Balances technical accuracy with customer empathy.
- **Reading Pattern**: Reads Agent B's findings first, then Agent A's escalation notes, then full ticket history to understand what the customer was told.
- **Knowledge Gaps**: Wasn't involved in initial diagnosis; relies entirely on Agent B's findings
- **Key Challenge**: The customer was told one thing (DNS) and now needs to hear another (connection pool exhaustion). Must correct without undermining trust.
- **Example**: *"The customer was told it was DNS caching. Now we're saying it's connection pool exhaustion. This is delicate."*

#### Injected Conflicts

| ID | Users | Topic | Nature | Resolution |
|----|-------|-------|--------|------------|
| C5-1 | Agent A vs Agent B | Root cause (DNS vs connection pool) | A: DNS caching issue. B: Connection pool exhaustion. | Temporal supersession — B's later, evidence-based diagnosis overrides |
| C5-2 | Agent B (docs vs config) | Parameter name (`max_pool_wait` vs `pool_max_wait`) | Product docs are wrong. Config is authoritative. | Factual supersession — config overrides documentation |

#### Correction Chains Tested

- Agent A diagnoses DNS caching → Applies KB-2847 fix → Fix fails → Agent B finds connection pool exhaustion → Agent C applies correct fix → Customer was told wrong diagnosis → **Is the correction communicated to the customer?**
- Product docs say `max_pool_wait` → Config shows `pool_max_wait` → Agent B flags documentation error → **Is the doc correction filed?**
- Full audit trail: initial diagnosis → escalation → correct diagnosis → fix → customer update → **Can the system reconstruct the complete chain?**

---

## 4-Phase Generation Pipeline

```
Phase 1: Document Preparation → Load PDFs, extract text, validate token counts
                                         ↓
Phase 2: Conversation Generation → Per-user, per-session with summary feed-forward
                                         ↓
Phase 3: Annotation → Memory extraction, conflict detection, eval questions
                                         ↓
Phase 4: Validation → 5-point quality check with LLM-as-judge
```

### Phase 1: Document Preparation

- Loads all PDFs from `documents/scenario_N/` using PyMuPDF (fitz)
- Extracts text and counts tokens with tiktoken (cl100k_base encoding)
- Validates token counts are within ±20% of scenario targets
- Builds a concatenated context block with document markers
- **Output**: `DocumentContext` with documents dict, token counts, and context block

### Phase 2: Conversation Generation

For each user in the scenario, for each of 7 sessions:

1. Builds a system prompt containing:
   - Full user persona (10+ roleplay fields)
   - Authority context for the relationship type
   - Complete document context
   - Session-specific behavior (from `session_evolution`)
   - Prior sessions' summaries (feed-forward continuity)
   - Seeded conflicts (emerge naturally from persona, not forced)
2. Sends system prompt + trigger to LLM (temperature: 0.7)
3. LLM generates exactly N turns (user + assistant pairs) in JSON format
4. Generates a `SessionSummary` (150-250 words) immediately after each session
5. Accumulates summaries for feed-forward into subsequent sessions

**Output**: `list[ConversationSession]`, `list[SessionSummary]`

### Phase 3: Annotation

Three sub-phases in sequence:

**3a. Memory Extraction** (per user, per session):
- Extracts atomic, durable memories from the USER's statements only
- Tags each with `memory_type`, `status`, `authority_level`, domain/side tags
- Detects supersession: if a memory contradicts an earlier one, marks the old one "superseded"
- Target: ~3.5 memories per session

**3b. Conflict Detection** (cross-user):
- Takes ALL extracted memories across all users
- Identifies both pre-specified conflicts (from scenario config) and organic ones
- For each conflict: determines `conflict_type`, `resolution`, maps evidence to sessions

**3c. Evaluation Question Generation** (per category):
- Generates target number of questions per applicable evaluation category
- Each question includes: gold answer, session-level evidence links, required memory/conflict IDs, difficulty

**Output**: `list[ExtractedMemory]`, `list[ConflictAnnotation]`, `list[EvalQuestion]`

Temperature for all annotation: 0.2 (consistent, precise extraction)

### Phase 4: Validation

Five quality checks, each with pass/fail thresholds:

| Check | Description | Threshold |
|-------|-------------|-----------|
| **Conflict Coverage** | % of injected conflicts detected in annotations | ≥ 80% |
| **Memory Extractability** | Can an LLM extract similar memories from conversations? (sampled) | LLM-as-judge |
| **Evidence Validity** | Are all evidence links valid (user/session pairs exist)? | Structural check |
| **Question Answerability** | Can a model answer eval questions using the benchmark data? (sampled) | LLM-as-judge |
| **Persona Fidelity** | Do conversations reflect user personas? | Avg score ≥ 3.5/5 |

**Overall Pass**: All 5 checks pass their thresholds.

---

## Evaluation Framework

### Implementing a Memory System

Implement the `MemorySystemInterface` abstract class:

```python
from eval.run_eval import MemorySystemInterface

class MyMemorySystem(MemorySystemInterface):
    def process_conversations(self, conversations, documents):
        """Build your internal memory representation."""
        # conversations: list[ConversationSession]
        # documents: dict of document texts
        pass

    def answer_question(self, question):
        """Answer an evaluation question using your memory system."""
        # question: EvalQuestion
        return "your answer"
```

### Running Evaluation

```bash
# Run built-in baselines
python -m eval.run_eval --scenario 1 --baseline full_context
python -m eval.run_eval --scenario 1 --baseline rag_summaries

# Run your custom system
python -m eval.run_eval --scenario 1 --system my_system
```

### Built-in Baselines

**Full Context Baseline** (`eval/baselines/full_context.py`):
- Concatenates all conversations + documents into a single context
- Asks the LLM to reason over the full context to answer questions
- Tests whether raw context is sufficient (no memory extraction needed)

**RAG Summaries Baseline** (`eval/baselines/rag_summaries.py`):
- Builds session summaries from conversations
- Embeds summaries with OpenAI's `text-embedding-3-small`
- Retrieves top-k summaries via cosine similarity for each question
- Tests whether retrieval over summaries helps compared to full context

### Scoring Metrics

| Metric | Function | Used For |
|--------|----------|----------|
| Exact Match | `exact_match()` | User Attribution, Adversarial Confusion |
| Contains Match | `exact_match_contains()` | Binary checks |
| ROUGE | `compute_rouge()` | ROUGE-1, ROUGE-2, ROUGE-L F-measure |
| F1 (Set) | `compute_f1()` | Document Coverage, Conflict Detection |
| LLM-as-Judge | `llm_judge_score()` | Most open-ended categories (0.0-1.0 with reasoning) |
| FactScore | `factscore()` | Cross-User Synthesis |

---

## 12 Evaluation Categories

Every evaluation task requires reasoning about **who said what about which document, when, and with what authority**.

### Category Applicability

| # | Category | Sc.1 | Sc.2 | Sc.3 | Sc.4 | Sc.5 | Scope |
|---|----------|:----:|:----:|:----:|:----:|:----:|-------|
| 1 | User Attribution QA | x | x | x | x | x | Universal |
| 2 | Cross-User Synthesis | x | x | x | x | x | Universal |
| 3 | Conflict Detection & Resolution | x | x | x | x | x | Universal |
| 4 | Information Gap Detection | x | x | x | x | x | Universal |
| 5 | Role-Appropriate Briefing | x | x | x | x | x | Universal |
| 6 | Adversarial User Confusion | x | x | x | x | x | Universal |
| 7 | Multi-User Document Coverage | x | x | x | x | x | Universal |
| 8 | Cross-User Provenance Tracking | x | x | x | x | x | Universal |
| 9 | Authority & Hierarchy Resolution | - | x | x | x | - | Cluster |
| 10 | Temporal Correction Chain | x | x | - | - | x | Cluster |
| 11 | Adversarial Side Isolation | - | - | - | x | - | Specific |
| 12 | Sequential Handoff Continuity | - | - | - | - | x | Specific |

### Universal Categories (All 5 Scenarios)

**1. User Attribution QA** — "Who said/flagged/proposed X?" Track which user stated specific information. Metric: exact-match accuracy.

**2. Cross-User Synthesis** — "Combining all users' analyses, what is the complete picture?" Merge complementary knowledge from users who each examined different document aspects. Metric: FactScore (P/R/F1).

**3. Conflict Detection & Resolution** — Identify disagreements between users and apply the correct resolution strategy for the scenario's relationship type (symmetric = preserve both; hierarchical = authority wins on opinions; cross-functional = domain expert wins; adversarial = preserve both as-is; sequential = later supersedes). Metric: P/R/F1 + LLM-as-judge.

**4. Information Gap Detection** — "What has User A discovered that User B needs?" Identify cross-user information asymmetry. Metric: LLM-as-judge.

**5. Role-Appropriate Briefing** — "Generate a briefing for User X." Tailor summaries to a user's role, authority level, and information needs, incorporating all other users' findings. Metric: LLM-as-judge.

**6. Adversarial User Confusion** — Deliberately swap users/positions to test resistance to misattribution. Questions are designed to confuse which user said what. Expected answer is always "No" with a correction. Metric: binary accuracy.

**7. Multi-User Document Coverage** — "Which sections reviewed by which users? What was missed?" Track collective coverage and identify blind spots. Metric: P/R/F1.

**8. Cross-User Provenance Tracking** — "Trace the history of [fact/decision/error] across all users and sessions." Track how information was discovered, evolved, and propagated. Metric: LLM-as-judge.

### Scenario-Cluster Categories

**9. Authority & Hierarchy Resolution** (Scenarios 2, 3, 4) — Test whether the system understands that different users carry different weight depending on relationship type and topic. Hierarchical: VP overrides on narrative, but facts are flat. Cross-functional: domain expert wins on their domain. Adversarial: dual hierarchies within each side, no cross-side authority. Metric: LLM-as-judge.

**10. Temporal Correction Chain** (Scenarios 1, 2, 5) — Track the full lifecycle of a correction: original wrong claim, who made it, when corrected, by whom, whether accepted, and current state. Examples: BFT formula correction (Sc.1), $42M revenue error (Sc.2), DNS misdiagnosis chain (Sc.5). Metric: LLM-as-judge.

### Scenario-Specific Categories

**11. Adversarial Side Isolation** (Scenario 4 only) — Maintain information firewalls between buyer and seller teams. Sub-tasks: side leakage detection (40 Qs), position preservation (30 Qs), cross-side reasoning (30 Qs). Metric: LLM-as-judge + binary leakage detection.

**12. Sequential Handoff Continuity** (Scenario 5 only) — Track what each agent knew at handoff time, what was communicated to the customer, and whether corrections propagated. Sub-tasks: handoff completeness (35 Qs), customer communication audit (35 Qs), diagnosis chain accuracy (30 Qs). Metric: LLM-as-judge.

### Question Distribution Per Scenario

| Category | Sc.1 | Sc.2 | Sc.3 | Sc.4 | Sc.5 | Total |
|----------|------|------|------|------|------|-------|
| 1. User Attribution | 20 | 20 | 20 | 25 | 15 | 100 |
| 2. Cross-User Synthesis | 20 | 20 | 25 | 20 | 15 | 100 |
| 3. Conflict Resolution | 15 | 15 | 15 | 40 | 15 | 100 |
| 4. Information Gap | 20 | 20 | 20 | 15 | 25 | 100 |
| 5. Role-Appropriate Briefing | 15 | 25 | 25 | 25 | 10 | 100 |
| 6. Adversarial Confusion | 20 | 20 | 20 | 25 | 15 | 100 |
| 7. Document Coverage | 20 | 15 | 25 | 25 | 15 | 100 |
| 8. Provenance Tracking | 15 | 20 | 15 | 15 | 35 | 100 |
| 9. Authority & Hierarchy | - | 40 | 35 | 25 | - | 100 |
| 10. Temporal Correction | 30 | 40 | - | - | 30 | 100 |
| 11. Side Isolation | - | - | - | 100 | - | 100 |
| 12. Handoff Continuity | - | - | - | - | 100 | 100 |
| **Total** | **175** | **235** | **200** | **315** | **275** | **1,200** |

---

## Data Schemas

All data structures are Pydantic models defined in `src/models/`.

### Enums (`src/models/enums.py`)

| Enum | Values |
|------|--------|
| `RelationshipType` | symmetric, hierarchical, cross_functional, adversarial, sequential |
| `AuthorityLevel` | low, medium, high, equal |
| `ConflictType` | factual, interpretive, strategic, authority |
| `ConflictResolution` | preserve_both, supersede, authority_wins, temporal_supersession |
| `MemoryType` | fact, opinion, strategy, directive, risk_flag, assessment, position, diagnosis, customer_communication |
| `MemoryStatus` | active, superseded, corrected |
| `EvalQuestionCategory` | 12 categories (see above) |

### Core Models (`src/models/schemas.py`)

**`UserProfile`** — User persona with 10+ roleplay fields:
- Identity: `user_id`, `display_name`, `authority_level`, `authority_weight`
- Expertise: `expertise`, `focus_areas`, `biases`, `knowledge_gaps`, `misconceptions`
- Behavior: `communication_style`, `document_reading_pattern`, `reaction_to_corrections`, `emotional_tendencies`, `reference_style`
- Evolution: `session_evolution` (dict mapping session to behavior), `example_utterances`
- Optional: `domain_authority`, `side` (adversarial), `sequence_order` (sequential)

**`ConversationSession`** — One user's single session with the AI assistant. Contains `session_id`, `user_id`, `session_number`, `session_timestamp`, `turns: list[ConversationTurn]`, `target_conflicts`.

**`ConversationTurn`** — `turn_number`, `role` (user/assistant), `content`, `timestamp`.

**`SessionSummary`** — 150-250 word session summary with `key_facts`, `positions_taken`, `positions_changed`.

**`ExtractedMemory`** — Atomic, durable memory statement. Session-level attribution (not turn-level). Fields: `memory_id`, `memory_type`, `content`, `status` (active/superseded/corrected), `superseded_by`, `authority_level`, `domain_tag`, `side_tag`.

**`EvidenceLink`** — Session-level reference: `(user_id, session_id)`. No turn-level granularity.

**`ConflictAnnotation`** — Cross-user disagreement. Fields: `conflict_id`, `users_involved`, `topic`, `conflict_type`, `positions` (dict per user), `resolution`, `resolution_detail`, `evidence: list[EvidenceLink]`, `first_surfaced`, `last_updated`.

**`EvalQuestion`** — Evaluation question with `gold_answer`, `evidence: list[EvidenceLink]`, `required_memories`, `required_conflicts`, `category`, `difficulty` (easy/medium/hard).

**`ScenarioConfig`** — Full scenario definition: users, documents, timeline (with per-user `session_schedule`), `injected_conflicts`, `annotation_targets` (with `eval_breakdown`).

**`BenchmarkDataset`** — Top-level container: `version`, `generated_at`, `scenarios: list[ScenarioOutput]`, `aggregate_stats`, `generation_report`.

---

## Configuration

### Model Configuration (`config/models.yaml`)

```yaml
iteration:           # Early testing (cheaper)
  conversation: gpt-4o-mini
  summary: gpt-4o-mini
  annotation: gpt-4o-mini

final:               # Production runs
  conversation: gpt-4.1
  summary: gpt-4o-mini
  annotation: gpt-4o

eval:                # Evaluation scoring
  judge: gpt-4o
  baseline: gpt-4o
```

### Generation Parameters (`config/generation.yaml`)

```yaml
defaults:
  sessions_per_user: 7
  temperature_conversation: 0.7
  temperature_annotation: 0.2
  max_retries: 3
  rate_limit_rpm: 500
  summary_min_words: 150
  summary_max_words: 250
  memories_per_session: 3.5
  response_format: json_object
  max_tokens_conversation: 8192
  max_tokens_annotation: 4096
  max_tokens_summary: 2048
```

### Scenario Configuration (`config/scenarios/scenario_N.yaml`)

Each scenario YAML defines:

```yaml
scenario_id: "1"
name: "Study Group: Distributed Systems Exam"
relationship_type: symmetric
domain: "Education / CS"
sessions_per_user: 7
turns_per_session: 18

timeline:
  start_date: "2025-04-01T09:00:00Z"
  end_date: "2025-04-14T22:00:00Z"
  session_schedule:
    student_a:
      - {session: 1, date: "2025-04-01"}
      # ...

users:
  - user_id: student_a
    display_name: "Student A"
    authority_level: equal
    authority_weight: 1.0
    expertise: "Strong on theory, weak on implementation"
    focus_areas: ["Clock algorithms", "ordering proofs"]
    biases: ["Prefers mathematical notation over pseudocode"]
    communication_style: "Formal and precise..."
    document_reading_pattern: "Deep reader of theoretical sections..."
    reaction_to_corrections: "Gracefully accepts corrections on theory..."
    knowledge_gaps: ["Cannot read pseudocode fluently"]
    misconceptions: ["Believes vector clocks are 30% of exam"]
    emotional_tendencies: "Confident but anxious about implementation..."
    reference_style: "Formal citations by number..."
    session_evolution:
      session_1: "Tentative, asks basic questions"
      session_2: "More confident, challenges explanations"
      # ...
    example_utterances:
      - "Can you walk me through the proof on page 42?"

documents:
  - name: "Textbook Chapter"
    filename: "textbook_chapter.pdf"
    target_tokens: 8000

injected_conflicts:
  - conflict_id: "C1-1"
    users: ["student_b", "student_d"]
    topic: "Paxos vs Raft Algorithm"
    nature: "B prefers Raft, D prefers Paxos"
    resolution: "Preserve both as peer preferences"
    target_sessions: [3, 4, 5]

annotation_targets:
  memories_per_session: 3.5
  conflicts: 8
  eval_questions: 100
  eval_breakdown:
    user_attribution: 12
    cross_user_synthesis: 12
    # ...
```

---

## CLI Reference

### Full Pipeline

```bash
python -m src.cli generate --scenario 1                 # Single scenario
python -m src.cli generate --all                        # All 5 scenarios
python -m src.cli generate --scenario 1 --dry-run       # Test with mock LLM
python -m src.cli generate --scenario 1 --model gpt-4o  # Override model
```

### Individual Phases

```bash
python -m src.cli generate-conversations --scenario 1 --model gpt-4o-mini
python -m src.cli annotate --scenario 1 --model gpt-4o
python -m src.cli validate --scenario 1 --model gpt-4o
```

### Utilities

```bash
python -m src.cli validate-docs --scenario 1    # Check PDFs exist, count tokens
python -m src.cli validate-docs --all           # All scenarios
python -m src.cli estimate-cost --all            # Pre-run cost estimation
python -m src.cli export --format huggingface    # Export to HuggingFace datasets
```

### Makefile Shortcuts

```bash
make install                         # pip install -e ".[dev]"
make test                            # pytest tests/ -v
make lint                            # ruff check src/ eval/ tests/
make generate-all                    # Generate all scenarios
make generate-scenario SCENARIO=1    # Generate single scenario
make validate SCENARIO=1             # Validate single scenario
make validate-docs SCENARIO=1        # Validate documents
make eval SCENARIO=1 BASELINE=full_context
make cost-estimate                   # Estimate costs
make dry-run                         # Dry run scenario 1
make export                          # Export to HuggingFace
make clean                           # Remove output/ and __pycache__
```

---

## Prompt Design

### Design Principles

1. **Persona-driven generation**: 10+ roleplay fields per user fully define voice, reading patterns, emotional state, and session-by-session evolution
2. **Session-level feed-forward**: Session summaries accumulate and feed into subsequent session prompts for continuity
3. **Natural conflict emergence**: Conflicts are seeded via persona differences, not forced — the prompt describes expected conflicts but instructs the LLM to let them arise naturally from character
4. **Structured JSON output**: All LLM calls use `response_format: {"type": "json_object"}` for reliable parsing

### Temperature Settings

| Phase | Temperature | Rationale |
|-------|------------|-----------|
| Conversations | 0.7 | Creative, varied output |
| Annotations | 0.2 | Consistent, precise extraction |
| Validation/Scoring | 0.0 | Deterministic judgment |

### Prompt Modules

| Module | Purpose |
|--------|---------|
| `conversation_system.py` | Full system prompt: persona, documents, session behavior, conflict seeding, continuity rules |
| `conversation_user.py` | Minimal trigger: "Generate Session X now with exactly Y turns" |
| `session_summary.py` | Summary generation: 150-250 words, key facts, positions taken/changed |
| `memory_extraction.py` | Memory extraction: atomic statements, type tagging, supersession detection |
| `conflict_detection.py` | Cross-user conflict identification, typing, and resolution mapping |
| `eval_question_gen.py` | Per-category question generation with category-specific instructions |
| `quality_check.py` | Validation prompts for LLM-as-judge assessments |

---

## Adding a New Scenario

1. Create `config/scenarios/scenario_N.yaml` following the existing pattern
2. Define all users with the full set of roleplay fields (see existing YAMLs for reference)
3. Define `injected_conflicts` with target sessions where they should surface
4. Set `annotation_targets` with eval question breakdown per category
5. Create a scenario class in `src/scenarios/scenario_N.py` extending `BaseScenario`
6. Register it in `src/scenarios/__init__.py`

### Source Documents

Each scenario requires 3 PDFs placed in `documents/scenario_{id}/`. Token counts should be within ±20% of targets defined in the scenario YAML.

| Scenario | Documents | Target Tokens |
|----------|-----------|--------------|
| 1 | textbook_chapter.pdf, lecture_slides.pdf, past_exam.pdf | ~15K |
| 2 | quarterly_report.pdf, data_appendix.pdf, board_presentation.pdf | ~22K |
| 3 | master_services_agreement.pdf, statement_of_work.pdf, pricing_schedule.pdf | ~18K |
| 4 | asset_purchase_agreement.pdf, valuation_report.pdf, comparable_transactions.pdf | ~20K |
| 5 | customer_ticket.pdf, product_documentation.pdf, system_error_logs.pdf | ~12K |

---

## Testing

```bash
# Run all tests
make test
# or
pytest tests/ -v

# Individual test files
pytest tests/test_schemas.py -v     # Enum and model validation
pytest tests/test_prompts.py -v     # Prompt generation correctness
pytest tests/test_scenarios.py -v   # Scenario loading and factory methods
pytest tests/test_pipeline.py -v    # Pipeline integration (orchestrator)
```

---

## Design Principles

1. **Session-Level Attribution**: All evidence, memories, and questions reference `(user_id, session_id)` — not turn-level granularity. This keeps the problem tractable while still challenging.

2. **Persona-Driven Generation**: 10+ roleplay fields per user fully define voice, reading patterns, and evolution across 7 sessions. The prompt system ensures consistency across sessions.

3. **Feed-Forward Continuity**: Session summaries accumulate and feed into the next session's prompt. Positions evolve; users revisit and change minds; corrections happen naturally.

4. **Natural Conflict Emergence**: Conflicts are seeded via persona biases and injected conflict definitions, but they emerge naturally in conversation rather than being scripted.

5. **Authority-Aware Resolution**: Different relationship types have different conflict resolution rules — symmetric (preserve both), hierarchical (authority wins on opinions, facts are flat), cross-functional (domain expert wins), adversarial (strict side isolation), sequential (later supersedes with audit trail).

6. **Structured JSON Output**: All LLM calls use `response_format: {"type": "json_object"}` with explicit output schemas in prompts for reliable parsing.

7. **Cost Transparency**: Every API call is tracked (model, input tokens, output tokens, phase, cost). Aggregated reports per phase and per scenario in `generation_report.json`.

8. **Validation as First-Class**: Phase 4 validation is not optional — quality metrics must pass or the scenario is marked FAIL.

---

## Dependencies

### Runtime
- `openai >= 1.30.0` — API client
- `pydantic >= 2.0` — Data validation
- `tiktoken >= 0.7.0` — Token counting
- `pymupdf >= 1.24.0` — PDF text extraction
- `pyyaml >= 6.0` — Config parsing
- `click >= 8.0` — CLI framework
- `rich >= 13.0` — Terminal formatting
- `python-dotenv >= 1.0` — Environment variables
- `tenacity >= 8.0` — Retry logic with exponential backoff
- `rouge-score >= 0.1.2` — ROUGE metrics
- `numpy >= 1.24.0` — Numeric operations

### Development
- `pytest >= 7.0`
- `pytest-cov >= 4.0`
- `ruff >= 0.4.0` — Linting

### Optional
- `datasets >= 2.0` — HuggingFace export (`pip install -e ".[export]"`)

---

## License

MIT
