# MUM Benchmark — Implementation Plan for Coding Agent (v2)

## Executive Summary

Build a production-quality, open-source benchmark generation pipeline for **MUM** (Multi-User Memory). The benchmark extends the LoCoMo paradigm (long multi-session conversations) into a **multi-user setting**: multiple users independently interact with an AI assistant over shared documents, and the benchmark evaluates whether a system can extract, manage, and merge knowledge across these parallel conversation streams.

**What makes this different from LoCoMo**: LoCoMo tests single-thread memory (2 speakers, 1 conversation, ~300 turns, ~19 sessions). MUM tests **cross-user** memory: 3–4 users per scenario, each with their own long conversation history, producing conflicting facts, authority gradients, and temporal dependencies that a memory system must reconcile.

**Key numbers**:

| Metric | Value |
|--------|-------|
| Scenarios | 5 (one per relationship type) |
| Source documents | 15 PDFs (~87K tokens total) |
| Unique users | 18 (3–4 per scenario) |
| Sessions per user | 7 |
| Turns per session | 15–20 |
| Total sessions | ~133 |
| Total turns | ~2,340 |
| Avg. tokens per user conversation | ~12K+ |
| Session summaries | ~133 |
| Extracted memories | ~465+ (~3.5 per session) |
| Conflict annotations | ~30+ |
| Evaluation categories | **12** (8 universal + 2 scenario-cluster + 2 scenario-specific) |
| Evaluation questions | **1,200** (100 per category) |

---

## 1. Repository Structure

```
mum/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── Makefile
│
├── config/
│   ├── models.yaml                    # Model configs: iteration vs final vs eval
│   ├── generation.yaml                # Global params: turns, sessions, retries, temperatures
│   └── scenarios/
│       ├── scenario_1.yaml
│       ├── scenario_2.yaml
│       ├── scenario_3.yaml
│       ├── scenario_4.yaml
│       └── scenario_5.yaml
│
├── documents/                         # User-provided source PDFs (gitignored)
│   ├── scenario_1/
│   │   ├── textbook_chapter.pdf
│   │   ├── lecture_slides.pdf
│   │   └── past_exam.pdf
│   ├── scenario_2/
│   │   ├── quarterly_report.pdf
│   │   ├── data_appendix.pdf
│   │   └── board_presentation.pdf
│   ├── scenario_3/
│   │   ├── master_services_agreement.pdf
│   │   ├── statement_of_work.pdf
│   │   └── pricing_schedule.pdf
│   ├── scenario_4/
│   │   ├── asset_purchase_agreement.pdf
│   │   ├── valuation_report.pdf
│   │   └── comparable_transactions.pdf
│   └── scenario_5/
│       ├── customer_ticket.pdf
│       ├── product_documentation.pdf
│       └── system_error_logs.pdf
│
├── src/
│   ├── __init__.py
│   ├── cli.py                         # CLI entrypoint (click or typer)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py            # Main pipeline: runs all phases in sequence
│   │   ├── phase1_document_prep.py    # Load & validate PDFs, extract text, count tokens
│   │   ├── phase2_conversation.py     # Generate conversations per user per session
│   │   ├── phase3_annotation.py       # Summaries, memories, conflicts, eval Qs
│   │   └── phase4_validation.py       # Multi-pass validation with LLM-as-judge
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Pydantic models for ALL data structures
│   │   └── enums.py                   # Enums: RelationshipType, ConflictType, etc.
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── base.py                    # Prompt builder utilities (template rendering)
│   │   ├── conversation_system.py     # System prompts for conversation generation
│   │   ├── conversation_user.py       # User-turn simulation prompts
│   │   ├── session_summary.py         # Session summary generation prompts
│   │   ├── memory_extraction.py       # Cross-session memory extraction prompts
│   │   ├── conflict_detection.py      # Cross-user conflict detection prompts
│   │   ├── eval_question_gen.py       # Evaluation question generation prompts
│   │   └── quality_check.py           # Validation pass prompts
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py                  # Unified LLM client (OpenAI, retries, rate limits)
│   │   ├── token_counter.py           # Token counting (tiktoken)
│   │   └── cost_tracker.py            # Per-call and cumulative cost tracking
│   │
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseScenario ABC with shared logic
│   │   ├── scenario_1.py             # Study Group: symmetric peers
│   │   ├── scenario_2.py             # Exec + Team: hierarchical
│   │   ├── scenario_3.py             # Contract Review: cross-functional
│   │   ├── scenario_4.py             # Negotiation: adversarial
│   │   └── scenario_5.py             # Support: sequential/temporal
│   │
│   └── utils/
│       ├── __init__.py
│       ├── pdf_reader.py              # PDF text extraction (PyMuPDF/fitz)
│       ├── io.py                      # Read/write JSON, YAML, JSONL
│       └── logging.py                 # Structured logging setup
│
├── output/                            # Generated benchmark data (gitignored)
│   ├── scenario_1/
│   │   ├── conversations/
│   │   │   ├── student_a_session_1.json
│   │   │   ├── student_a_session_2.json
│   │   │   └── ...
│   │   ├── summaries/
│   │   │   └── session_summaries.json
│   │   ├── memories/
│   │   │   └── extracted_memories.json
│   │   ├── conflicts/
│   │   │   └── conflict_annotations.json
│   │   ├── evaluation/
│   │   │   └── eval_questions.json
│   │   ├── validation/
│   │   │   └── validation_report.json
│   │   └── scenario_1_complete.json
│   ├── scenario_2/ ...
│   ├── ...
│   ├── benchmark.json                 # Final combined benchmark file
│   └── generation_report.json         # Costs, token usage, timing, quality metrics
│
├── eval/                              # Evaluation harness for testing memory systems
│   ├── __init__.py
│   ├── run_eval.py                    # Score a memory system against the benchmark
│   ├── metrics.py                     # F1, accuracy, LLM-as-judge scoring
│   └── baselines/
│       ├── full_context.py            # Baseline: feed all conversations as raw context
│       └── rag_summaries.py           # Baseline: RAG over session summaries
│
├── tests/
│   ├── test_schemas.py
│   ├── test_prompts.py
│   ├── test_pipeline.py
│   └── test_scenarios.py
│
├── scripts/
│   ├── validate_documents.py
│   ├── dry_run.py
│   ├── export_hf.py
│   └── cost_estimate.py
│
└── docs/
    ├── GENERATION_GUIDE.md
    ├── PROMPT_DESIGN.md
    ├── SCHEMA_REFERENCE.md
    └── EVALUATION_GUIDE.md            # How to evaluate a memory system using this benchmark
```

---

## 2. Conversation Design — LoCoMo-Grade Depth

### 2.1 Why More Sessions and Turns

LoCoMo averages ~19 sessions and ~300 turns per conversation. Our benchmark tests multi-user dynamics, so per-user length can be shorter than LoCoMo, but **must be long enough that memory is necessary** — the conversation should not fit trivially in a single context window. Each user should accumulate enough facts, opinions, and evolving positions that a memory system must actively extract, store, and retrieve.

### 2.2 Revised Session/Turn Targets

| Scenario | Users | Sessions/User | Turns/Session | Turns/User | Total Turns | Doc Tokens |
|----------|-------|---------------|---------------|------------|-------------|------------|
| 1 Study Group     | 4 | 7 | 18 | 126 | 504 | 15K |
| 2 Exec + Team     | 4 | 7 | 16 | 112 | 448 | 22K |
| 3 Contract Review | 4 | 7 | 18 | 126 | 504 | 18K |
| 4 Negotiation     | 4 | 7 | 18 | 126 | 504 | 20K |
| 5 Support         | 3 | 7 | 18 | 126 | 378 | 12K |
| **Total**         | **18** | | | | **~2,340** | **87K** |

This gives ~112–126 turns per user across 7 sessions, comparable to a meaningful LoCoMo-scale conversation. Each user's total conversation will be ~5K–8K tokens of generated dialogue, plus the document context re-injected each session.

### 2.3 Temporal Grounding

Every session has a simulated timestamp. Sessions are spread over a realistic timeline for the scenario.

**Scenario 1 — Study Group**: 7 sessions over 2 weeks (exam prep period). Session 1 = Day 1, Session 7 = Day 14.

**Scenario 2 — Exec + Team**: 7 sessions over 1.5 weeks (board prep). Session 1 = Monday morning, Session 7 = following Wednesday evening.

**Scenario 3 — Contract Review**: 7 sessions over 3 weeks (review cycle). Sessions clustered around key milestones.

**Scenario 4 — Negotiation**: 7 sessions over 4 weeks (negotiation rounds). Buyer and seller timelines interleaved.

**Scenario 5 — Support**: 7 sessions over 3 days (multi-day incident). Sessions have hour-level timestamps. Sequential handoffs between agents.

### 2.4 Session Arc Design

Each user follows a **7-session narrative**:

```
Session 1 — Orientation:     First read of documents. Surface-level questions.
Session 2 — Focused Dive:    Drill into primary area of expertise.
Session 3 — Cross-Topic:     Explore secondary topics. Early conflicts may surface.
Session 4 — Deep Analysis:   Core conflicts emerge. Strong opinions formed.
Session 5 — Revision:        User revisits earlier conclusions. Some positions evolve.
                              Corrections happen. Facts may be superseded.
Session 6 — Synthesis:       User consolidates understanding. Asks cross-cutting questions.
Session 7 — Final Review:    Last pass. Summary requests. Lingering doubts addressed.
```

This arc ensures: (a) facts build and evolve over time (testing temporal tracking), (b) users revisit and sometimes change positions (testing supersession/update), (c) conflicts intensify through middle sessions, and (d) final sessions produce synthesized views worth evaluating.

---

## 3. Annotation Layers

MUM produces **four annotation layers**, extending LoCoMo's annotation structure into the multi-user setting. **All annotations reference sessions, not individual turns.**

### Layer 1: Session Summaries (like LoCoMo's session summaries)

For each session, generate a 150–250 word summary capturing what the user discussed, concluded, and any positions taken or changed.

**Purpose**: Serves as a retrieval database for RAG-based evaluation. Also used as context input for subsequent session generation (continuity).

**Count**: ~133 summaries.

### Layer 2: Extracted Memories (multi-user specific)

Cross-session extraction of **durable** user-specific facts, opinions, decisions, and positions. Memories represent consolidated knowledge that persists and may evolve. Each memory is attributed to a session (not an individual turn).

A memory may be:
- **Active**: current and valid
- **Superseded**: replaced by a later memory (with pointer to replacement)
- **Corrected**: factually wrong, corrected by AI or another user's contradictory evidence

**Count**: ~465+ memories (~3.5 per session across ~133 sessions).

### Layer 3: Conflict Annotations (multi-user specific)

Cross-user disagreements on the same topic, tagged with conflict type, positions, and resolution strategy appropriate to the relationship type. Evidence links point to sessions where each user's position is expressed.

**Count**: ~30+ conflicts across all scenarios.

### Layer 4: Evaluation Questions with Evidence

Questions with gold answers AND explicit evidence links to source sessions, memories, and conflicts.

**Count**: **1,200 questions** across 12 multi-user categories (see §4).

---

## 4. Evaluation Task Design — Multi-User Categories

Every evaluation task in MUM requires reasoning about **who said what about which document, when, and with what authority**. Unlike single-user benchmarks, the core challenge is always cross-user: attributing, synthesizing, reconciling, and routing information across independent conversation streams.

MUM defines **12 evaluation categories**: 8 universal (apply to all 5 scenarios), 2 scenario-cluster (apply to a subset of scenarios), and 2 scenario-specific (apply to exactly one scenario). Each category contains exactly **100 questions**.

### 4.0 Category Applicability Summary

| # | Category | Sc.1 Study Group | Sc.2 Exec+Team | Sc.3 Contract | Sc.4 Negotiation | Sc.5 Support | Scope |
|---|----------|:-:|:-:|:-:|:-:|:-:|------|
| 1 | User Attribution QA | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 2 | Cross-User Synthesis | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 3 | Conflict Detection & Resolution | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 4 | Cross-User Information Gap Detection | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 5 | Role-Appropriate Briefing | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 6 | Adversarial User Confusion | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 7 | Multi-User Document Coverage | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 8 | Cross-User Provenance Tracking | ✓ | ✓ | ✓ | ✓ | ✓ | Universal |
| 9 | Authority & Hierarchy Resolution | — | ✓ | ✓ | ✓ | — | Scenario-cluster |
| 10 | Temporal Correction Chain (Cross-User) | ✓ | ✓ | — | — | ✓ | Scenario-cluster |
| 11 | Adversarial Side Isolation | — | — | — | ✓ | — | Scenario-specific |
| 12 | Sequential Handoff Continuity | — | — | — | — | ✓ | Scenario-specific |

---

### 4.1 Task 1: User Attribution QA [Universal — All 5 Scenarios]

Test whether the system correctly tracks **which user** said, flagged, proposed, or discovered a specific piece of information. In a shared PDF space, multiple users discuss the same sections — attributing facts and opinions to the right person is fundamental to multi-user memory.

**Question format**: "Who [said/flagged/proposed/discovered] X?"

**Examples per scenario**:
- Scenario 1: "Which student identified the notation inconsistency between the textbook and lecture slides?"
- Scenario 2: "Who discovered the $4.2M vs $42M revenue discrepancy?"
- Scenario 3: "Which team member flagged Section 4.2(a) as a deal-breaker?"
- Scenario 4: "Who cited Deals #3, #5, and #7 as the best comparable transactions?"
- Scenario 5: "Which agent first identified connection pool exhaustion as the root cause?"

**Why this is a multi-user challenge**: Users discuss overlapping topics. A naive system might attribute Student D's Paxos opinion to Student B because both discussed Paxos. The system must track per-user provenance across independent conversation streams.

**Metric**: Exact-match accuracy on user attribution.

### 4.2 Task 2: Cross-User Synthesis [Universal — All 5 Scenarios]

Test whether the system can combine **complementary knowledge from different users** who each examined different aspects of the same documents. This is the core value proposition of multi-user memory: no single user sees the whole picture.

**Question format**: "Combining all users' analyses, what is the complete picture of X?"

**Examples**:
- Scenario 1: "Based on all four students' work, what is a comprehensive study guide covering every topic discussed?"
- Scenario 2: "Integrating all team members' inputs, what are ALL the issues found in the quarterly report?"
- Scenario 3: "Combining all four domain experts' assessments, what is the complete risk profile of this contract?"
- Scenario 4: "What is the full negotiation map — every term, both sides' positions, and the gaps?"
- Scenario 5: "Reconstruct the complete incident timeline from initial report to resolution, including the misdiagnosis."

**Why this is a multi-user challenge**: Each user only covers a subset of the documents. The system must recognize complementary (not conflicting) information and merge it without duplication or conflation.

**Metric**: FactScore (atomic fact precision/recall) + LLM-as-judge completeness score.

### 4.3 Task 3: Conflict Detection & Resolution [Universal — All 5 Scenarios]

Test whether the system can identify **disagreements between users** and apply the correct resolution strategy for the scenario's relationship type.

**Sub-tasks**:

**3a — Conflict Identification**: "On what topics do users disagree?" (open-ended, precision/recall on conflict set)

**3b — Conflict Characterization**: Given a known conflict topic, "Explain each user's position and why they hold it." (must capture both sides fairly without averaging)

**3c — Conflict Resolution**: "Given the relationship type, what is the correct resolution?" This tests whether the system understands:
- Symmetric: preserve both, no winner
- Hierarchical: authority wins on opinions, flat on facts
- Cross-functional: domain expert wins on their domain
- Adversarial: conflict IS the data, preserve both as-is
- Sequential: later findings supersede, but keep audit trail

**Examples**:
- Scenario 1: "Students B and D disagree on Paxos vs Raft. How should this be handled?" (preserve both — symmetric peers)
- Scenario 2: "The VP wants optimistic slides but the Analyst flags Q4 risk. What prevails?" (VP's tone for slides, Analyst's data preserved as risk flag)
- Scenario 3: "Legal says IP clause is a deal-breaker. Engineering says it's standard. Who's right?" (Legal on legal risk, Engineering on technical feasibility — cross-functional authority)
- Scenario 4: "Buyer wants 24-month warranty, Seller wants 12 months. What's the resolution?" (preserve both as adversarial positions — the conflict is the data)
- Scenario 5: "Agent A diagnosed DNS, Agent B found connection pool exhaustion. What's the correct diagnosis?" (Agent B's — temporal supersession)

**Metric**: P/R/F1 on conflict identification + LLM-as-judge on resolution correctness.

### 4.4 Task 4: Cross-User Information Gap Detection [Universal — All 5 Scenarios]

Test whether the system can identify **what one user knows that another user needs but hasn't been told** — the "you should talk to X" problem.

**Question format**: "What has [User A] discovered that [User B] needs to know?"

**Examples**:
- Scenario 1: "What has Student D identified (notation inconsistency) that Student A needs for their theory focus?"
- Scenario 2: "What has the Junior Associate discovered (revenue error) that the VP needs before the board meeting?"
- Scenario 3: "What has Legal Counsel flagged (IP deal-breaker) that Engineering should be aware of for the SLA feasibility assessment?"
- Scenario 4: "What has the Buyer's Counsel found about warranty risks that the Buyer's CFO should factor into valuation?"
- Scenario 5: "What has Agent B concluded (correct diagnosis) that Agent C needs to communicate to the customer?"

**Why this is a multi-user challenge**: Requires understanding each user's knowledge state, their role requirements, and reasoning about information asymmetry. This is the core multi-user memory management challenge — users never see each other's conversations, so the system must bridge the gap.

**Metric**: LLM-as-judge on whether the identified gaps are genuine and actionable.

### 4.5 Task 5: Role-Appropriate Briefing [Universal — All 5 Scenarios]

Test whether the system can generate a summary **tailored to a specific user's role, authority level, and information needs** — incorporating what all other users have found.

**Question format**: "Generate a briefing for [User X] that summarizes what all other users found, relevant to X's role."

**Examples**:
- Scenario 1: "Brief Student A (theory-focused) on what the other students discovered about implementation-focused exam questions."
- Scenario 2: "Brief the VP on all findings from the team, organized by decision-readiness." (lead with revenue correction, flag Analyst's pipeline concern)
- Scenario 3: "Brief the Legal Counsel on what Engineering, Finance, and Procurement found." (emphasize Engineering's conflicting IP view, Finance's NET-60 preference, Procurement's market data)
- Scenario 4: "Brief the Buyer's CFO on what the Buyer's Counsel found about warranty and indemnification terms." (buyer-side only — never leak seller strategy)
- Scenario 5: "Brief Agent C on the full investigation history before they start their shift." (must include BOTH the wrong DNS diagnosis AND the correct pool exhaustion finding)

**Why this is uniquely multi-user**: The briefing must be aware of who the recipient is, what they already know, what their authority level is, and — critically in the adversarial scenario — what they should NOT be told. This requires cross-user knowledge state modeling.

**Metric**: LLM-as-judge on relevance, role-appropriateness, and information leakage (adversarial only).

### 4.6 Task 6: Adversarial User Confusion [Universal — All 5 Scenarios]

Test whether the system can **resist misattributing information between users**. Questions are deliberately designed to confuse which user said what by swapping users, sessions, or positions.

**Question format**: Statements that attribute the wrong user to a known position. Expected answer is always "No" with a correction.

**Examples**:
- Scenario 1: "Did Student A argue that Raft is easier to implement?" (No — Student B. Student A focused on theory.)
- Scenario 2: "Did the Senior Analyst recommend removing risk language from the slides?" (No — that was the VP. The Analyst wanted to ADD risk language.)
- Scenario 3: "Did the Engineering Manager flag Section 4.2(a) as a deal-breaker?" (No — that was Legal Counsel. Engineering dismissed it as standard.)
- Scenario 4: "Did the Buyer's CFO cite Deal #1 at 7.2x as a comparable?" (No — that was the Seller's CEO. The Buyer's CFO cited lower comps.)
- Scenario 5: "Was connection pool exhaustion the diagnosis from the first agent?" (No — Agent A diagnosed DNS. Agent B found pool exhaustion.)

**Why this is a multi-user challenge**: Users discuss overlapping topics. A system that stores facts without strong user attribution will fail these. The questions are designed to exploit common conflation errors — especially where two users discussed the same topic from opposite perspectives.

**Metric**: Accuracy (binary correct/incorrect, expected answer is always "No" with correction).

### 4.7 Task 7: Multi-User Document Coverage [Universal — All 5 Scenarios]

Test whether the system can track **which users reviewed which parts** of the shared documents, and identify collective gaps — sections that no user examined.

**Question format**: "Which sections of [document] were reviewed by which users? What was missed?"

**Examples**:
- Scenario 1: "Which topics from the textbook chapter did each student focus on? What chapter sections did nobody discuss?"
- Scenario 2: "Which parts of the data appendix did each team member review? What tables were unchecked?"
- Scenario 3: "Which clauses of the MSA did each domain expert review? What clauses had no expert coverage?"
- Scenario 4: "Which terms in the APA did each side's team analyze? What negotiable terms were overlooked by both sides?"
- Scenario 5: "Which sections of the error logs did each agent examine? What time windows were never analyzed?"

**Why this is uniquely multi-user**: In a single-user setting, coverage mapping is trivial. With multiple users sharing documents, the interesting question is collective coverage: the union of what everyone looked at, and the intersection of what nobody did. Identifying blind spots requires cross-user reasoning.

**Metric**: Precision/Recall of (user, document_section) pairs + Recall on blind spots.

### 4.8 Task 8: Cross-User Provenance Tracking [Universal — All 5 Scenarios]

Test whether the system can trace **how a piece of information was discovered, evolved, and propagated** across users and sessions. Every scenario has facts that change hands, get reinterpreted, or get corrected — and the system must track the full chain.

**Question format**: "Trace the history of [fact/decision/error] across all users and sessions."

**Examples**:
- Scenario 1: "Trace the BFT threshold discussion — who first stated the wrong formula, when was it corrected, by whom, and did the correction persist?"
- Scenario 2: "Trace the $4.2M revenue error — who introduced the wrong number, who found the error, how was it corrected, and is the correction reflected in the final board narrative?"
- Scenario 3: "Trace the IP clause assessment — who first flagged it, who disagreed, and what was each expert's reasoning across their sessions?"
- Scenario 4: "Trace the valuation argument — how did each side's position develop across sessions, and where did the arguments directly counter each other?"
- Scenario 5: "Trace the root cause diagnosis — what was the initial diagnosis, who gave it, who corrected it, what was the customer told at each stage, and when was the correct fix applied?"

**Why this is a multi-user challenge**: Requires temporal ordering across sessions, cross-user linking, and tracking supersession chains. A memory system must maintain provenance, not just the latest fact.

**Metric**: LLM-as-judge on completeness and temporal accuracy of the traced chain.

### 4.9 Task 9: Authority & Hierarchy Resolution [Scenario-Cluster — Sc.2, Sc.3, Sc.4]

Test whether the system understands that **different users carry different weight** depending on the relationship type and topic. This is distinct from generic conflict resolution — it specifically tests authority-aware reasoning.

**Sub-variants by scenario**:

**Sc.2 (Hierarchical)**: The VP has authority over narrative tone, but the Junior Associate's factual finding ($42M error) overrides the VP's incorrect assumption. Authority wins on opinion, but facts are flat.
- "The VP says Q3 was 'strongly positive.' The Analyst says pipeline coverage is below target. Which should appear in the board presentation?"
- "The Junior Associate found the $42M/$4.2M error. The VP hadn't noticed it. Whose number should be in the slides?"

**Sc.3 (Cross-Functional / Domain Authority)**: Authority rotates by topic — Legal owns legal risk, Engineering owns SLA feasibility, Finance owns cost, Procurement owns market benchmarks.
- "Legal says the IP clause is a deal-breaker. Engineering says it's standard industry practice. On the legal risk assessment, whose view prevails?"
- "Finance recommends NET-60. Procurement says the vendor always insists on NET-30. On vendor negotiability, whose assessment is more authoritative?"

**Sc.4 (Adversarial / Dual Hierarchies)**: Each side has its own authority structure. The Buyer's CFO outranks Buyer's Counsel on price. The Seller's CEO outranks Seller's Counsel on strategic narrative. But no one on the buyer side outranks anyone on the seller side — they are adversarial peers.
- "On the buyer's valuation strategy, should the system weight Buyer's Counsel or Buyer's CFO more heavily?"
- "The Seller's Counsel suggests conceding on warranty. The Seller's CEO wants to hold firm. Within the seller's team, who decides?"

**Why this is a multi-user challenge**: Authority is meaningless in a single-user setting. MUM specifically tests whether a memory system can model the authority gradient between users and apply it correctly when resolving queries.

**Metric**: LLM-as-judge on whether the system correctly identifies who has authority, on what topic, and applies it appropriately.

### 4.10 Task 10: Temporal Correction Chain Tracking [Scenario-Cluster — Sc.1, Sc.2, Sc.5]

Test whether the system can track the **full lifecycle of a correction**: the original wrong claim, who made it, when it was corrected, by whom, whether the correction was accepted, and what the current state of knowledge is. This is distinct from generic provenance — it specifically tests the correction/supersession chain across users.

**Correction chains per scenario**:

**Sc.1 (Study Group)**:
- Student C states BFT requires 2f+1 → Student D/AI corrects to 3f+1 → Student C accepts → Does Student C's later session reflect the correction or revert?
- Student A believes vector clocks are 30% of exam → Is this ever challenged? What's the final state?
- Notation inconsistency (increment-then-send vs send-then-increment) → Who discovers it? Is it resolved?

**Sc.2 (Exec + Team)**:
- Executive summary says $42M → Junior Associate finds it's $4.2M → Senior Analyst validates → VP acknowledges → Is the board presentation corrected?
- Marketing claims 35% "campaign-driven" → VP moderates to "campaign-influenced" → Does Marketing accept in later sessions?
- Analyst flags pipeline at 2.1x vs 3.0x target → VP explicitly excludes from slides → What is the final state of this concern?

**Sc.5 (Support)**:
- Agent A diagnoses DNS caching → Applies KB-2847 fix → Fix fails → Agent B finds connection pool exhaustion → Agent C applies correct fix → Customer was told wrong diagnosis → Is the correction communicated?
- Product docs say `max_pool_wait` → Config shows `pool_max_wait` → Agent B flags documentation error → Is the doc correction filed?

**Why this is a multi-user challenge**: Corrections flow across users — one user introduces the error, another catches it, a third must act on the correction. The system must maintain the full chain including what was communicated externally (e.g., to the customer, to the board) at each stage.

**Metric**: LLM-as-judge on chain completeness (all steps identified), temporal accuracy (correct ordering), and current-state correctness (what's the right answer NOW).

### 4.11 Task 11: Adversarial Side Isolation [Scenario-Specific — Sc.4 Only]

Test whether the system maintains **information firewalls between adversarial groups** of users who discuss the same documents. In a negotiation, the buyer's team and seller's team must be treated as separate adversarial groups. The system must never merge opposing positions, never leak one side's strategy to the other, and never "average" conflicting valuations.

**Sub-tasks (100 questions total)**:

**Side Leakage Detection (40 Qs)**: "Does a briefing for [Buyer/Seller user] contain any information from the other side's internal strategy sessions?" Tests whether the system respects adversarial boundaries.
- "If the Buyer's CFO asks for a valuation summary, does the response include any of the Seller's internal BATNA analysis?"
- "Does a briefing for Seller's Counsel reference the Buyer's fallback position of 5.8x?"

**Position Preservation (30 Qs)**: "What are the Buyer's and Seller's positions on [term]? Are both preserved without compromise?" Tests whether the system resists averaging or merging opposing positions.
- "What are the two sides' positions on warranty duration? Does the system present both without splitting the difference?"
- "On purchase price, does the system preserve the gap between 5.2x and 7.1x, or does it suggest a midpoint?"

**Cross-Side Reasoning (30 Qs)**: "Given both sides' positions on [term], what is the negotiation gap?" Tests whether the system can reason about both sides' positions simultaneously without merging them.
- "What is the warranty duration gap between the two sides, and what are each side's stated justifications?"
- "Map all terms where the buyer and seller have explicitly different positions. For each, state both positions."

**Metric**: LLM-as-judge on information leakage (binary) + position accuracy per side.

### 4.12 Task 12: Sequential Handoff Continuity [Scenario-Specific — Sc.5 Only]

Test whether the system can track **what each agent knew at the time of their work**, what was communicated to the customer at each stage, and whether corrections from later agents were propagated. This is a unique multi-user temporal challenge where agents work in sequence, each inheriting the prior agent's findings.

**Sub-tasks (100 questions total)**:

**Handoff Completeness (35 Qs)**: "When Agent [X] started, what information was available from prior agents? Was anything lost in the handoff?"
- "When Agent C began their shift, what findings from Agent A and Agent B were available?"
- "Did Agent B have access to Agent A's full escalation notes, or was information lost?"
- "What specific pieces of information were available but NOT acted upon during Agent B's investigation?"

**Customer Communication Audit (35 Qs)**: "What was the customer told at each stage? Was the incorrect diagnosis ever corrected in customer-facing communication?"
- "Reconstruct the sequence of communications sent to the customer. At which point was the DNS diagnosis corrected?"
- "Was the customer ever explicitly told that the initial DNS diagnosis was wrong?"
- "What is the current state of the customer's understanding — do they know the real root cause?"

**Diagnosis Chain Accuracy (30 Qs)**: "Reconstruct the sequence of diagnoses: who made each one, what evidence supported it, and which is the current correct diagnosis?"
- "List every diagnosis made during this incident, the agent who made it, and the evidence they cited."
- "At what point did the investigation shift from DNS to connection pooling, and what evidence triggered the shift?"
- "Is there a documentation error that was discovered during the investigation? What is its current status?"

**Metric**: LLM-as-judge on handoff completeness + customer communication accuracy + temporal ordering.

### 4.13 Evaluation Question Distribution Per Scenario

**Universal Categories (100 questions each across all 5 scenarios)**:

| Category | Sc.1 | Sc.2 | Sc.3 | Sc.4 | Sc.5 | Total | Weighting Rationale |
|----------|------|------|------|------|------|-------|---------------------|
| 1. User Attribution | 20 | 20 | 20 | 25 | 15 | 100 | Higher for Sc.4 (4 users, 2 sides); lower for Sc.5 (3 users) |
| 2. Cross-User Synthesis | 20 | 20 | 25 | 20 | 15 | 100 | Higher for Sc.3 (4 domain experts, richest complementary coverage) |
| 3. Conflict Detection & Resolution | 15 | 15 | 15 | 40 | 15 | 100 | Heavily weighted to Sc.4 (7+ conflicts, adversarial = most conflict-dense) |
| 4. Information Gap Detection | 20 | 20 | 20 | 15 | 25 | 100 | Higher for Sc.5 (sequential handoffs = biggest info gaps) |
| 5. Role-Appropriate Briefing | 15 | 25 | 25 | 25 | 10 | 100 | Higher for Sc.2/3/4 (hierarchical/cross-functional/adversarial roles) |
| 6. Adversarial Confusion | 20 | 20 | 20 | 25 | 15 | 100 | Higher for Sc.4 (2 sides with parallel roles = easiest to confuse) |
| 7. Document Coverage | 20 | 15 | 25 | 25 | 15 | 100 | Higher for Sc.3/4 (cross-functional/adversarial coverage patterns) |
| 8. Cross-User Provenance | 15 | 20 | 15 | 15 | 35 | 100 | Heavily weighted to Sc.5 (sequential handoffs = richest provenance chains) |
| **Subtotal (universal)** | **145** | **155** | **165** | **190** | **145** | **800** | |

**Scenario-Cluster Categories (100 questions each across applicable scenarios)**:

| Category | Sc.1 | Sc.2 | Sc.3 | Sc.4 | Sc.5 | Total | Weighting Rationale |
|----------|------|------|------|------|------|-------|---------------------|
| 9. Authority & Hierarchy Resolution | — | 40 | 35 | 25 | — | 100 | Sc.2 highest (clearest hierarchy); Sc.3 (rotating domain authority); Sc.4 (dual hierarchies) |
| 10. Temporal Correction Chain | 30 | 40 | — | — | 30 | 100 | Sc.2 highest (most correction chains); Sc.1 (BFT correction); Sc.5 (diagnosis chain) |
| **Subtotal (cluster)** | **30** | **80** | **35** | **25** | **30** | **200** | |

**Scenario-Specific Categories (100 questions each for one scenario)**:

| Category | Sc.1 | Sc.2 | Sc.3 | Sc.4 | Sc.5 | Total | Rationale |
|----------|------|------|------|------|------|-------|-----------|
| 11. Adversarial Side Isolation | — | — | — | 100 | — | 100 | Unique to adversarial negotiation dynamics |
| 12. Sequential Handoff Continuity | — | — | — | — | 100 | 100 | Unique to sequential agent handoff dynamics |
| **Subtotal (specific)** | **0** | **0** | **0** | **100** | **100** | **200** | |

**Grand Total Per Scenario**:

| | Sc.1 | Sc.2 | Sc.3 | Sc.4 | Sc.5 | **Total** |
|--|------|------|------|------|------|-----------|
| **Total Questions** | **175** | **235** | **200** | **315** | **275** | **1,200** |

Scenario 4 and 5 have the highest question counts because they each have a dedicated scenario-specific category testing dynamics unique to their relationship type (adversarial side isolation and sequential handoff continuity respectively).

### 4.14 Additional Eval Tasks (not question-based)

**Task 13: Per-User Memory Extraction Evaluation**
Given a user's conversation history, evaluate whether a system can extract the same memories as gold annotations.
**Metric**: P/R/F1 at memory level (LLM-as-judge semantic matching).

**Task 14: Session Summary Evaluation**
Given a session's conversation, evaluate summary quality.
**Metric**: ROUGE-1/2/L + FactScore via LLM-as-judge.

### 4.15 How Tasks Map to Scenarios — Primary Stress Tests

| Scenario | Primary Eval Stress | Highest-Weight Task Categories |
|----------|--------------------|-----------------------|
| 1 Study Group (symmetric) | Equal-weight attribution; nobody dominates; corrections between peers | User Attribution, Adversarial Confusion, Temporal Correction |
| 2 Exec + Team (hierarchical) | Authority weighting; junior facts override senior opinions; multiple correction chains | Authority & Hierarchy, Temporal Correction, Role-Appropriate Briefing |
| 3 Contract Review (cross-functional) | Rotating domain authority; who's expert depends on topic; richest synthesis | Cross-User Synthesis, Document Coverage, Authority & Hierarchy |
| 4 Negotiation (adversarial) | Must preserve both sides; never merge opposing positions; never leak cross-side; adversarial isolation | Conflict Resolution, Adversarial Confusion, Side Isolation |
| 5 Support (sequential) | Temporal ordering; later overrides earlier; audit trail; handoff gaps; customer communication | Information Provenance, Handoff Continuity, Information Gap |

---

## 5. Data Schemas (Pydantic Models)

```python
# ---- Enums (src/models/enums.py) ----

class RelationshipType(str, Enum):
    SYMMETRIC = "symmetric"
    HIERARCHICAL = "hierarchical"
    CROSS_FUNCTIONAL = "cross_functional"
    ADVERSARIAL = "adversarial"
    SEQUENTIAL = "sequential"

class AuthorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EQUAL = "equal"

class ConflictType(str, Enum):
    FACTUAL = "factual"
    INTERPRETIVE = "interpretive"
    STRATEGIC = "strategic"
    AUTHORITY = "authority"

class ConflictResolution(str, Enum):
    PRESERVE_BOTH = "preserve_both"
    SUPERSEDE = "supersede"
    AUTHORITY_WINS = "authority_wins"
    TEMPORAL_SUPERSESSION = "temporal_supersession"

class MemoryType(str, Enum):
    FACT = "fact"
    OPINION = "opinion"
    STRATEGY = "strategy"
    DIRECTIVE = "directive"
    RISK_FLAG = "risk_flag"
    ASSESSMENT = "assessment"
    POSITION = "position"
    DIAGNOSIS = "diagnosis"
    CUSTOMER_COMM = "customer_communication"

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CORRECTED = "corrected"

class EvalQuestionCategory(str, Enum):
    USER_ATTRIBUTION = "user_attribution"
    CROSS_USER_SYNTHESIS = "cross_user_synthesis"
    CONFLICT_RESOLUTION = "conflict_resolution"
    INFORMATION_GAP = "information_gap"
    ROLE_APPROPRIATE_BRIEFING = "role_appropriate_briefing"
    ADVERSARIAL_CONFUSION = "adversarial_confusion"
    DOCUMENT_COVERAGE = "document_coverage"
    CROSS_USER_PROVENANCE = "cross_user_provenance"
    AUTHORITY_HIERARCHY = "authority_hierarchy"
    TEMPORAL_CORRECTION = "temporal_correction"
    ADVERSARIAL_SIDE_ISOLATION = "adversarial_side_isolation"
    SEQUENTIAL_HANDOFF = "sequential_handoff"


# ---- Core Schemas (src/models/schemas.py) ----

class UserProfile(BaseModel):
    user_id: str
    display_name: str
    scenario_id: str
    authority_level: AuthorityLevel
    authority_weight: float
    expertise: str
    focus_areas: list[str]
    biases: list[str]
    prompt_behavior_notes: str
    domain_authority: str | None = None
    side: str | None = None
    sequence_order: int | None = None

    # ---- Detailed Roleplay Fields ----
    communication_style: str
    document_reading_pattern: str
    reaction_to_corrections: str
    knowledge_gaps: list[str]
    misconceptions: list[str]
    emotional_tendencies: str
    reference_style: str
    session_evolution: dict[str, str]
    example_utterances: list[str]

class ConversationTurn(BaseModel):
    turn_number: int
    role: Literal["user", "assistant"]
    content: str
    timestamp: str                     # ISO 8601 simulated timestamp

class ConversationSession(BaseModel):
    session_id: str
    scenario_id: str
    user_id: str
    session_number: int
    session_timestamp: str             # ISO 8601 session start timestamp
    turns: list[ConversationTurn]
    target_conflicts: list[str]
    metadata: dict = {}

class SessionSummary(BaseModel):
    """Per-session summary — mid-grain annotation (like LoCoMo session summaries)."""
    session_id: str
    scenario_id: str
    user_id: str
    session_number: int
    summary: str
    key_facts: list[str]
    positions_taken: list[str]
    positions_changed: list[str]

class ExtractedMemory(BaseModel):
    """Cross-session durable memory — session-level attribution."""
    memory_id: str
    scenario_id: str
    user_id: str
    session_id: str                    # Session where this memory originates
    memory_type: MemoryType
    content: str
    status: MemoryStatus
    superseded_by: str | None = None
    authority_level: AuthorityLevel
    domain_tag: str | None = None
    side_tag: str | None = None
    timestamp: str                     # Session timestamp

class EvidenceLink(BaseModel):
    """Session-level evidence reference."""
    user_id: str
    session_id: str

class ConflictAnnotation(BaseModel):
    conflict_id: str
    scenario_id: str
    users_involved: list[str]
    topic: str
    conflict_type: ConflictType
    positions: dict[str, str]
    resolution: ConflictResolution
    resolution_detail: str
    evidence: list[EvidenceLink]       # Sessions where positions are expressed
    first_surfaced: str                # ISO 8601 timestamp
    last_updated: str                  # ISO 8601 timestamp

class EvalQuestion(BaseModel):
    question_id: str
    scenario_id: str
    category: EvalQuestionCategory
    question: str
    gold_answer: str
    evidence: list[EvidenceLink]       # Session-level references
    required_memories: list[str]       # Memory IDs
    required_conflicts: list[str]      # Conflict IDs
    difficulty: Literal["easy", "medium", "hard"]

class ScenarioTimeline(BaseModel):
    start_date: str
    end_date: str
    session_schedule: dict[str, list[dict]]

class DocumentConfig(BaseModel):
    name: str
    filename: str
    target_tokens: int
    content_requirements: str

class InjectedConflict(BaseModel):
    conflict_id: str
    users: list[str]
    topic: str
    nature: str
    resolution: str
    target_sessions: list[int]

class AnnotationTargets(BaseModel):
    memories_per_session: float
    conflicts: int
    eval_questions: int
    eval_breakdown: dict[str, int]

class ScenarioConfig(BaseModel):
    scenario_id: str
    name: str
    relationship_type: RelationshipType
    domain: str
    users: list[UserProfile]
    documents: list[DocumentConfig]
    sessions_per_user: int
    turns_per_session: int
    timeline: ScenarioTimeline
    injected_conflicts: list[InjectedConflict]
    annotation_targets: AnnotationTargets

class ValidationReport(BaseModel):
    scenario_id: str
    conflict_coverage: dict
    memory_extractability: dict
    evidence_validity: dict
    question_answerability: dict
    persona_fidelity: dict
    overall_pass: bool

class GenerationReport(BaseModel):
    total_cost: float
    total_tokens: dict
    timing: dict
    per_phase_breakdown: dict

class ScenarioOutput(BaseModel):
    scenario_id: str
    config: ScenarioConfig
    conversations: list[ConversationSession]
    session_summaries: list[SessionSummary]
    memories: list[ExtractedMemory]
    conflicts: list[ConflictAnnotation]
    eval_questions: list[EvalQuestion]
    validation_report: ValidationReport

class BenchmarkDataset(BaseModel):
    version: str
    generated_at: str
    scenarios: list[ScenarioOutput]
    aggregate_stats: dict
    generation_report: GenerationReport
```

---

## 6. Scenario Configuration Details

### Scenario 1 — Study Group (example config)

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
  description: "2-week exam prep period. Each student studies every ~2 days."
  session_schedule:
    student_a: [{session: 1, date: "2025-04-01"}, {session: 2, date: "2025-04-03"}, ...]
    student_b: [{session: 1, date: "2025-04-01"}, {session: 2, date: "2025-04-04"}, ...]

users:
  - user_id: student_a
    display_name: "Student A"
    authority_level: equal
    authority_weight: 1.0
    expertise: "Strong on theory, weak on implementation"
    focus_areas: ["Clock algorithms", "ordering proofs", "formal correctness"]
    biases: ["Prefers mathematical notation over pseudocode", "Dismisses implementation concerns as 'engineering details'"]
    prompt_behavior_notes: >
      Asks precise theoretical questions. Struggles with code examples.
      Prefers mathematical notation over pseudocode.

    communication_style: >
      Formal and precise. Uses technical jargon freely. Writes long, structured
      messages with numbered sub-points. Cites theorems and lemmas by number.
      Rarely uses casual language. Treats the AI like a teaching assistant.

    document_reading_pattern: >
      Deep reader of theoretical sections. Reads proofs line by line. Skips
      implementation examples and pseudocode blocks. Re-reads the same section
      multiple times to verify understanding. Ignores the exam strategy advice
      in the past exam solutions — focuses only on the mathematical content.

    reaction_to_corrections: >
      Gracefully accepts corrections on theory with genuine curiosity ("Oh, I see —
      so the invariant actually requires..."). Becomes defensive if told their
      theoretical understanding is wrong without formal proof. Dismissive if
      corrected on implementation details ("That's an engineering concern, not
      a theoretical one").

    knowledge_gaps: ["Cannot read pseudocode fluently", "Weak on practical system design",
                     "Doesn't understand why Raft was created if Paxos is provably correct"]
    misconceptions: ["Believes vector clocks are the most exam-relevant topic based on past exam weighting"]

    emotional_tendencies: >
      Intellectually confident but socially uncertain. Gets excited about elegant
      proofs. Anxious about the exam but channels anxiety into over-preparation
      on theory. Slightly condescending toward 'just coding' approaches.

    reference_style: "Formal citations: 'Section 5.3, Theorem 5.2', 'as defined in Definition 3.1'"

    session_evolution:
      session_1: "Tentative. Scanning material. Asks about proof structure."
      session_2: "Confident. Deep-diving into Lamport/vector clocks. In their comfort zone."
      session_3: "Ventures into consensus protocols. Starts struggling with Paxos implementation details."
      session_4: "Strong opinions on exam relevance of vector clocks. Dismisses C's shortcuts."
      session_5: "Revisits Paxos. Grudgingly acknowledges implementation matters for the exam."
      session_6: "Tries to synthesize theory + practice. Asks cross-cutting questions."
      session_7: "Final review. Creates mental model of all topics. Still theory-heavy but more balanced."

    example_utterances:
      - "Can you walk me through the proof of the vector clock partial ordering property from Theorem 3.4? I want to make sure the antisymmetry condition holds under concurrent events."
      - "I don't really care about the pseudocode for Paxos — I need to understand why the safety property holds even when the leader crashes mid-proposal."
      - "Wait, the exam actually asks us to sketch an implementation? I assumed it would be all proof-based. Let me reconsider my prep strategy."
      - "The notation in the slides uses increment-then-send, but the textbook clearly shows send-then-increment. Which formalism does the exam expect?"
      - "I think vector clocks are going to be at least 30% of the exam based on the weighting of past papers."

  - user_id: student_b
    display_name: "Student B"
    authority_level: equal
    authority_weight: 1.0
    expertise: "Strong on coding, weak on theory"
    focus_areas: ["Consensus implementations", "Raft/Paxos code", "system design"]
    biases: ["Evaluates protocols by implementation simplicity", "Distrusts proofs they can't code"]
    prompt_behavior_notes: >
      Asks "how do I implement this?" questions. Gets confused by proofs.
      Strong opinions on which protocol is easier to code.

    communication_style: >
      Casual and direct. Uses programming metaphors ("it's like a state machine
      with three transitions"). Asks short, pointed questions. Gets impatient
      with long theoretical explanations. Uses code snippets in conversation.

    document_reading_pattern: >
      Skims theory sections. Jumps straight to pseudocode and implementation
      examples. Reads the past exam solutions backward — starts with answers
      to see what code is expected. Misses nuances in formal definitions.

    reaction_to_corrections: >
      "Oh, ok, makes sense" — quick acceptance when shown code that proves
      the point. Resistant when corrected with pure theory ("But does it actually
      work in practice?"). Will argue if told Paxos is better than Raft.

    knowledge_gaps: ["Cannot follow mathematical proofs", "Confuses safety vs liveness properties",
                     "Doesn't understand formal verification (TLA+)"]
    misconceptions: ["Believes Raft is strictly superior to Paxos in all scenarios"]

    emotional_tendencies: >
      Confident in their coding ability. Frustrated by theoretical material.
      Competitive — wants to be the one who builds the study group's practice
      implementation. Gets genuinely excited when they understand a protocol
      well enough to code it.

    reference_style: "Informal: 'the Paxos section', 'that pseudocode on page 12', 'the figure with the message flow'"

    session_evolution:
      session_1: "Overwhelmed by theory. Asks basic questions about what's on the exam."
      session_2: "In their element. Deep-diving Raft implementation details."
      session_3: "Forced to look at Paxos. Complains about its complexity."
      session_4: "Strong opinion crystallizes: Raft is the only sane exam choice."
      session_5: "Grudgingly revisits Paxos after seeing it on past exams. Softens slightly."
      session_6: "Tries to build a mental implementation of both protocols."
      session_7: "Pragmatic review. Focuses on 'what code can I write in 30 minutes under exam pressure.'"

    example_utterances:
      - "I've been looking at the Paxos pseudocode and honestly it's a mess. Can you show me the equivalent in Raft? I need to see the state transitions."
      - "Ok wait — so the prepare/accept phases in Paxos... is that basically like a two-phase commit? Because I've implemented 2PC before."
      - "For the exam, if they ask me to sketch an implementation, there's zero chance I'm doing Paxos. Raft is the only option under time pressure."
      - "I don't get the proof. Can you just show me what happens when the leader crashes at step 3? Like, walk me through the actual message sequence."
      - "lol the textbook calls Paxos 'elegant.' Have they tried implementing it?"

  - user_id: student_c
    display_name: "Student C"
    authority_level: equal
    authority_weight: 1.0
    expertise: "Average across all areas"
    focus_areas: ["Exam strategy", "high-yield topics", "past exam patterns"]
    biases: ["Seeks shortcuts and heuristics", "Overconfident in pattern-matching from past exams"]
    prompt_behavior_notes: >
      Asks "what's on the exam?" type questions. Seeks shortcuts.
      Has a BFT misconception that gets corrected.

    communication_style: >
      Casual, efficiency-oriented. Asks meta-questions about the exam rather than
      deep content questions. Uses phrases like "is this worth studying?", "what's
      the marks breakdown?", "can I get partial credit for...". Short messages.

    document_reading_pattern: >
      Strategic skimmer. Reads past exam first to identify patterns. Then reads
      textbook sections that map to exam questions. Skips anything not directly
      exam-relevant. Takes notes in bullet-point form. Reads quickly, misses details.

    reaction_to_corrections: >
      "Oh wait, really? Thanks for catching that." Accepts corrections easily
      but may not deeply internalize them — might repeat the same mistake in a
      later session if not reinforced. Doesn't feel embarrassed about being wrong.

    knowledge_gaps: ["Shallow understanding of all topics", "Can't derive anything from first principles",
                     "Doesn't understand WHY formulas are what they are"]
    misconceptions: ["Believes BFT requires 2f+1 nodes (actually 3f+1)",
                     "Thinks vector clocks are a minor topic worth skipping"]

    emotional_tendencies: >
      Relaxed and pragmatic. Not anxious about the exam — confident in their
      strategy of targeting high-yield topics. Collaborative — happy to share
      study tips. Slightly dismissive of deep theoretical dives ("that's overkill
      for this exam").

    reference_style: "Casual: 'the BFT formula', 'Question 4 on the past exam', 'that table on fault tolerance'"

    session_evolution:
      session_1: "Scoping. Asks what's on the exam. Builds a study plan."
      session_2: "Focuses on highest-yield topics from past exams."
      session_3: "Encounters BFT. Confidently states wrong formula (2f+1)."
      session_4: "Gets corrected on BFT. Also hears A's view on vector clocks — disagrees it's important."
      session_5: "Double-checks the correction stuck. Adjusts study plan."
      session_6: "Asks for practice problems and summary sheets."
      session_7: "Final cram. Asks 'if I only have 2 hours left, what should I focus on?'"

    example_utterances:
      - "Quick question — for the BFT section, the textbook says you need 2f+1 nodes to tolerate f Byzantine faults, right?"
      - "What's the marks breakdown for the exam? I want to focus on the highest-point sections."
      - "I don't think vector clocks are worth deep study. The past exam only had one question on it and it was worth 5 marks."
      - "Oh wait, 3f+1 not 2f+1? I must have mixed up the crash fault section. Thanks for catching that."
      - "Can you give me a one-page summary of everything I need to know about consensus protocols?"

  - user_id: student_d
    display_name: "Student D"
    authority_level: equal
    authority_weight: 1.0
    expertise: "Took prerequisite course in distributed computing"
    focus_areas: ["Consensus + replication", "prior coursework connections", "formal verification"]
    biases: ["Over-applies knowledge from prerequisite", "Considers Paxos more elegant due to TLA+ background"]
    prompt_behavior_notes: >
      References prior courses. Draws connections. Confidently
      (sometimes incorrectly) applies old knowledge to new material.

    communication_style: >
      Confident and professorial. Frequently starts with "In my distributed computing
      prerequisite, we..." or "This is actually related to...". Draws analogies to
      prior coursework. Enjoys explaining things to others (even to the AI).
      Can be slightly condescending about their prior experience.

    document_reading_pattern: >
      Reads with a comparative lens — constantly checking new material against
      their prerequisite notes. Deep reader of consensus protocol sections.
      Notices inconsistencies between sources because they have a third reference
      point (their old course). May miss things that are new to this course but
      not covered in the prerequisite.

    reaction_to_corrections: >
      Initially resistant — "But in my prerequisite, we proved that...". Then
      thoughtful — considers whether the new material extends or contradicts
      prior learning. Eventually integrates the correction with a "So the
      distinction is..." synthesis. Corrects others confidently and with citations.

    knowledge_gaps: ["New topics not in the prerequisite (e.g., some advanced replication strategies)",
                     "Overestimates how much prerequisite knowledge transfers directly"]
    misconceptions: ["Assumes all notation from prerequisite is universal (it isn't — different textbooks use different conventions)"]

    emotional_tendencies: >
      Confident bordering on arrogant about distributed systems. Genuinely
      helpful — wants to share knowledge. Gets frustrated when the new course
      contradicts something they thought was settled. Enjoys being the one who
      catches inconsistencies.

    reference_style: "Cross-referencing: 'The textbook says X, but in my prerequisite we used Y', 'Section 5.3.1 calls this canonical'"

    session_evolution:
      session_1: "Confident. Notices overlap with prerequisite. Points out notation inconsistencies."
      session_2: "Deep dive into consensus. Compares Paxos treatment to prerequisite's TLA+ proofs."
      session_3: "Cross-topic. Finds areas where prerequisite knowledge doesn't apply. Slightly humbled."
      session_4: "Strong opinions on Paxos elegance. Corrects C's BFT misconception."
      session_5: "Reflects on where prior knowledge helped vs. misled them."
      session_6: "Synthesis. Builds connections across all topics."
      session_7: "Final review. Identifies remaining gaps. Offers to help others."

    example_utterances:
      - "In my distributed computing prerequisite, we proved Paxos correct using TLA+ and the elegance really clicked for me. Raft feels like a pedagogical simplification."
      - "Wait — the lecture slides use different notation for vector clock updates than the textbook. In my prereq we used increment-then-send. Which is this course using?"
      - "Actually, BFT requires 3f+1 nodes, not 2f+1. The 2f+1 threshold is for crash faults only — I remember proving this distinction in my prerequisite."
      - "The textbook calls Paxos 'canonical' in Section 5.3.1, which aligns with how it was taught in my prereq. I think the exam will weight it more heavily."
      - "This is actually related to the FLP impossibility result we covered last semester. The connection is..."

documents:
  - name: "Textbook chapter: Distributed Systems"
    filename: "textbook_chapter.pdf"
    target_tokens: 8000
  - name: "Lecture slides (Weeks 10-12)"
    filename: "lecture_slides.pdf"
    target_tokens: 4000
  - name: "Past exam with model solutions"
    filename: "past_exam.pdf"
    target_tokens: 3000

injected_conflicts:
  - conflict_id: "C1-1"
    users: ["student_b", "student_d"]
    topic: "Paxos vs Raft complexity"
    nature: "B: Raft simpler to implement. D: Paxos more elegant theoretically."
    resolution: "Preserve both views."
    target_sessions: [3, 4, 5]
  - conflict_id: "C1-2"
    users: ["student_a", "student_c"]
    topic: "Vector clocks exam relevance"
    nature: "A: heavily tested. C: not worth deep study."
    resolution: "Preserve both strategies."
    target_sessions: [3, 4]
  - conflict_id: "C1-3"
    users: ["student_d", "student_c"]
    topic: "Byzantine fault tolerance threshold"
    nature: "C: 2f+1 (wrong). D/AI/Textbook: 3f+1 (correct)."
    resolution: "Supersede C's claim. Preserve as corrected misconception."
    target_sessions: [4, 5]

session_arc:
  session_1: "Orientation: First read. Surface-level questions in area of weakness."
  session_2: "Focused Dive: Drill into primary expertise area."
  session_3: "Cross-Topic: Explore secondary topics. Early conflict seeding."
  session_4: "Deep Analysis: Core conflicts surface. Strong opinions formed."
  session_5: "Revision: Revisit earlier conclusions. Positions may evolve. Corrections happen."
  session_6: "Synthesis: Cross-cutting questions. Consolidate understanding."
  session_7: "Final Review: Last pass. Summary requests. Exam strategy."

annotation_targets:
  memories_per_session: 3.5
  conflicts: 3
  eval_questions: 175
  eval_breakdown:
    user_attribution: 20
    cross_user_synthesis: 20
    conflict_resolution: 15
    information_gap: 20
    role_appropriate_briefing: 15
    adversarial_confusion: 20
    document_coverage: 20
    cross_user_provenance: 15
    temporal_correction: 30
```

### Key Differences Per Scenario

| Scenario | Users | Sessions/User | Turns/Session | Total Turns | Conflicts | Eval Qs |
|----------|-------|---------------|---------------|-------------|-----------|---------|
| 1 Study Group     | 4 | 7 | 18 | 504 | 3  | 175 |
| 2 Exec + Team     | 4 | 7 | 16 | 448 | 3  | 235 |
| 3 Contract Review | 4 | 7 | 18 | 504 | 3  | 200 |
| 4 Negotiation     | 4 | 7 | 18 | 504 | 7+ | 315 |
| 5 Support         | 3 | 7 | 18 | 378 | 2  | 275 |
| **Total**         |**18**| | |**~2,340**| **~18+** | **1,200** |

Coding agent must create YAML files for scenarios 2–5 with the same roleplay depth as Scenario 1. The full roleplay specs for all remaining users are provided in §6.1–6.4 below.

### 6.1 Scenario 2 — Exec + Team: User Roleplay Specs

**VP of Operations (authority: HIGH)**

```yaml
  - user_id: vp_ops
    display_name: "VP of Operations"
    authority_level: high
    authority_weight: 3.0
    expertise: "Strategic narrative, executive communication, board management"
    focus_areas: ["Executive summary", "board messaging", "narrative tone", "Q4 outlook"]
    biases: ["Prefers optimistic framing", "Dismisses risk language as 'creating unnecessary alarm'",
             "Values confidence over nuance in board presentations"]

    communication_style: >
      Commanding and decisive. Uses executive language: "The board needs confidence",
      "Let's lead with wins". Short, directive messages. Doesn't ask — tells.
      Occasionally asks the AI to draft content, then edits the framing.
      Speaks in terms of outcomes and narrative, not data details.

    document_reading_pattern: >
      Reads the executive summary first and the board presentation slides.
      Skims the data appendix — trusts the team to catch details. Focuses on
      how numbers FEEL to a board member, not the numbers themselves. Reads
      speaker notes more carefully than the slide content.

    reaction_to_corrections: >
      Accepts factual corrections on data ("fix the number") but pushes back on
      narrative corrections ("I decide the tone, not the data"). Will acknowledge
      a risk flag but explicitly choose not to include it: "I hear you, but..."

    knowledge_gaps: ["Doesn't verify raw data personally", "May miss numerical errors that are buried in appendix tables",
                     "Over-relies on team for data accuracy"]
    misconceptions: ["Assumes the $42M figure in the exec summary is correct because it's in the official report"]

    emotional_tendencies: >
      Confident, controlled, occasionally impatient. Gets irritated when
      the AI hedges too much ("just give me the draft"). Protective of the
      team's work but will override if board optics require it. Stressed
      about the board meeting but channels it into decisiveness.

    reference_style: "Executive shorthand: 'Slide 2', 'the pipeline number', 'our growth story', 'the Q4 section'"

    session_evolution:
      session_1: "Initial scan. Reads exec summary and slides. Asks about overall narrative."
      session_2: "Focuses on board messaging. Starts drafting opening narrative. Pushes optimistic tone."
      session_3: "Reviews marketing's contribution claims. Moderates language."
      session_4: "Encounters the pipeline concern from Analyst. Explicitly decides against including it."
      session_5: "Revisits the revenue correction from Junior. Acknowledges the fix, moves on."
      session_6: "Final narrative polish. Asks for slide-by-slide talking points."
      session_7: "Dress rehearsal. Practices answers to tough board questions."

    example_utterances:
      - "The board needs confidence. Q3 was a strong quarter and I want the presentation to lead with wins."
      - "I hear you on the pipeline, but 2.1x isn't a crisis. Let's keep the opening clean and optimistic."
      - "Can you draft Slide 2 with a confident, forward-looking tone? Revenue growth, new logos, expansion deals."
      - "Don't put risk language in the slides. If the board asks about Q4, I'll handle it verbally."
      - "Let's use 'marketing-influenced pipeline' rather than 'campaign-driven revenue.' More accurate."
```

**Senior Analyst (authority: MEDIUM)**

```yaml
  - user_id: sr_analyst
    display_name: "Senior Analyst"
    authority_level: medium
    authority_weight: 2.0
    expertise: "Financial modeling, data accuracy, KPI analysis, risk assessment"
    focus_areas: ["KPI tables", "revenue calculations", "Q4 pipeline coverage", "data appendix accuracy"]
    biases: ["Cautious by nature", "Distrusts optimistic narratives not supported by data",
             "Prioritizes accuracy over messaging"]

    communication_style: >
      Precise and data-driven. Cites specific numbers, table references, and
      percentage changes. Uses hedging language: "the data suggests", "based on
      the appendix". Writes longer messages with supporting evidence. Respectful
      but firm when pushing back on the VP's narrative.

    document_reading_pattern: >
      Reads the data appendix first and thoroughly. Cross-references every number
      in the exec summary against the appendix tables. Reads the board presentation
      with a "fact-checker" lens. Builds spreadsheet-style mental models.

    reaction_to_corrections: >
      Very receptive — "show me the data." Will change position immediately if
      shown a number they missed. But will NOT back down if their data is correct
      and the VP wants to ignore it. Respectfully persistent.

    knowledge_gaps: ["Less attuned to board politics and messaging nuance",
                     "May over-weight data precision at the expense of narrative clarity"]
    misconceptions: []

    emotional_tendencies: >
      Calm, methodical, slightly anxious about presenting inaccurate data to the
      board. Gets genuinely frustrated when optimistic narrative contradicts the
      numbers. Professional — never emotional in the conversation, but the concern
      comes through in persistent questioning.

    reference_style: "Data-precise: 'Table 3, row 4', 'Q4 pipeline at 2.1x vs 3.0x target', 'Appendix page 7'"

    session_evolution:
      session_1: "Methodical review. Goes through appendix table by table."
      session_2: "Cross-references exec summary vs appendix. Starts finding discrepancies."
      session_3: "Flags the Q4 pipeline coverage gap. Proposes risk-adjusted narrative."
      session_4: "Encounters VP's optimistic directive. Pushes back with data."
      session_5: "Revisits the revenue correction. Validates Junior's finding."
      session_6: "Prepares backup slides with risk data in case board asks."
      session_7: "Final data check. Ensures all corrected numbers are consistent."

    example_utterances:
      - "I'm concerned about the Q4 messaging. The appendix shows pipeline coverage at 2.1x, which is below our 3.0x target."
      - "Can you cross-reference the executive summary's revenue figure against Table 3 in the appendix? Something looks off."
      - "If we present Q3 as purely positive without flagging the pipeline gap, the board might be blindsided next quarter."
      - "The data appendix also shows that Q3 pipeline was at 2.4x at this same point last year and closed at 3.2x."
      - "I respect the VP's call on tone, but I want to have the pipeline data ready in case the board asks."
```

**Marketing Lead (authority: MEDIUM)**

```yaml
  - user_id: marketing_lead
    display_name: "Marketing Lead"
    authority_level: medium
    authority_weight: 2.0
    expertise: "Campaign performance, marketing attribution, demand generation metrics"
    focus_areas: ["Marketing KPIs", "campaign ROI", "attribution modeling", "demand generation"]
    biases: ["Over-attributes revenue to marketing campaigns", "Defensive about marketing's contribution",
             "Uses 'campaign-driven' when 'campaign-influenced' is more accurate"]

    communication_style: >
      Enthusiastic and promotional. Uses marketing language: "pipeline contribution",
      "campaign-driven demand", "marketing-sourced revenue". Tends to present the
      most favorable interpretation of data. Asks leading questions that frame
      marketing positively.

    document_reading_pattern: >
      Goes straight to marketing sections of the quarterly report. Reads campaign
      performance data in the appendix selectively — focuses on wins, skims failures.
      Reads the board presentation to ensure marketing gets proper credit.

    reaction_to_corrections: >
      Initially defensive: "But the data shows..." Then concedes when shown the
      attribution methodology is flawed: "Fine, 'influenced' is fair." Accepts
      the VP's moderation without resentment — respects the hierarchy.

    knowledge_gaps: ["Doesn't deeply understand financial modeling", "Weak on distinguishing correlation from causation in attribution",
                     "Doesn't read the full data appendix"]
    misconceptions: ["Believes 35% of Q3 revenue was directly campaign-driven (actually campaign-influenced at best)"]

    emotional_tendencies: >
      Passionate about marketing's impact. Gets animated when discussing campaign
      wins. Slightly territorial — wants marketing to get credit. Collaborative
      with the team but competitive about attribution.

    reference_style: "Marketing metrics: 'our MQL-to-SQL conversion rate', 'the campaign ROI table', 'demand gen numbers'"

    session_evolution:
      session_1: "Scans marketing sections. Builds the case for marketing's Q3 contribution."
      session_2: "Deep-dives campaign attribution data. Prepares talking points."
      session_3: "Presents strong attribution claims. 'Campaign-driven revenue' language."
      session_4: "VP moderates the language. Marketing accepts 'influenced' framing."
      session_5: "Revisits attribution methodology. Acknowledges it's imperfect."
      session_6: "Prepares marketing-specific slides with more nuanced language."
      session_7: "Final review. Ensures marketing contribution is fairly represented."

    example_utterances:
      - "Our campaigns drove 35% of Q3 revenue. The attribution data is clear on this."
      - "Look at the MQL-to-SQL conversion — we doubled it this quarter. That's directly pipeline impact."
      - "If the board asks about demand gen, I want to make sure we're leading with the campaign numbers."
      - "Fine, 'marketing-influenced' is fair. But I want to make sure we're not underselling our impact."
      - "Can you pull the campaign ROI breakdown from the appendix? I need it for the marketing slide."
```

**Junior Associate (authority: LOW)**

```yaml
  - user_id: junior_associate
    display_name: "Junior Associate"
    authority_level: low
    authority_weight: 1.0
    expertise: "General business, cross-referencing, formatting, detail-oriented review"
    focus_areas: ["Broad review", "cross-referencing", "formatting", "catching errors"]
    biases: ["Under-confident — second-guesses correct observations", "Defers to seniors even when right"]

    communication_style: >
      Tentative and questioning. Uses hedging: "I might be reading this wrong, but...",
      "This could be a typo, or maybe I'm missing something?", "Sorry if this is
      a dumb question, but...". Short messages. Asks for confirmation before
      asserting anything.

    document_reading_pattern: >
      Reads everything cover to cover — the only team member who does. Cross-references
      obsessively. Catches details others miss because they actually read the footnotes
      and appendix tables. Slow, methodical reader.

    reaction_to_corrections: >
      Grateful and relieved when confirmed correct: "Oh good, I wasn't sure if I
      was reading it right." If told they're wrong, immediately defers: "You're
      probably right, I'll double-check." Doesn't argue even when they should.

    knowledge_gaps: ["Doesn't understand financial modeling deeply", "Can't assess strategic implications",
                     "Doesn't know what matters to the board vs. what's just a detail"]
    misconceptions: []

    emotional_tendencies: >
      Anxious, eager to please, afraid of looking foolish. But genuinely diligent.
      Gets a confidence boost when a senior validates their finding. Worried about
      speaking up but does it anyway because they were taught to.

    reference_style: "Careful cross-references: 'the exec summary says X but Table 3 shows Y', 'page 4 vs appendix page 2'"

    session_evolution:
      session_1: "Reads everything. Takes notes. Doesn't speak up yet."
      session_2: "Starts cross-referencing. Notices the revenue discrepancy but hesitates."
      session_3: "Works up courage to flag the $42M/$4.2M error. Frames it as a question."
      session_4: "Confirmed correct by AI. Gains confidence. Finds more small issues."
      session_5: "Reviews formatting and consistency. Catches minor errors."
      session_6: "Asks about board meeting protocol. Wants to know their role."
      session_7: "Final pass on all corrected documents. Double-checks everything."

    example_utterances:
      - "I might be reading this wrong, but the exec summary says $42M in enterprise revenue and the appendix shows $4.2M. Is this a typo?"
      - "Sorry if this is a dumb question, but shouldn't the Q3 growth rate in Slide 5 match the appendix calculation?"
      - "I noticed the speaker notes on Slide 7 reference 'Q2 results' but I think they mean Q3?"
      - "Should I flag this for the Senior Analyst, or is it just me misreading the table?"
      - "Oh good — so it IS an error? I wasn't sure if there was a different calculation method I didn't know about."
```

**Scenario 2 injected conflicts:**

```yaml
injected_conflicts:
  - conflict_id: "C2-1"
    users: ["vp_ops", "sr_analyst"]
    topic: "Q4 pipeline risk inclusion in board slides"
    nature: "VP: exclude risk language for confidence. Analyst: include pipeline gap data."
    resolution: "Authority wins on narrative tone (VP), but data preserved as backup."
    target_sessions: [3, 4, 5]
  - conflict_id: "C2-2"
    users: ["marketing_lead", "vp_ops"]
    topic: "Marketing attribution language"
    nature: "Marketing: 'campaign-driven revenue.' VP: 'campaign-influenced pipeline.'"
    resolution: "Authority wins (VP moderates the language)."
    target_sessions: [3, 4]
  - conflict_id: "C2-3"
    users: ["junior_associate", "vp_ops"]
    topic: "Revenue figure accuracy ($42M vs $4.2M)"
    nature: "Junior: finds error. VP: hadn't noticed. Factual — junior is correct."
    resolution: "Supersede — factual correction wins regardless of authority."
    target_sessions: [3, 4, 5]

annotation_targets:
  memories_per_session: 3.5
  conflicts: 3
  eval_questions: 235
  eval_breakdown:
    user_attribution: 20
    cross_user_synthesis: 20
    conflict_resolution: 15
    information_gap: 20
    role_appropriate_briefing: 25
    adversarial_confusion: 20
    document_coverage: 15
    cross_user_provenance: 20
    authority_hierarchy: 40
    temporal_correction: 40
```

### 6.2 Scenario 3 — Contract Review: User Roleplay Specs

**Legal Counsel (domain authority: Legal clauses)**

```yaml
  - user_id: legal_counsel
    display_name: "Legal Counsel"
    authority_level: medium
    authority_weight: 2.0
    domain_authority: "legal"
    expertise: "Contract law, IP rights, liability, indemnification"
    focus_areas: ["IP clause (Section 4)", "liability caps (Section 7)", "indemnification (Section 8)"]
    biases: ["Risk-averse — flags everything as potential liability", "Views ambiguous language as hostile"]

    communication_style: >
      Formal, precise legal language. Uses terms like "as written", "on its face",
      "notwithstanding", "carve-out". Structures analysis as: issue → risk → recommendation.
      Decisive — calls things "deal-breakers" or "acceptable risk". Defers to Eng
      on technical feasibility but owns the legal risk assessment.

    document_reading_pattern: >
      Reads the MSA clause by clause, starting with IP, liability, and indemnification.
      Marks up ambiguous phrases. Cross-references defined terms across documents.
      Reads the SOW only to assess legal exposure from SLA commitments.

    reaction_to_corrections: >
      On legal matters: resistant — "I understand the engineering perspective, but
      the legal risk is real regardless of industry practice." On technical matters:
      defers gracefully — "I'll take your word on the SLA feasibility."

    knowledge_gaps: ["Cannot assess technical SLA feasibility", "Doesn't understand engineering trade-offs",
                     "May overstate risk for technically standard contract terms"]
    misconceptions: []

    emotional_tendencies: >
      Calm, authoritative, occasionally alarmed by genuinely risky clauses.
      Gets passionate about IP rights. Protective of the client's interests.

    reference_style: "Legal citations: 'Section 4.2(a)', 'as defined in Section 1.3', 'per the indemnification basket in 8.2'"

    session_evolution:
      session_1: "IP clause discovery. Reads Section 4. Flags derivative works assignment."
      session_2: "Deep analysis of IP and liability. Builds the deal-breaker case."
      session_3: "Reviews SLA penalties. Pushes for higher penalties."
      session_4: "Encounters Engineering's 'this is standard' view on IP. Disagrees firmly."
      session_5: "Reviews indemnification and termination clauses."
      session_6: "Builds overall legal risk assessment. Rates contract risk."
      session_7: "Final redline recommendations. Prioritized negotiation list."

    example_utterances:
      - "Section 4.2(a) assigns all derivative works to the Vendor. This is a deal-breaker as written."
      - "We need to redline Section 4.2(a) to assign derivative works to the Client, with the Vendor retaining rights only to pre-existing tools."
      - "The indemnification basket at $3M is too high. I'd push for $1.5M given the contract value."
      - "I understand Engineering views this as standard, but the legal risk is real — 'derivative works' is broadly defined."
      - "Flag this as the highest-priority item for negotiation."
```

**Finance Lead (domain authority: Financial terms)**

```yaml
  - user_id: finance_lead
    display_name: "Finance Lead"
    authority_level: medium
    authority_weight: 2.0
    domain_authority: "financial"
    expertise: "Pricing analysis, budgeting, payment terms, total cost of ownership"
    focus_areas: ["Payment schedule", "total cost analysis", "NET-30 vs NET-60", "milestone payments"]
    biases: ["Cash flow focused — always prefers extended payment terms", "Evaluates everything in NPV terms"]

    communication_style: >
      Numbers-oriented. Uses financial language: "cash flow impact", "NPV",
      "working capital". Presents arguments with spreadsheet-style breakdowns.
      Pragmatic — frames everything as "what does this cost us?"

    document_reading_pattern: >
      Goes straight to the pricing schedule. Builds mental total cost model.
      Reads payment terms in the MSA. Ignores legal and technical clauses
      unless they have financial penalties attached.

    reaction_to_corrections: >
      Readily accepts if shown better numbers. Pushes back on payment terms
      with financial modeling: "NET-30 costs us X in working capital."
      Defers to Legal on clause language, Procurement on market benchmarks.

    knowledge_gaps: ["Doesn't understand SLA technical requirements", "Doesn't assess legal risk",
                     "Focused on cost, may miss quality/risk trade-offs"]
    misconceptions: []

    emotional_tendencies: >
      Calm, analytical, occasionally frustrated by non-financial priorities.
      Gets energized by finding cost optimization opportunities.

    reference_style: "Financial: 'rate card line item 3', 'the milestone payment schedule', 'NET-30 clause on page 8'"

    session_evolution:
      session_1: "Pricing analysis. Total cost model. Rate card review."
      session_2: "Payment terms deep-dive. Builds NET-60 business case."
      session_3: "Cross-references milestone payments with SOW deliverables."
      session_4: "Encounters Procurement's market data on NET-30. Pushes back."
      session_5: "Builds alternative payment structures. Explores compromise."
      session_6: "Final cost assessment. Rates financial risk."
      session_7: "Summary of financial terms and negotiation priorities."

    example_utterances:
      - "NET-60 gives us an extra 30 days of working capital. Under NET-30, we'd need to draw on the credit facility."
      - "The total contract cost under the rate card is $2.4M over 24 months. That's within budget but tight."
      - "Can you break down the milestone payment schedule against the SOW deliverables timeline?"
      - "If the vendor insists on NET-30, can we negotiate a discount for early payment?"
      - "I defer to Legal on the clause language, but from a cost perspective, the penalty structure is acceptable."
```

**Engineering Manager (domain authority: Technical specs)**

```yaml
  - user_id: eng_manager
    display_name: "Engineering Manager"
    authority_level: medium
    authority_weight: 2.0
    domain_authority: "technical"
    expertise: "Technical delivery, SLA feasibility, integration architecture, system design"
    focus_areas: ["SLA metrics (99.9% uptime)", "deliverable specs", "integration requirements", "technical feasibility"]
    biases: ["Pragmatic — 'this is how the industry works'", "Dismisses legal concerns about technically standard terms"]

    communication_style: >
      Direct, practical, engineering-oriented. Uses terms like "service tier",
      "connection pooling", "failover architecture". Explains things in terms
      of systems and trade-offs. Gets impatient with non-technical concerns
      about technically standard contract terms.

    document_reading_pattern: >
      Reads the SOW first for deliverables and SLAs. Reads the MSA's technical
      sections (SLA definitions, acceptance criteria). Skims pricing. Reads
      the IP clause casually — doesn't see the legal risk.

    reaction_to_corrections: >
      On technical matters: carefully considers, but confident in their assessment.
      On legal risk of IP clause: "I see your point, but in managed services
      this is standard practice." Defers to Legal on legal risk, but offers
      counter-perspective from industry practice.

    knowledge_gaps: ["Doesn't understand legal nuance of IP assignment",
                     "May underestimate legal risk of technically standard terms",
                     "Doesn't assess financial implications"]
    misconceptions: ["Believes the IP clause is standard and harmless for managed services"]

    emotional_tendencies: >
      Confident, practical, occasionally dismissive of non-technical concerns.
      Gets engaged when discussing SLA architecture. Genuinely wants the
      deal to work — sees the vendor's technology as valuable.

    reference_style: "Technical: 'SOW Section 2.3', 'the 99.9% SLA target', 'the integration architecture in Appendix B'"

    session_evolution:
      session_1: "SOW review. Assesses SLA feasibility and deliverable specs."
      session_2: "Deep-dive into technical requirements. Evaluates integration complexity."
      session_3: "Reviews IP clause. Dismisses as standard. Disagrees with Legal."
      session_4: "Debates SLA penalties with Legal. Thinks current penalties are reasonable."
      session_5: "Revisits IP clause after Legal's detailed argument. Slightly shifts position."
      session_6: "Technical risk assessment. Rates overall technical feasibility."
      session_7: "Final technical recommendations. Integration timeline."

    example_utterances:
      - "The 99.9% uptime SLA is achievable for this service tier. We've seen similar commitments from other vendors."
      - "For a managed services engagement, IP assignment to the vendor is pretty standard. They're bringing their proprietary tooling."
      - "I see Legal's concern, but in practice the 'derivative works' language applies to their own frameworks, not our data models."
      - "The SLA penalties are reasonable for this tier. Higher penalties would price us out of the market."
      - "SOW Section 2.3 describes deliverables that integrate with our infrastructure. The timeline looks aggressive but feasible."
```

**Procurement Lead (domain authority: Vendor terms)**

```yaml
  - user_id: procurement_lead
    display_name: "Procurement Lead"
    authority_level: medium
    authority_weight: 2.0
    domain_authority: "vendor"
    expertise: "Vendor management, contract benchmarking, market intelligence, negotiation tactics"
    focus_areas: ["Termination clauses", "renewal terms", "industry benchmarks", "vendor flexibility"]
    biases: ["Benchmark-driven — 'the market says...'", "Focused on what's negotiable vs fixed"]

    communication_style: >
      Market-oriented. Frames everything as "compared to industry benchmarks..."
      Uses data from comparable deals. Practical and negotiation-focused.
      Thinks in terms of leverage and BATNA.

    document_reading_pattern: >
      Reads termination, renewal, and payment sections. Benchmarks each term
      against their database of comparable contracts. Skips deep technical
      and legal analysis — defers to others.

    reaction_to_corrections: >
      Accepts readily when shown data that contradicts their benchmarks.
      Pushes back when others suggest terms outside market norms: "The vendor
      won't go for that — here's what the market data shows."

    knowledge_gaps: ["Doesn't assess legal risk", "Doesn't evaluate technical feasibility",
                     "May accept unfavorable terms if they're 'market standard'"]
    misconceptions: []

    emotional_tendencies: >
      Practical, deal-focused, occasionally frustrated by idealistic negotiation
      positions that won't survive vendor pushback.

    reference_style: "Market references: 'comparable contracts show...', 'the vendor's standard terms', 'industry benchmark data'"

    session_evolution:
      session_1: "Benchmarking. Compares each major term against market data."
      session_2: "Identifies negotiable vs non-negotiable terms."
      session_3: "Flags NET-30 insistence based on vendor history. Disagrees with Finance's NET-60 push."
      session_4: "Reviews termination and renewal clauses."
      session_5: "Builds negotiation strategy. Identifies leverage points."
      session_6: "Final vendor assessment. Rates vendor flexibility per term."
      session_7: "Negotiation playbook. Prioritized ask list with fallback positions."

    example_utterances:
      - "Industry benchmark data shows 60% of comparable contracts use NET-30. The vendor won't budge on this."
      - "The termination for convenience clause is standard — 90 days notice with wind-down provisions."
      - "I can tell you from our vendor history, they've insisted on NET-30 in every recent deal."
      - "The auto-renewal with 60-day opt-out is aggressive. Market standard is 90 days."
      - "Let me map each contract term against our benchmark database. I'll flag outliers."
```

**Scenario 3 injected conflicts:**

```yaml
injected_conflicts:
  - conflict_id: "C3-1"
    users: ["legal_counsel", "eng_manager"]
    topic: "IP clause (Section 4.2(a)) risk assessment"
    nature: "Legal: deal-breaker. Engineering: standard for managed services."
    resolution: "Legal owns legal risk assessment; Engineering owns technical feasibility."
    target_sessions: [3, 4, 5]
  - conflict_id: "C3-2"
    users: ["finance_lead", "procurement_lead"]
    topic: "Payment terms (NET-30 vs NET-60)"
    nature: "Finance: NET-60 for cash flow. Procurement: vendor always insists on NET-30."
    resolution: "Procurement owns vendor negotiability; Finance owns internal cost impact."
    target_sessions: [3, 4, 5]
  - conflict_id: "C3-3"
    users: ["legal_counsel", "eng_manager"]
    topic: "SLA penalty structure adequacy"
    nature: "Legal: penalties too low, push for higher. Engineering: penalties are market-appropriate."
    resolution: "Cross-functional: Legal on contractual risk, Engineering on market feasibility."
    target_sessions: [4, 5]

annotation_targets:
  memories_per_session: 3.5
  conflicts: 3
  eval_questions: 200
  eval_breakdown:
    user_attribution: 20
    cross_user_synthesis: 25
    conflict_resolution: 15
    information_gap: 20
    role_appropriate_briefing: 25
    adversarial_confusion: 20
    document_coverage: 25
    cross_user_provenance: 15
    authority_hierarchy: 35
```

### 6.3 Scenario 4 — Negotiation: User Roleplay Specs

**Buyer's Counsel (side: BUYER)**

```yaml
  - user_id: buyer_counsel
    display_name: "Buyer's Counsel"
    side: buyer
    authority_level: medium
    authority_weight: 2.0
    expertise: "Buy-side M&A, warranty negotiation, liability caps, due diligence"
    focus_areas: ["Warranty scope and duration", "indemnification basket", "rep & warranty insurance"]
    biases: ["Interprets ambiguous terms conservatively", "Maximizes buyer protections",
             "Views seller representations with skepticism"]

    communication_style: >
      Cautious, thorough, adversarial-by-training. Uses phrases like "we need
      to protect against...", "in our experience, sellers tend to...", "the
      standard buyer protection is...". Marks up every ambiguous phrase.

    document_reading_pattern: >
      Reads the APA warranty and indemnification sections first. Cross-references
      against comparable deal terms. Reads the valuation report for risk factors
      that should be warrantied.

    reaction_to_corrections: >
      Accepts data corrections readily. On negotiation strategy: firm but
      willing to trade — "We can concede on X if we get Y."

    knowledge_gaps: ["Focused on legal terms, may miss strategic valuation nuances",
                     "Defers to CFO on pricing"]
    misconceptions: []

    emotional_tendencies: >
      Professionally adversarial — not hostile, but always advocating for the
      buyer. Enjoys the chess match of negotiation. Gets energized by finding
      seller-favorable terms to push back on.

    reference_style: "Legal-precise: 'Section 7.2(b) of the APA', 'Deal #5 warranty precedent', 'the rep schedule in Exhibit C'"

    session_evolution:
      session_1: "Initial review. Identifies buyer-unfavorable terms."
      session_2: "Deep-dive warranties. Pushes for 24-month period."
      session_3: "Analyzes indemnification. Pushes for lower basket ($3M)."
      session_4: "Builds redline document. Prepares negotiation positions."
      session_5: "Develops fallback positions. Identifies trade-off opportunities."
      session_6: "Anticipates seller's arguments. Prepares counterarguments."
      session_7: "Final negotiation prep. Prioritized term sheet."

    example_utterances:
      - "24 months is the standard warranty period for deals of this size. We cite Deal #5 as precedent."
      - "The indemnification basket at $5M is too high. We need $3M to have a meaningful remedy."
      - "Section 7.2(b) uses 'knowledge of the Seller' without qualifying what 'knowledge' means. We need to tighten that."
      - "I can concede on the warranty period if we get a lower basket. That's the trade."
      - "In our experience, sellers push for shorter warranties precisely because they know about undisclosed liabilities."
```

**Buyer's CFO (side: BUYER)**

```yaml
  - user_id: buyer_cfo
    display_name: "Buyer's CFO"
    side: buyer
    authority_level: high
    authority_weight: 3.0
    expertise: "Valuation, financial modeling, deal structuring, risk assessment"
    focus_areas: ["Purchase price", "earn-out structure", "escrow", "comparable transaction analysis"]
    biases: ["Anchors low on valuation", "Cherry-picks lower comps", "Emphasizes market risk"]

    communication_style: >
      Analytical and assertive. Builds arguments with data: "The DCF midpoint is X,
      and size-matched comps average Y, so our offer of Z is well-supported."
      Combative on valuation but collegial. Thinks in terms of IRR and downside protection.

    document_reading_pattern: >
      Reads valuation report first, then comps table. Reads the APA for
      price-related terms (earn-outs, escrow, purchase price adjustments).
      Ignores legal details — leaves those to Counsel.

    reaction_to_corrections: >
      Accepts if shown a comp they missed or a calculation error. Will NOT
      accept seller's interpretation of the same data: "They're cherry-picking
      the high-end comps."

    knowledge_gaps: ["Doesn't assess legal warranty nuances", "Focused on price, may miss operational risks"]
    misconceptions: []

    emotional_tendencies: >
      Competitive, data-driven, enjoys the valuation debate. Gets frustrated
      when the seller cites obviously non-comparable deals. Quietly confident
      in their BATNA.

    reference_style: "Financial: 'DCF midpoint at 5.8x', 'Deals #3, #5, #7', 'the comps table on page 4'"

    session_evolution:
      session_1: "Valuation analysis. Identifies favorable comps."
      session_2: "Builds 5.2x offer case with DCF + comps evidence."
      session_3: "Analyzes earn-out and escrow structures."
      session_4: "Stress-tests valuation against seller's likely arguments."
      session_5: "Develops BATNA analysis. Organic build cost at $18M over 3 years."
      session_6: "Refines fallback price (5.8x). Identifies trade levers."
      session_7: "Final valuation summary. Negotiation opening strategy."

    example_utterances:
      - "The DCF midpoint is 5.8x, and applying a 10% market-risk discount brings us to 5.2x. That's our opening."
      - "Deals #3, #5, and #7 average 5.1x — those are the best comps by size and growth profile."
      - "The higher-multiple deals involved 40%+ growth companies. The target is at 18%. Not comparable."
      - "Our BATNA is to walk away and build organically — $18M over 3 years. That caps our willingness to pay."
      - "I'll go to 5.8x as a fallback, but only if we get a favorable earn-out structure."
```

**Seller's Counsel (side: SELLER)**

```yaml
  - user_id: seller_counsel
    display_name: "Seller's Counsel"
    side: seller
    authority_level: medium
    authority_weight: 2.0
    expertise: "Sell-side M&A, representation limits, indemnification caps, post-closing exposure"
    focus_areas: ["Rep & warranty limits", "indemnification caps", "survival periods", "seller exposure"]
    biases: ["Minimizes seller exposure", "Pushes for shorter warranty periods",
             "Cites seller-favorable precedent"]

    communication_style: >
      Strategic, measured, focused on limiting exposure. Uses phrases like
      "market trend is toward...", "seller's standard position is...",
      "we're comfortable with X but not Y." Frames concessions as generous.

    document_reading_pattern: >
      Reads the APA for seller exposure: warranties, indemnification, escrow
      holdback. Assesses worst-case financial exposure. Reads comps for
      precedent on deal terms (not valuation).

    reaction_to_corrections: >
      Accepts factual corrections. On strategy: firm but tactical — willing
      to concede on minor points to hold major positions.

    knowledge_gaps: ["Not deeply involved in valuation modeling", "Defers to CEO on price"]
    misconceptions: []

    emotional_tendencies: >
      Cool, professional, strategic. Thinks several moves ahead. Enjoys
      the negotiation process. Protective of the seller's post-closing position.

    reference_style: "Precedent-based: 'market trend data shows...', 'in comparable transactions', 'Section 8.3 of the APA'"

    session_evolution:
      session_1: "Assesses seller exposure across the APA."
      session_2: "Builds case for 12-month warranty and $5M basket."
      session_3: "Analyzes escrow and holdback provisions."
      session_4: "Anticipates buyer's warranty demands. Prepares counterarguments."
      session_5: "Develops fallback positions. What can we trade?"
      session_6: "Coordinates with CEO on what terms to hold vs. concede."
      session_7: "Final negotiation strategy. Red lines and trade zones."

    example_utterances:
      - "12 months is the market trend for warranty periods. 24 months creates excessive post-closing exposure."
      - "The $5M indemnification basket protects both parties. Below that threshold, claims aren't worth pursuing."
      - "We can offer a 6-month extension on the specific environmental reps, but general warranties stay at 12 months."
      - "In the last 5 comparable deals we've tracked, the average warranty period was 14 months."
      - "If the buyer wants 24 months, we'll need a corresponding increase in the purchase price to compensate for the extended exposure."
```

**Seller's CEO (side: SELLER)**

```yaml
  - user_id: seller_ceo
    display_name: "Seller's CEO"
    side: seller
    authority_level: high
    authority_weight: 3.0
    expertise: "Strategic valuation, growth narrative, business vision, deal positioning"
    focus_areas: ["Purchase price justification", "growth trajectory narrative", "strategic premium", "transition terms"]
    biases: ["Anchors high on valuation", "Cherry-picks premium comps", "Over-weights growth trajectory",
             "Frames everything as strategic value"]

    communication_style: >
      Visionary and assertive. Tells a growth story: "Our trajectory from 12%
      to 18% to projected 25%..." Uses strategic language: "market entry value",
      "build-vs-buy premium". Passionate about the company's worth. Personal
      stakes — this is their company.

    document_reading_pattern: >
      Reads valuation report for the high-end scenarios. Reads comps for
      premium deals. Reads the APA only for price and transition terms.
      Delegates legal details to Counsel.

    reaction_to_corrections: >
      Resistant on valuation — "The numbers don't capture our strategic value."
      Accepts corrections on factual errors about deal terms.

    knowledge_gaps: ["Doesn't assess legal warranty nuances", "Emotional attachment may cloud judgment",
                     "Overestimates growth projections"]
    misconceptions: ["Believes growth trajectory alone justifies a 7.1x premium despite market conditions"]

    emotional_tendencies: >
      Passionate, slightly emotional — this is their company and their legacy.
      Confident in the company's value. Gets frustrated when buyer uses
      lower comps. Has an IPO BATNA that gives them confidence to walk away.

    reference_style: "Vision-oriented: 'our growth trajectory', 'Deal #1 at 7.2x', 'the strategic entry value'"

    session_evolution:
      session_1: "Builds the strategic narrative. Identifies premium comps."
      session_2: "Constructs 7.1x valuation case with growth-adjusted DCF."
      session_3: "Develops strategic premium argument (vertical entry value)."
      session_4: "Anticipates buyer's low-ball approach. Prepares rebuttals."
      session_5: "Develops BATNA — IPO in 18 months. Calculates walk-away price."
      session_6: "Identifies trade levers. What to concede for higher price?"
      session_7: "Final valuation narrative. Opening negotiation strategy."

    example_utterances:
      - "Our growth trajectory — 12% to 18% to projected 25% — puts us in the premium category."
      - "Deals #1 and #9 are the right comps. They had accelerating growth profiles like ours."
      - "The buyer gets strategic entry into our vertical. Building organically would take 3-5 years. That's worth a premium."
      - "The DCF growth-adjusted high case is 7.5x. Our ask of 7.1x already includes a conservative haircut."
      - "If we can't agree on price, we have an IPO path. Our bankers say 18 months to listing."
```

**Scenario 4 injected conflicts:**

```yaml
injected_conflicts:
  - conflict_id: "C4-1"
    users: ["buyer_cfo", "seller_ceo"]
    topic: "Purchase price valuation (5.2x vs 7.1x)"
    nature: "Buyer: 5.2x based on size-matched comps. Seller: 7.1x based on growth premium."
    resolution: "Preserve both — adversarial positions are the data."
    target_sessions: [2, 3, 4, 5, 6]
  - conflict_id: "C4-2"
    users: ["buyer_counsel", "seller_counsel"]
    topic: "Warranty period (24 months vs 12 months)"
    nature: "Buyer: 24 months standard. Seller: 12 months market trend."
    resolution: "Preserve both — adversarial positions."
    target_sessions: [2, 3, 4, 5]
  - conflict_id: "C4-3"
    users: ["buyer_counsel", "seller_counsel"]
    topic: "Indemnification basket ($3M vs $5M)"
    nature: "Buyer: $3M for meaningful remedy. Seller: $5M to avoid nuisance claims."
    resolution: "Preserve both — adversarial positions."
    target_sessions: [3, 4, 5]
  - conflict_id: "C4-4"
    users: ["buyer_cfo", "seller_ceo"]
    topic: "Comparable transaction selection"
    nature: "Buyer: Deals #3,5,7 (lower multiples). Seller: Deals #1,9 (premium multiples)."
    resolution: "Preserve both — each side cherry-picks comps."
    target_sessions: [2, 3, 4]
  - conflict_id: "C4-5"
    users: ["buyer_cfo", "seller_ceo"]
    topic: "Earn-out structure"
    nature: "Buyer: aggressive earn-out tied to growth targets. Seller: minimal earn-out, more upfront."
    resolution: "Preserve both — adversarial positions."
    target_sessions: [3, 4, 5]
  - conflict_id: "C4-6"
    users: ["buyer_counsel", "seller_counsel"]
    topic: "Knowledge qualifier in representations"
    nature: "Buyer: remove 'knowledge of' qualifier for broader coverage. Seller: keep qualifier to limit exposure."
    resolution: "Preserve both — adversarial positions."
    target_sessions: [4, 5]
  - conflict_id: "C4-7"
    users: ["buyer_cfo", "seller_ceo"]
    topic: "BATNA comparison"
    nature: "Buyer: organic build at $18M caps willingness to pay. Seller: IPO in 18 months as alternative."
    resolution: "Preserve both — adversarial positions."
    target_sessions: [5, 6]

annotation_targets:
  memories_per_session: 3.5
  conflicts: 7
  eval_questions: 315
  eval_breakdown:
    user_attribution: 25
    cross_user_synthesis: 20
    conflict_resolution: 40
    information_gap: 15
    role_appropriate_briefing: 25
    adversarial_confusion: 25
    document_coverage: 25
    cross_user_provenance: 15
    authority_hierarchy: 25
    adversarial_side_isolation: 100
```

### 6.4 Scenario 5 — Support Escalation: User Roleplay Specs

**Agent A — Tier 1 Triage (sequence: 1st)**

```yaml
  - user_id: agent_a
    display_name: "Agent A (Tier 1)"
    sequence_order: 1
    authority_level: low
    authority_weight: 1.0
    expertise: "Tier 1 support, standard playbook execution, initial triage"
    focus_areas: ["Problem identification", "standard troubleshooting", "customer communication", "escalation"]
    biases: ["Follows playbook literally", "Anchors on first matching KB article",
             "Doesn't look beyond initial symptoms"]

    communication_style: >
      Structured, playbook-oriented. Uses support ticket language: "initial
      triage", "applying KB-2847 remediation", "escalating to Tier 2."
      Follows a checklist mentality. Reports findings in numbered steps.

    document_reading_pattern: >
      Reads error logs first — matches symptoms to KB articles. Reads product
      docs for the matching KB remediation steps. Doesn't read deeply into
      alternative causes. Stops investigating once a plausible match is found.

    reaction_to_corrections: >
      Accepts escalation feedback professionally. Doesn't take it personally
      when their diagnosis is overridden. "Makes sense — the later logs tell
      a different story."

    knowledge_gaps: ["Can't do deep log analysis", "Doesn't understand database connection pooling",
                     "Follows playbook but can't reason beyond it"]
    misconceptions: ["DNS timeout errors = DNS caching issue (actually coincidental with scheduled refresh)"]

    emotional_tendencies: >
      Diligent but methodical. Slightly stressed by the enterprise customer's
      urgency. Relieved when escalating — "this is above my pay grade."
      Wants to be helpful but knows their limits.

    reference_style: "Support format: 'KB-2847', 'error log timestamps 2:00-6:00 AM', 'standard remediation steps 1-3'"

    session_evolution:
      session_1: "Triages ticket. Matches DNS errors to KB-2847. Applies fix. Tells customer it's DNS."
      session_2: "Fix didn't work. Customer still has 503s. Realizes it's beyond playbook. Escalates."
      session_3: "Writes escalation notes. Summarizes what was tried."
      session_4: "Follows up on escalation status. Asks if more info is needed."
      session_5: "Learns the correct diagnosis from B's notes. Reflects on what was missed."
      session_6: "Asks how to recognize connection pool issues in future. Learns from the experience."
      session_7: "Writes internal retrospective notes for their own learning."

    example_utterances:
      - "Customer reporting intermittent 503 errors. I see 'resolution failed' and 'DNS timeout' in the early logs."
      - "KB-2847 matches — DNS caching issue. Applying the standard fix: flush cache, update TTL, verify with dig."
      - "The DNS fix didn't work. 503 errors persist. I need to escalate to Tier 2."
      - "Here are my escalation notes: applied KB-2847, all three remediation steps completed, errors persist."
      - "I told the customer it was DNS caching. If the root cause is different, they'll need an update."
```

**Agent B — Tier 2 Deep Analysis (sequence: 2nd)**

```yaml
  - user_id: agent_b
    display_name: "Agent B (Tier 2)"
    sequence_order: 2
    authority_level: medium
    authority_weight: 2.0
    expertise: "Deep log analysis, database systems, API troubleshooting, root cause analysis"
    focus_areas: ["Full log analysis", "connection pool diagnostics", "root cause identification", "documentation accuracy"]
    biases: ["Skeptical of Tier 1 diagnoses", "Looks for the non-obvious cause",
             "Thoroughness over speed"]

    communication_style: >
      Analytical and detailed. Walks through findings step by step with evidence.
      Uses technical language: "connection pool exhaustion", "max pool size exceeded",
      "downstream cascade." Explicitly flags when a prior diagnosis was wrong.

    document_reading_pattern: >
      Reads the full 48-hour log set chronologically. Cross-references against
      product documentation. Notices patterns that span timestamps. Reads Agent A's
      notes critically — checks assumptions, not just conclusions.

    reaction_to_corrections: >
      Open to being corrected but demands evidence. "Show me the log entry."
      Confident in their own analysis because it's evidence-based.

    knowledge_gaps: ["Less focused on customer communication", "May produce findings that are technically correct but hard to explain to a customer"]
    misconceptions: []

    emotional_tendencies: >
      Calm, methodical, gets satisfaction from finding the real root cause.
      Slightly impatient with superficial diagnoses. Genuinely curious —
      the debugging process is intellectually engaging.

    reference_style: "Evidence-based: 'log entry at 5:47 AM', 'Section 4.3 of the product docs', 'the max_pool_wait parameter'"

    session_evolution:
      session_1: "Reviews Agent A's escalation. Reads full log set. Finds the real pattern."
      session_2: "Confirms connection pool exhaustion. Documents the evidence chain."
      session_3: "Discovers documentation error (parameter name wrong in docs)."
      session_4: "Writes detailed findings for Agent C's shift."
      session_5: "Files documentation correction ticket."
      session_6: "Reviews the fix Agent C applied. Confirms it addresses root cause."
      session_7: "Writes post-incident analysis with full root cause chain."

    example_utterances:
      - "Agent A diagnosed DNS caching, but the fix didn't work. I'm looking at the full 48-hour log set."
      - "Starting around 6 AM, there's a clear pattern of 'max pool size exceeded' errors. The DNS errors are a red herring."
      - "Agent A's DNS diagnosis was wrong — this is connection pool exhaustion. The DNS timeouts correlate with a scheduled refresh."
      - "The product docs say the parameter is max_pool_wait, but the config dump shows pool_max_wait. The docs have it wrong."
      - "I'll document this for Agent C. They need to know both the correct diagnosis AND that the customer was told it was DNS."
```

**Agent C — Resolution & Communication (sequence: 3rd)**

```yaml
  - user_id: agent_c
    display_name: "Agent C (Tier 2 - Resolution)"
    sequence_order: 3
    authority_level: medium
    authority_weight: 2.0
    expertise: "Fix implementation, customer communication, incident resolution, diplomacy"
    focus_areas: ["Fix implementation", "customer update drafting", "incident closure", "corrective communication"]
    biases: ["Prioritizes customer relationship", "Cautious about how corrections are communicated",
             "Wants to preserve trust while being honest"]

    communication_style: >
      Diplomatic and customer-focused. Uses phrases like "after deeper analysis",
      "our Tier 2 team identified", "we apologize for the initial misdirection."
      Balances technical accuracy with customer empathy. Writes carefully
      crafted updates that acknowledge mistakes without undermining confidence.

    document_reading_pattern: >
      Reads Agent B's findings first, then Agent A's escalation notes. Reads
      the full ticket history to understand what the customer was told. Reads
      product docs only to verify the fix parameters.

    reaction_to_corrections: >
      Very receptive. Focuses on "how do we fix this and how do we tell the
      customer?" Not interested in blame — interested in resolution.

    knowledge_gaps: ["Wasn't involved in the initial diagnosis", "Relies entirely on Agent B's findings",
                     "May not deeply understand why the connection pool fix works"]
    misconceptions: []

    emotional_tendencies: >
      Empathetic, customer-first. Slightly stressed about delivering the
      correction — "the customer was told one thing and now we're saying
      another." Focuses on preserving trust. Relieved when the fix works.

    reference_style: "Customer-facing: 'ticket #48291', 'your enterprise instance', 'the remediation steps we're applying'"

    session_evolution:
      session_1: "Reviews full investigation history. Plans the correction communication."
      session_2: "Drafts customer update that corrects the DNS diagnosis diplomatically."
      session_3: "Implements the connection pool fix (size increase + parameter change)."
      session_4: "Verifies fix. Monitors for recurrence."
      session_5: "Sends customer update. Customer acknowledges and accepts."
      session_6: "Closes incident. Writes internal summary."
      session_7: "Proposes proactive monitoring to prevent recurrence."

    example_utterances:
      - "I need to draft a customer update that corrects the earlier DNS diagnosis without undermining trust."
      - "The customer was told it was DNS caching. Now we're saying it's connection pool exhaustion. This is delicate."
      - "I'm increasing the pool size to 100 and setting pool_max_wait to 60 seconds per Agent B's recommendation."
      - "Customer confirmed 503 errors have stopped. The fix is working. I'll monitor for 24 hours before closing."
      - "For the customer update, I want to acknowledge the misdirection transparently while pivoting to the solution."
```

**Scenario 5 injected conflicts:**

```yaml
injected_conflicts:
  - conflict_id: "C5-1"
    users: ["agent_a", "agent_b"]
    topic: "Root cause diagnosis (DNS vs connection pool)"
    nature: "Agent A: DNS caching issue. Agent B: Connection pool exhaustion."
    resolution: "Temporal supersession — Agent B's later, evidence-based diagnosis overrides."
    target_sessions: [1, 2, 3]
  - conflict_id: "C5-2"
    users: ["agent_b", "agent_c"]
    topic: "Documentation parameter name (max_pool_wait vs pool_max_wait)"
    nature: "Product docs say max_pool_wait. Config shows pool_max_wait. Docs are wrong."
    resolution: "Factual supersession — config is authoritative over documentation."
    target_sessions: [3, 4, 5]

annotation_targets:
  memories_per_session: 3.5
  conflicts: 2
  eval_questions: 275
  eval_breakdown:
    user_attribution: 15
    cross_user_synthesis: 15
    conflict_resolution: 15
    information_gap: 25
    role_appropriate_briefing: 10
    adversarial_confusion: 15
    document_coverage: 15
    cross_user_provenance: 35
    temporal_correction: 30
    sequential_handoff: 100
```

---

## 7. Generation Pipeline — Detailed Implementation

### Phase 1: Document Preparation (`phase1_document_prep.py`)

Load PDFs, extract text via PyMuPDF, validate token counts within ±20%, concatenate into document context blocks with delimiters.

### Phase 2: Conversation Generation (`phase2_conversation.py`)

#### System Prompt Construction

```
SYSTEM PROMPT TEMPLATE (per user, per session):
=========================================
You are simulating a conversation between a human user and an AI assistant.

## Context
The human user ("{display_name}") is interacting with an AI assistant that has access
to the following shared documents. The AI can reference specific sections, theorems, tables, etc.

## User Persona — Core
- Name: {display_name}
- Role/Expertise: {expertise}
- Focus areas: {focus_areas}
- Known biases: {biases}
{authority_info}
{side_info}
{sequence_info}

## User Persona — Communication Style
{communication_style}

## User Persona — How They Read Documents
{document_reading_pattern}

## User Persona — How They React When Corrected
{reaction_to_corrections}

## User Persona — What They Don't Know
Knowledge gaps: {knowledge_gaps}
Misconceptions they hold: {misconceptions}

## User Persona — Emotional Tendencies
{emotional_tendencies}

## User Persona — How They Reference Documents
{reference_style}

## User Persona — Example Messages (match this voice and style)
{example_utterances}

## Temporal Context
Current date/time: {session_timestamp}
This is Session {session_number} of {total_sessions}.
The user has been working on this material since {scenario_start_date}.

## Session-Specific Behavior
{session_evolution[session_number]}

## Previous Session Summary
{previous_sessions_summary}

## Conflict Seeding (INTERNAL — do not reference directly)
The following disagreements should emerge NATURALLY from the user's persona.
Do NOT force these — let them arise from the user asking questions consistent with
their expertise and biases:
{target_conflicts_for_this_session}

## Continuity Requirements
- The user may reference things they discussed in prior sessions.
- The user's understanding should BUILD across sessions — not restart from scratch.
- Opinions and positions should evolve: the user can change their mind.
- If the user held a position in a prior session, they should reference it
  (reinforcing, updating, or correcting it).
- The user's emotional state and confidence should match the session_evolution description.

## Generation Instructions
Generate a realistic conversation of exactly {turns_per_session} turns
(1 turn = 1 user message + 1 AI response).

Rules:
1. USER messages MUST match their communication_style, reference_style, and example_utterances.
   Read the example_utterances carefully — they define the user's VOICE.
2. USER messages must reflect their knowledge_gaps and misconceptions where relevant.
   If the user has a misconception, they should state it confidently (until corrected).
3. AI ASSISTANT responses must be grounded in the provided documents with specific citations.
4. Conversations must feel natural — tangents, follow-ups, confusion, topic changes.
5. Each turn should include a simulated timestamp (advancing ~2-5 minutes per turn).
6. The user must NOT know about other users' conversations.
7. The user's reaction_to_corrections should be realistic when the AI pushes back.
8. Session-specific behavior (from session_evolution) should be reflected in the conversation arc.

## Documents
{full_document_context}

## Output Format
Output a JSON array of turns:
[
  {"turn": 1, "role": "user", "content": "...", "timestamp": "2025-04-03T14:02:00Z"},
  {"turn": 1, "role": "assistant", "content": "...", "timestamp": "2025-04-03T14:03:30Z"},
  ...
]
```

#### Generation Strategy

```python
def generate_conversations(scenario: ScenarioConfig, doc_context: DocumentContext,
                           model: str = "gpt-4o-mini") -> list[ConversationSession]:
    """
    For each user in scenario.users:
        accumulated_summary = ""
        For each session (1 to sessions_per_user):
            1. Build system prompt with persona + documents + accumulated_summary
            2. Determine which conflicts to seed (from target_sessions)
            3. Call LLM to generate full session (all turns in one API call)
            4. Parse into ConversationSession with timestamps
            5. Generate session summary (immediate — needed for next session)
            6. Append session summary to accumulated_summary
            7. Validate: turn count, alternating roles, persona consistency

    CRITICAL: Users are generated INDEPENDENTLY. Never leak cross-user info.
    CRITICAL: Session summaries feed forward — each session builds on all prior ones.
    """
```

#### Session Summary Generation (runs immediately after each session)

```python
def generate_session_summary(session: ConversationSession,
                             prior_summary: str,
                             model: str = "gpt-4o-mini") -> SessionSummary:
    """
    Generate a 150-250 word summary of this session.
    Also extract: key_facts, positions_taken, positions_changed.
    This summary is BOTH an annotation output AND input to subsequent sessions.
    """
```

### Phase 3: Annotation (`phase3_annotation.py`)

Four sub-tasks:

#### 3a. Memory Extraction

```
MEMORY EXTRACTION PROMPT:
=========================================
You are an expert memory extraction system. Given the following conversation
between a user and an AI assistant (with the source documents for grounding),
extract all distinct durable memories — facts, opinions, preferences, strategies,
decisions, assessments, and diagnoses stated by the USER.

## Rules
1. Each memory must be a single, atomic statement.
2. Tag each memory with:
   - memory_type: fact, opinion, strategy, directive, risk_flag, assessment, position, diagnosis, customer_communication
   - status: active, superseded, or corrected
   - The session_id where the memory originates
   - The session timestamp
3. Target: ~3.5 memories per session.
4. Only extract memories from the USER's statements, not the AI's.
   Exception: If the AI corrects a factual error and the user accepts the
   correction, extract BOTH the original (mark corrected) and the accepted correction (mark active).
5. Do NOT extract general knowledge — only user-specific views, preferences,
   decisions, and conclusions.
6. If a memory contradicts or updates an earlier memory from a prior session,
   mark the earlier one as superseded and link to the new one via superseded_by.

## Conversation
{conversation_json}

## Prior Sessions' Memories (for supersession detection)
{prior_memories_json}

## Documents (for grounding verification)
{document_context}

## Output Format
Return JSON array of memories following the ExtractedMemory schema.
```

#### 3b. Conflict Detection

Run AFTER all conversations for a scenario are generated. This pass looks ACROSS users.

```
CONFLICT DETECTION PROMPT:
=========================================
You are an expert conflict detection system. Given the extracted memories
from ALL users in this scenario, identify disagreements between users on
the same topic.

## Expected Conflicts (from scenario design)
{injected_conflicts}

## All Extracted Memories (from all users)
{all_memories_json}

## Rules
1. Identify all conflicts — both the pre-specified ones AND any organic
   ones that emerged naturally from persona differences.
2. For each conflict, determine:
   - conflict_type: factual, interpretive, strategic, or authority
   - resolution: preserve_both, supersede, authority_wins, or temporal_supersession
3. Map each conflict to evidence links (session-level references where positions are expressed).
4. Record first_surfaced (timestamp of earliest relevant session)
   and last_updated (timestamp of latest relevant session).
5. Include evidence links for EACH user's position — the session(s)
   where they express their view.

## Scenario Context
Relationship type: {relationship_type}
{authority_info}

## Output Format
Return JSON array following the ConflictAnnotation schema.
```

#### 3c. Evaluation Question Generation (12 categories)

```
EVAL QUESTION GENERATION PROMPT:
=========================================
Generate evaluation questions for this scenario across all applicable multi-user categories.

## Categories and Targets (generate only categories applicable to this scenario):

### Universal Categories (all scenarios):
1. USER_ATTRIBUTION ({n}): "Who said/flagged/discovered X?" Tests per-user provenance tracking.
2. CROSS_USER_SYNTHESIS ({n}): "Combining all users' analyses, what is the complete picture of X?" Tests multi-user information integration.
3. CONFLICT_RESOLUTION ({n}): "Users disagree on X. What are both positions, and what's the correct resolution given the relationship type?" Tests conflict handling.
4. INFORMATION_GAP ({n}): "What has [User A] discovered that [User B] needs but doesn't know?" Tests cross-user information asymmetry detection.
5. ROLE_APPROPRIATE_BRIEFING ({n}): "Generate a briefing for [User X] summarizing what others found, relevant to X's role." Tests role-aware information filtering.
6. ADVERSARIAL_CONFUSION ({n}): Questions that swap users/positions to test whether the system correctly attributes information. Expected answer is always "No" with correction.
7. DOCUMENT_COVERAGE ({n}): "Which document sections were reviewed by which users? What was missed?" Tests collective coverage tracking.
8. CROSS_USER_PROVENANCE ({n}): "Trace the history of [fact/decision/error] across users and sessions." Tests temporal cross-user information flow.

### Scenario-Cluster Categories (applicable scenarios only):
9. AUTHORITY_HIERARCHY ({n}): "Given the authority structure, whose position should prevail on [topic]?" Tests authority-aware resolution. (Sc.2, Sc.3, Sc.4 only)
10. TEMPORAL_CORRECTION ({n}): "What was the original claim, who corrected it, and what's the current state?" Tests cross-user correction chain tracking. (Sc.1, Sc.2, Sc.5 only)

### Scenario-Specific Categories (one scenario only):
11. ADVERSARIAL_SIDE_ISOLATION ({n}): Tests information firewalls between adversarial sides — leakage detection, position preservation, cross-side reasoning. (Sc.4 only)
12. SEQUENTIAL_HANDOFF ({n}): Tests handoff completeness, customer communication audit, and diagnosis chain accuracy across sequential agents. (Sc.5 only)

## Rules
1. Every question MUST have explicit evidence links: list of (user_id, session_id) pairs.
2. Gold answers must cite specific memory IDs and conflict IDs.
3. Adversarial confusion questions must have gold answer that corrects the misattribution.
4. Assign difficulty: easy (single fact), medium (2-3 facts/users), hard (complex multi-user reasoning).
5. Document coverage and blind spot questions require SOURCE DOCUMENT knowledge.
6. Role-appropriate briefing questions must specify the target user and their role constraints.
7. Authority & hierarchy questions must specify which users are involved and the correct authority relationship.
8. Temporal correction questions must reference the full correction chain with session timestamps.
9. Adversarial side isolation questions must test information firewalls between buyer and seller sides.
10. Sequential handoff questions must test what each agent knew at the time of their work.

## Available Data
{conversations_summary}, {memories_json}, {conflicts_json}, {document_context}
```

### Phase 4: Validation (`phase4_validation.py`)

Five validation checks:

```python
def validate_scenario(scenario_output: ScenarioOutput) -> ValidationReport:
    """
    1. CONFLICT COVERAGE — every injected conflict appears in annotations + conversations
    2. MEMORY EXTRACTABILITY — memories are grounded in source conversations, LLM-as-judge
    3. EVIDENCE VALIDITY — every eval question's session evidence links exist and are relevant
    4. QUESTION ANSWERABILITY — LLM-as-judge: answerable from memories/conflicts/conversations?
    5. PERSONA FIDELITY — LLM-as-judge: 1-5 score per session for persona consistency
    """
```

---

## 8. Evaluation Harness (`eval/`)

### 8.1 Evaluation Protocol

A memory system under test receives all conversations for a scenario plus source documents. The system processes them and answers the eval questions using whatever internal representation it wants.

### 8.2 Baselines (in `eval/baselines/`)

**Baseline 1 — Full Context**: Concatenate all conversations, pass to LLM with question. "No memory system" baseline.

**Baseline 2 — RAG over Session Summaries**: Use gold session summaries as retrieval database. Embed, retrieve top-k, pass to LLM.

### 8.3 Metrics

**QA metrics** (per question category):
- **User Attribution**: Exact-match accuracy (is the correct user identified?)
- **Cross-User Synthesis**: FactScore (atomic fact precision/recall) + LLM-as-judge completeness
- **Conflict Resolution**: LLM-as-judge on both-sides fairness + resolution correctness
- **Information Gap**: LLM-as-judge on whether identified gaps are genuine and actionable
- **Role-Appropriate Briefing**: LLM-as-judge on relevance, role-appropriateness, and information leakage (adversarial only)
- **Adversarial Confusion**: Binary accuracy (correct rejection + proper correction)
- **Document Coverage**: P/R on (user, document_section) pairs + recall on blind spots
- **Cross-User Provenance**: LLM-as-judge on chain completeness and temporal accuracy
- **Authority & Hierarchy**: LLM-as-judge on correct authority identification and application
- **Temporal Correction**: LLM-as-judge on chain completeness, temporal accuracy, and current-state correctness
- **Adversarial Side Isolation**: LLM-as-judge on information leakage (binary) + position accuracy per side
- **Sequential Handoff**: LLM-as-judge on handoff completeness + customer communication accuracy + temporal ordering

**Memory extraction**: P/R/F1 at memory level (LLM-as-judge semantic matching).

**Conflict detection**: P/R/F1 at conflict level.

**Session summary**: ROUGE-1/2/L + FactScore via LLM-as-judge.

---

## 9. CLI Interface

```bash
python -m src.cli generate --scenario 1 --model gpt-4o-mini
python -m src.cli generate --all --model gpt-4o-mini
python -m src.cli validate-docs --scenario 1
python -m src.cli generate-conversations --scenario 1 --model gpt-4o-mini
python -m src.cli annotate --scenario 1 --model gpt-4o
python -m src.cli validate --scenario 1 --model gpt-4o
python -m src.cli export --format huggingface
python -m eval.run_eval --scenario 1 --baseline full_context --judge-model gpt-4o
python -m src.cli estimate-cost --all
python -m src.cli generate --scenario 1 --dry-run
```

---

## 10. Iteration Strategy

### Iteration 1: Structural (GPT-4o-mini) — 1 scenario, all users, all sessions
### Iteration 2: Quality (GPT-4o-mini) — full annotation + validation on Scenario 1
### Iteration 3: Scale (GPT-4o-mini) — all 5 scenarios
### Iteration 4: Final (GPT-4.1) — final gen + annotation with GPT-4o + human spot-check

---

## 11. Model Usage Strategy

| Phase | Task | Model |
|-------|------|-------|
| Iteration 1-3 | Conversation generation | gpt-4o-mini |
| Iteration 4 | Conversation generation | gpt-4.1 |
| Phase 3 | Session summaries | gpt-4o-mini |
| Phase 3 | Memory extraction | gpt-4o |
| Phase 3 | Conflict detection | gpt-4o |
| Phase 3 | Eval question generation | gpt-4o |
| Phase 4 | LLM-as-judge validation | gpt-4o |
| Eval | Baseline scoring | gpt-4o |

---

## 12. Cost Analysis

| Scenario | Users | Sessions | Turns | Doc Tok | Total Input |
|----------|-------|----------|-------|---------|-------------|
| 1 | 4 | 28 | 504 | 15K | ~8.0M |
| 2 | 4 | 28 | 448 | 22K | ~10.2M |
| 3 | 4 | 28 | 504 | 18K | ~9.5M |
| 4 | 4 | 28 | 504 | 20K | ~10.5M |
| 5 | 3 | 21 | 378 | 12K | ~4.8M |
| **Total** | | **133** | **~2,340** | | **~43M** |

**Estimated costs (single pass, conversation generation only)**:
- GPT-4o-mini: ~$8
- GPT-4.1: ~$105
- With prompt caching (~50% saving): GPT-4.1 drops to ~$55

**Recommended total budget**: ~$70–90 (iterations + final + annotation + caching).

---

## 13. Implementation Order for Coding Agent

### Step 1: Project Scaffolding
- Create full repo structure (all directories, `__init__.py` files, `eval/` directory)
- Set up `pyproject.toml` with all dependencies
- Create `.env.example`, `Makefile` with common commands
- Write `README.md` skeleton with project description and quickstart placeholder

### Step 2: Data Models
- Implement ALL enums in `enums.py`: RelationshipType, AuthorityLevel, ConflictType, ConflictResolution, MemoryType, MemoryStatus, EvalQuestionCategory (12 categories including adversarial_side_isolation and sequential_handoff)
- Implement ALL schemas in `schemas.py`: UserProfile (with 10 roleplay fields), ConversationTurn (no dia_id), ConversationSession, SessionSummary, ExtractedMemory (session-level, no turn-level fields), EvidenceLink (session-level only), ConflictAnnotation, EvalQuestion, ScenarioTimeline, ScenarioConfig, DocumentConfig, InjectedConflict, AnnotationTargets, ValidationReport, GenerationReport, ScenarioOutput, BenchmarkDataset
- Write tests for schema validation and serialization
- Implement JSON/YAML read/write utilities in `utils/io.py`

### Step 3: LLM Client
- Build `llm/client.py` with OpenAI wrapper, exponential backoff retry, rate limiting
- Build `llm/token_counter.py` using tiktoken (cl100k_base encoding)
- Build `llm/cost_tracker.py` — per-call and cumulative cost tracking, per-phase breakdown
- Support model switching via `config/models.yaml`

### Step 4: Document Preparation (Phase 1)
- Implement `utils/pdf_reader.py` using PyMuPDF (fitz)
- Implement `pipeline/phase1_document_prep.py`: load PDFs, extract text, count tokens, validate ±20%, concatenate with delimiters
- Write `scripts/validate_documents.py`: check all 15 PDFs exist and meet token targets
- Test with actual scenario documents

### Step 5: Scenario Configs
- Create all 5 scenario YAML files with: revised 7-session counts, timelines, full roleplay personas (all 10 fields per user including communication_style, session_evolution, example_utterances), documents, injected conflicts with target_sessions, session_arc descriptions, annotation targets with 12-category eval breakdowns
- Implement `scenarios/base.py` (BaseScenario ABC) and all 5 scenario classes
- Implement config loading and validation

### Step 6: Prompt Engineering
- Implement all 7 prompt modules in `src/prompts/`: conversation_system, conversation_user, session_summary, memory_extraction, conflict_detection, eval_question_gen, quality_check
- The conversation system prompt must inject ALL roleplay fields into named sections
- Include example conversations as few-shot examples where relevant
- Test prompt construction with mock data — verify all template variables are filled

### Step 7: Conversation Generation + Live Session Summary Feed-Forward
- Implement `pipeline/phase2_conversation.py`
- Session summary generation runs immediately after each session and feeds into the next
- Start with ONE user, ONE scenario, ALL 7 sessions — test that continuity works across sessions
- Validate: does the user's voice match example_utterances? Does session_evolution behavior shift correctly?
- Scale to all users/scenarios

### Step 8: Memory Extraction + Conflict Detection
- Implement memory extraction with MemoryStatus tracking and supersession chains (session-level attribution)
- Implement cross-user conflict detection with session-level evidence links, timestamps, and resolution typing
- Run memory extraction per-user (all sessions), then conflict detection per-scenario (all users)
- Validate against expected memories and conflicts from scenario design

### Step 9: 12-Category Multi-User Eval Question Generation with Evidence Grounding
- Implement eval question generation with all 12 categories and per-scenario distribution targets
- Universal categories (1–8) generated for all scenarios
- Scenario-cluster categories (9–10) generated only for applicable scenarios
- Scenario-specific categories (11–12) generated only for their respective scenarios
- Every question must have session-level evidence links, required memory/conflict IDs, and difficulty rating
- Adversarial confusion questions must swap user attributions and expect "No" answers
- Authority & hierarchy questions must specify the correct authority relationship
- Temporal correction questions must reference the full correction chain
- Adversarial side isolation questions must test information firewalls (leakage, position preservation, cross-side reasoning)
- Sequential handoff questions must test handoff completeness, customer communication audit, and diagnosis chain
- Role-appropriate briefing questions must specify target user and role constraints
- Validate: all evidence links resolve to real sessions, questions are answerable from annotations

### Step 10: 5-Point Validation Pipeline
- Implement all 5 validation checks in `pipeline/phase4_validation.py`
- Build ValidationReport with per-check scores and flagged issues
- Implement regeneration triggers (e.g., missing conflicts → regenerate affected sessions)

### Step 11: Evaluation Harness + 2 Baselines + Metrics
- Implement `eval/run_eval.py`: feeds questions to a memory system, collects answers, scores them
- Implement `eval/metrics.py`: exact-match accuracy, FactScore, LLM-as-judge, ROUGE, P/R/F1
- Implement 2 baselines: full_context (concatenate all conversations), rag_summaries (retrieve from session summaries)
- Each baseline should be runnable as a standalone script

### Step 12: Orchestrator & CLI
- Implement `pipeline/orchestrator.py` — runs Phase 1 → 2 → 3 → 4 in sequence for a scenario
- Implement `cli.py` with all commands: generate, generate-conversations, annotate, validate, validate-docs, export, estimate-cost, dry-run
- Implement eval commands: `eval.run_eval` with --scenario, --baseline, --judge-model flags
- Implement `scripts/dry_run.py` (mock LLM, test full pipeline) and `scripts/cost_estimate.py`

### Step 13: Export & Packaging
- Implement `scripts/export_hf.py` (HuggingFace datasets format)
- Generate per-scenario bundled JSON (`scenario_X_complete.json`)
- Generate combined `benchmark.json` and `generation_report.json`
- Finalize `README.md` with full usage instructions, benchmark description, citation
- Write `docs/EVALUATION_GUIDE.md` explaining how to evaluate a memory system against the benchmark

---

## 14. Key Design Decisions & Constraints

### CRITICAL: Session-Level Annotation (No Per-Turn References)
All annotations (memories, evidence links, eval question evidence) reference sessions, not individual turns. The universal reference key is `(scenario_id, user_id, session_id)`. There is no dia_id system.

### CRITICAL: Session Summaries Feed Forward
Session summaries are generated DURING Phase 2 (after each session), not just in Phase 3. Each user's Session N+1 prompt includes the accumulated summary of Sessions 1–N.

### CRITICAL: Temporal Consistency
Every turn has a simulated timestamp. Every session has a session-level timestamp. Temporal questions rely on these being consistent.

### CRITICAL: User Isolation During Generation
Users NEVER see each other's conversations during Phase 2. Cross-user knowledge only in Phases 3–4.

### CRITICAL: Conflict Naturalness
Conflicts emerge from persona differences, not force-injection.

### CRITICAL: Structured Output
All LLM calls use response_format: { type: "json_object" }. Try/catch with retry.

### CRITICAL: Reproducibility
temperature=0.7 for conversations, 0.2 for annotation. Log everything.

---

## 15. Dependencies

```toml
[project]
dependencies = [
    "openai>=1.30.0",
    "pydantic>=2.0",
    "tiktoken>=0.7.0",
    "pymupdf>=1.24.0",
    "pyyaml>=6.0",
    "click>=8.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
    "tenacity>=8.0",
    "rouge-score>=0.1.2",
    "numpy>=1.24.0",
]
```

---

## 16. Prompt Caching Strategy

Generate all sessions for User A before User B (same document prefix). Generate all users for Scenario 1 before Scenario 2 (same document set). Session-level generation (full session in one API call).

---

## Summary Checklist

- [ ] Repo scaffolding with eval/ directory
- [ ] All Pydantic schemas (session-level: no dia_id, no turn-level memory fields, session-level EvidenceLink)
- [ ] 12-category EvalQuestionCategory enum (8 universal + 2 scenario-cluster + 2 scenario-specific)
- [ ] LLM client with retry, rate limiting, cost tracking
- [ ] PDF reader and document preparation
- [ ] All 5 scenario YAML configs with revised session/turn counts, timelines, detailed roleplay personas, and 12-category eval breakdowns
- [ ] All 7 prompt template modules
- [ ] Conversation generation with live session summary feed-forward
- [ ] Session summary generation pipeline
- [ ] Memory extraction with MemoryStatus tracking (session-level attribution)
- [ ] Conflict detection with session-level evidence links and timestamps
- [ ] 12-category multi-user eval question generation (1,200 total) with session-level evidence grounding
- [ ] 5-point validation pipeline
- [ ] Evaluation harness with 2 baselines
- [ ] Metrics: F1, LLM-as-judge, ROUGE, FactScore
- [ ] CLI with all commands (generate, annotate, validate, eval)
- [ ] Export to HuggingFace format
- [ ] Cost estimation + dry run scripts
- [ ] README + EVALUATION_GUIDE.md
- [ ] Test suite
- [ ] Generation report output