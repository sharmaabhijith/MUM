# MUM: Multi-User Memory Benchmark

A benchmark generation pipeline for evaluating AI memory systems in multi-user settings. MUM extends the LoCoMo paradigm (long multi-session conversations) into a **multi-user setting**: multiple users independently interact with an AI assistant over shared documents, and the benchmark evaluates whether a system can extract, manage, and reconcile knowledge across these parallel conversation streams.

**What makes this different from LoCoMo**: LoCoMo tests single-thread memory (2 speakers, 1 conversation, ~300 turns, ~19 sessions). MUM tests **cross-user** memory: 4 users per scenario, each with their own long conversation history, producing conflicting facts, authority gradients, and temporal dependencies that a memory system must reconcile.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [The 5 Scenarios](#the-5-scenarios)
- [3-Phase Generation Pipeline](#3-phase-generation-pipeline)
- [12 Evaluation Categories](#12-evaluation-categories)
- [Configuration](#configuration)
- [Data Schemas](#data-schemas)
- [Development](#development)

---

## Installation

```bash
# Clone and install
git clone <repo-url>
cd MUM
pip install -e ".[dev]"

# Set up API key
echo "DEEPINFRA_API_KEY=your_key_here" > .env
```

Requires Python >= 3.10.

## Quick Start

```bash
# Validate documents exist and meet token targets
mum validate-docs

# Estimate API cost before running
mum estimate-cost --all

# Dry run (mock LLM, no API calls)
mum generate --scenario 1 --dry-run

# Generate a single scenario
mum generate --scenario 1

# Generate all 5 scenarios
mum generate --all

# Generate conversations only (skip question generation)
mum generate-conversations --scenario 1

# Export to HuggingFace datasets format
mum export --format huggingface
```

Or via `make`:

```bash
make install           # pip install -e ".[dev]"
make dry-run           # scenario 1, mock LLM
make generate-scenario SCENARIO=1
make generate-all
make cost-estimate
make test
make export
```

---

## Project Structure

```
MUM/
├── config/
│   ├── models.yaml                  # Model selections per mode
│   └── scenarios/
│       ├── scenario_1.yaml          # Study Group (symmetric)
│       ├── scenario_2.yaml          # Municipal Committee (hierarchical)
│       ├── scenario_3.yaml          # Product Strategy (cross-functional)
│       ├── scenario_4.yaml          # M&A Negotiation (adversarial)
│       └── scenario_5.yaml         # Incident Response (sequential)
│
├── documents/
│   ├── scenario_1/                  # 7 Ficlandia PDFs
│   ├── scenario_2/                  # 6 municipal governance PDFs
│   ├── scenario_3/                  # 7 product strategy PDFs
│   ├── scenario_4/                  # 8 M&A deal PDFs
│   └── scenario_5/                  # 5 incident log PDFs
│
├── src/
│   ├── cli.py                       # Click CLI entry point
│   ├── llm/
│   │   ├── client.py                # LLMClient (OpenAI-compatible, DeepInfra)
│   │   ├── cost_tracker.py          # Per-call cost and token tracking
│   │   └── token_counter.py         # tiktoken-based counter
│   ├── models/
│   │   ├── enums.py                 # RelationshipType, AuthorityLevel, EvalQuestionCategory
│   │   └── schemas.py              # Pydantic models (UserProfile, ScenarioConfig, etc.)
│   ├── pipeline/
│   │   ├── phase1_document_prep.py  # PDF ingestion → DocumentContext
│   │   ├── phase2_conversation.py   # Conversation + session summary generation
│   │   ├── phase3_annotation.py     # Evaluation question generation
│   │   └── orchestrator.py          # Top-level pipeline runner
│   ├── prompts/
│   │   ├── base.py                  # PromptBuilder helpers
│   │   ├── conversation_system.py   # System prompt builder
│   │   ├── conversation_user.py     # User-turn trigger prompt
│   │   ├── session_summary.py       # Session summary prompt
│   │   └── eval_question_gen.py     # Eval question prompt + category instructions
│   ├── scenarios/
│   │   ├── base.py                  # BaseScenario ABC
│   │   └── scenarios.py             # 5 concrete scenario classes
│   └── utils/
│       ├── pdf_reader.py            # PyMuPDF text extraction
│       ├── io.py                    # JSON/YAML read/write helpers
│       └── logging.py              # Logger setup
│
├── scripts/
│   ├── cost_estimate.py             # Standalone cost estimation
│   ├── dry_run.py                   # Quick smoke test
│   ├── export_hf.py                 # HuggingFace datasets export
│   └── validate_documents.py        # PDF token validation
│
├── tests/
│   ├── test_schemas.py              # Pydantic model tests
│   ├── test_scenarios.py            # Scenario loading tests
│   ├── test_prompts.py              # Prompt builder tests
│   └── test_pipeline.py             # Pipeline integration tests
│
└── output/                          # Generated benchmark data
```

---

## The 5 Scenarios

Each scenario uses a **fictional setting** so all facts must come from provided documents, not LLM general knowledge.

| # | Name | Relationship | Domain | Users | Documents |
|---|------|-------------|--------|-------|-----------|
| 1 | Study Group: Ficlandia | Symmetric | Education / Area Studies | 4 students | 7 PDFs |
| 2 | Municipal Committee: FicTown | Hierarchical | Municipal Governance | 4 officials (Commissioner > Officers > Councillor) | 6 PDFs |
| 3 | Product Strategy: Ficduct | Cross-Functional | Technology / Product | 4 executives (CTO, CFO, CMO, VP Product) | 7 PDFs |
| 4 | M&A Negotiation: Project Ficguard | Adversarial | Finance / M&A | 4 (2 buyer + 2 seller, never merged) | 8 PDFs |
| 5 | Incident Response: Cascading Payments | Sequential | Technology / SRE | 4 analysts across shifts A→B→C→D | 5 PDFs |

Each scenario generates **4 users x 7 sessions = 28 sessions**. Sessions 1-2 have 50 turns; sessions 3-7 have 20 turns.

---

## 3-Phase Generation Pipeline

### Phase 1 — Document Preparation
- Extracts text from scenario PDFs using PyMuPDF
- Counts tokens (tiktoken, `cl100k_base` encoding)
- Validates token counts are within ±20% of YAML targets
- Produces a `DocumentContext` with concatenated text for prompt injection

### Phase 2 — Conversation Generation
- For each user × session, constructs a system prompt containing: full document text, detailed persona, authority context, session arc, prior session summaries, and conflict seeding instructions
- Calls the LLM to generate a multi-turn conversation as structured JSON
- After each session, generates a **session summary** (key facts, positions taken, changes) that feeds forward as context for subsequent sessions
- Supports **resume**: skips sessions already saved to disk
- Model: `deepseek-ai/DeepSeek-V3.2` via DeepInfra

### Phase 3 — Evaluation Question Generation
- For each applicable evaluation category, calls the LLM with all conversations, summaries, and document context
- Generates structured `EvalQuestion` objects with question, gold answer, evidence links, and difficulty rating
- Model: `google/gemini-2.5-pro` via DeepInfra

---

## 12 Evaluation Categories

**8 Universal** (all scenarios):

| Category | Tests |
|----------|-------|
| User Attribution | Who said X? |
| Cross-User Synthesis | Combine knowledge across users |
| Conflict Resolution | Identify and resolve disagreements |
| Information Gap | What does user A know that user B needs? |
| Role-Appropriate Briefing | Tailor information to user's role |
| Adversarial Confusion | Misattribution resistance (swapped users) |
| Document Coverage | Which sections were covered by which users? |
| Cross-User Provenance | Trace a fact's history across users/sessions |

**4 Scenario-Specific**:

| Category | Applicable Scenarios |
|----------|---------------------|
| Authority Hierarchy | 2 (hierarchical), 3 (cross-functional), 4 (adversarial) |
| Temporal Correction | 1 (symmetric), 2 (hierarchical), 5 (sequential) |
| Adversarial Side Isolation | 4 (adversarial) |
| Sequential Handoff | 5 (sequential) |

---

## Configuration

### Models (`config/models.yaml`)

```yaml
iteration:
  conversation: deepseek-ai/DeepSeek-V3.2
  summary: deepseek-ai/DeepSeek-V3.2
  question_gen: google/gemini-2.5-pro

final:
  conversation: deepseek-ai/DeepSeek-V3.2
  summary: deepseek-ai/DeepSeek-V3.2
  question_gen: google/gemini-2.5-pro
```

All API calls route through **DeepInfra** (OpenAI-compatible). Requires `DEEPINFRA_API_KEY` in `.env`.

### Scenario Configs (`config/scenarios/scenario_N.yaml`)

Each YAML defines:
- **Users**: detailed personas with expertise, biases, communication style, document reading patterns, knowledge gaps, misconceptions, and per-session evolution arcs
- **Documents**: list of PDFs with target token counts
- **Timeline**: per-user session schedule with dates
- **Conflicts**: injected disagreements with specified participants, topics, and resolution approaches
- **Annotation targets**: number of evaluation questions per category

---

## Data Schemas

Key Pydantic models in `src/models/schemas.py`:

- **`UserProfile`** — persona: expertise, biases, communication style, document reading pattern, reaction to corrections, knowledge gaps, misconceptions, example utterances, per-session evolution notes
- **`ScenarioConfig`** — top-level config: relationship type, users, documents, sessions, timeline, injected conflicts, annotation targets
- **`InjectedConflict`** — planted disagreement: participants, topic, nature, resolution approach, target sessions
- **`ConversationTurn` / `ConversationSession`** — raw conversation data
- **`SessionSummary`** — summary, key facts, positions taken/changed
- **`EvalQuestion`** — category, question, gold answer, evidence links, difficulty
- **`ScenarioOutput` / `BenchmarkDataset`** — full output containers with generation report

---

## Output Structure

```
output/
├── benchmark.json                               # Full BenchmarkDataset
├── generation_report.json                       # Cost + tokens summary
└── scenario_N/
    ├── conversations/
    │   └── {user_id}_session_{n}.json           # Per user per session
    ├── summaries/
    │   └── {user_id}_session_{n}_summary.json   # Session summary
    ├── evaluation/
    │   └── eval_questions.json                  # All eval questions
    └── scenario_N_complete.json                 # Full ScenarioOutput
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Clean generated output
make clean
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI-compatible client (DeepInfra API) |
| `pydantic` | Data validation and schemas |
| `tiktoken` | Token counting (`cl100k_base`) |
| `pymupdf` | PDF text extraction |
| `pyyaml` | YAML config parsing |
| `click` | CLI framework |
| `rich` | Console formatting |
| `python-dotenv` | `.env` loading |
| `tenacity` | Retry with exponential backoff |
| `rouge-score` | Evaluation metrics |
| `numpy` | Numerical operations |

Optional: `datasets` (HuggingFace export), `pytest` + `ruff` (dev).

---

## License

MIT
