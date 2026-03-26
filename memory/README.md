# Memory Management & Evaluation Module

This module implements **five memory management strategies** and an **end-to-end evaluation pipeline** for the MUM (Multi-User Memory) Benchmark. It measures how well different memory representations enable LLMs to answer questions that require reasoning across multiple users' parallel conversation histories.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Memory Methods](#memory-methods)
  - [BaseMemoryMethod](#basememorymethod)
  - [NoMemory (Baseline)](#1-nomemory--baseline)
  - [FullContext (Oracle)](#2-fullcontext--oracle-upper-bound)
  - [SummaryMemory](#3-summarymemory)
  - [RAGMemory](#4-ragmemory)
  - [StructuredMemory](#5-structuredmemory)
- [Evaluation Pipeline](#evaluation-pipeline)
  - [EvaluationRunner](#evaluationrunner)
  - [MultiModelEvaluator](#multimodelevaluator)
  - [LLM-as-Judge](#llm-as-judge-scoring)
  - [Reporting](#reporting)
- [Prompts](#prompts)
  - [Answer Generation](#answer-generation-prompt)
  - [Judge Prompts](#judge-prompts)
- [CLI Reference](#cli-reference)
- [Data Schemas](#data-schemas)
- [Evaluation Grid](#evaluation-grid)
- [Output Files](#output-files)

---

## Architecture Overview

```
memory/
├── __init__.py
├── cli.py                          # CLI entry point (mum-eval)
├── methods/                        # Memory management strategies
│   ├── __init__.py                 # METHODS registry
│   ├── base.py                     # BaseMemoryMethod ABC + MemoryContext
│   ├── no_memory.py                # Baseline: no context
│   ├── full_context.py             # Oracle: all conversations
│   ├── summary_memory.py           # Session summaries as context
│   ├── rag.py                      # ChromaDB-based retrieval
│   └── structured_memory.py        # Multi-user partitioned memory
├── evaluation/                     # Evaluation pipeline
│   ├── __init__.py
│   ├── runner.py                   # EvaluationRunner + MultiModelEvaluator
│   ├── judge.py                    # LLMJudge (dimensional + binary scoring)
│   └── report.py                   # Comparison tables + JSON export
└── prompts/                        # LLM prompts
    ├── __init__.py
    ├── answer_gen.py               # System prompt for answer generation
    └── judge.py                    # Judge prompts with category-specific guidance
```

### Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│ MUMBench/scenario_N/                                              │
│   conversations/*.json  +  summaries/*.json  +  eval_questions    │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Memory Method      │
                     │  .ingest()          │
                     │  .retrieve(q)       │
                     └──────────┬──────────┘
                                │ MemoryContext
                     ┌──────────▼──────────┐
                     │  Answer LLM         │
                     │  (generates answer  │
                     │   from context + q) │
                     └──────────┬──────────┘
                                │ predicted_answer
                     ┌──────────▼──────────┐
                     │  LLM Judge          │
                     │  (scores against    │
                     │   gold answer)      │
                     └──────────┬──────────┘
                                │ JudgeScore
                     ┌──────────▼──────────┐
                     │  Results saved to   │
                     │  MUMBench/results/  │
                     └─────────────────────┘
```

---

## Memory Methods

All methods inherit from `BaseMemoryMethod` and implement a three-phase lifecycle:

1. **`ingest(conversations, summaries)`** — Process all conversations for a scenario
2. **`retrieve(question, user_id=None)`** — Return relevant context as a `MemoryContext`
3. **`reset()`** — Clear state between scenarios

### BaseMemoryMethod

Defined in `methods/base.py`. The abstract base class that all methods extend.

**`MemoryContext`** — the return type of `retrieve()`:
```python
@dataclass
class MemoryContext:
    method_name: str       # e.g., "rag", "structured"
    context_text: str      # Plain text context injected into the LLM prompt
    token_count: int       # Accurate count via tiktoken (cl100k_base)
    metadata: dict         # Method-specific stats (num_chunks, retrieval_signals, etc.)
```

---

### 1. NoMemory — Baseline

**File:** `methods/no_memory.py`

The zero-shot baseline. No conversation history is provided — the LLM answers from parametric knowledge only (which should fail on fictional scenarios).

- **`ingest()`**: No-op
- **`retrieve()`**: Returns empty context (`""`, 0 tokens)
- **Purpose**: Establishes the performance floor

---

### 2. FullContext — Oracle Upper Bound

**File:** `methods/full_context.py`

All conversations concatenated verbatim. This is the oracle upper bound — everything is visible.

- **`ingest()`**: Stores conversations sorted by `(user_id, session_number)`
- **`retrieve()`**: Returns all conversations formatted as:
  ```
  === User: student_a | Session 3 (2025-03-15T14:00:00) ===
  [user]: How does Ficlandia's economy work?
  [assistant]: Based on the Economics document...
  ```
- **Metadata**: `num_conversations`, `num_turns`
- **Purpose**: Establishes the performance ceiling (what's theoretically achievable with perfect recall)

---

### 3. SummaryMemory

**File:** `methods/summary_memory.py`

Uses pre-generated session summaries instead of raw conversation text — a compression approach.

- **`ingest()`**: Stores summaries sorted by `(user_id, session_number)`
- **`retrieve()`**: Formats each summary with structured fields:
  ```
  === User: student_a | Session 3 ===
  Summary: Student_a explored economic policies and trade relationships...
  Key Facts:
    - Ficlandia's GDP is $45B with 60% from services
    - Main trading partners: Nordavia and Eastholm
  Positions:
    - Believes trade liberalization would benefit growth
  Positions Changed:
    - Revised earlier view on tariff effectiveness after seeing data
  ```
- **Supports**: Optional `user_id` scoping to filter by user
- **Metadata**: `num_summaries`, `scoped_to_user`

---

### 4. RAGMemory

**File:** `methods/rag.py`

Embedding-based retrieval using ChromaDB. Chunks conversations and summaries, then retrieves the most relevant chunks for each question.

- **`ingest()`**:
  1. Chunks conversations into ~512-token pieces with 64-token overlap (configurable)
  2. Chunks summaries similarly
  3. Batch upserts into a ChromaDB collection
  4. Each chunk gets metadata: `user_id`, `session_number`, `source_type`

- **`retrieve()`**:
  1. Queries ChromaDB with the question text (cosine similarity)
  2. Returns top-K chunks (default: 20)
  3. Each chunk includes source attribution:
    ```
    --- [conversation] User: student_a Session: 3 ---
    [user]: What are the main economic sectors?
    [assistant]: According to the Economics document...
    ```

- **Configuration**: `top_k=20`, `chunk_size=512`, `chunk_overlap=64`, optional `persist_dir`
- **Metadata**: `num_chunks_retrieved`, `top_k`, `chunk_size`

---

### 5. StructuredMemory

**File:** `methods/structured_memory.py`

The most sophisticated method. Builds a multi-user partitioned memory store with explicit fact structures and question-aware retrieval.

**Data Model:**
```python
@dataclass
class UserMemoryStore:
    user_id: str
    facts: list[dict]          # Key facts extracted from summaries
    positions: list[dict]      # Positions/opinions taken
    corrections: list[dict]    # Temporal corrections and changes
```

**Ingestion:**
1. Extracts and partitions facts, positions, and corrections by user
2. Builds a cross-user fact index (facts appearing across multiple users)
3. Groups corrections for temporal tracking

**Retrieval — Question-Aware:**

Analyzes each question for signal keywords to determine what context to assemble:

| Signal | Keywords | Triggers |
|--------|----------|----------|
| `is_attribution` | "who", "which user" | Per-user facts |
| `is_cross_user` | "combining", "across", "complete picture" | Cross-user section |
| `is_correction` | "correct", "error", "change", "update" | Correction timeline |
| `is_conflict` | "disagree", "conflict", "contradict" | Cross-user section |
| `is_gap` | "gap", "missing", "doesn't know" | Cross-user section |
| `is_handoff` | "handoff", "shift", "transfer" | Correction timeline |

**Output Format:**
```
=== User: student_a ===
[Facts]
  (S3) Ficlandia's GDP is $45B
  (S5) Main exports are technology and fish
[Positions]
  (S3) Supports trade liberalization
[Corrections/Changes]
  (S5) Revised GDP estimate from $40B to $45B after reviewing new data

=== Cross-User Shared Facts ===
  [student_a, student_c] Both discuss Ficlandia's trade relationships

=== Correction/Change Timeline ===
  student_a (S2): Initially stated GDP was $40B
  student_c (S4): Corrected to $45B citing Economics document
  student_a (S5): Accepted correction
```

- **Metadata**: `num_users`, `retrieval_signals`, `scoped_to_user`

---

## Evaluation Pipeline

### EvaluationRunner

**File:** `evaluation/runner.py` (class `EvaluationRunner`)

Runs evaluation for a single model + method + scenario combination.

**Lifecycle:**
```python
runner = EvaluationRunner(
    method=StructuredMemory(),
    answer_model="Qwen/Qwen3-14B",
    judge_model="google/gemini-2.5-pro",
)
result = runner.run_scenario("1", resume=True)
```

**For each evaluation question:**
1. Retrieve memory context via `method.retrieve(question)`
2. Build answer prompt with context
3. Call answer LLM to generate predicted answer
4. Judge predicted answer against gold answer
5. Save incrementally (every 5 questions)

**Resume Support:**
- Loads completed question IDs from existing result files
- Reloads prior `QuestionResult` objects
- Continues from the next incomplete question
- Provides fault tolerance for network failures or API rate limits

---

### MultiModelEvaluator

**File:** `evaluation/runner.py` (class `MultiModelEvaluator`)

Orchestrates the full evaluation grid across all models, methods, and scenarios.

**Default Evaluation Grid:**
- **7 answer models** (via DeepInfra):
  - Qwen/Qwen3-14B
  - Qwen/Qwen3-32B
  - Qwen/Qwen3-Next-80B-A3B-Instruct
  - meta-llama/Meta-Llama-3.1-8B-Instruct
  - meta-llama/Meta-Llama-3.1-70B-Instruct
  - deepseek-ai/DeepSeek-V3.2
  - google/gemini-2.5-flash
- **5 memory methods**: no_memory, full_context, rag, summary, structured
- **5 scenarios**: 1 through 5
- **1 judge model**: google/gemini-2.5-pro
- **Total: 7 x 5 x 5 = 175 evaluation runs**

---

### LLM-as-Judge Scoring

**File:** `evaluation/judge.py`

The `LLMJudge` class scores predicted answers using an LLM. Two scoring modes:

#### Dimensional Scoring (default)

Scores on 4 dimensions, each from 1 to 5:

| Dimension | Description | Weight |
|-----------|-------------|--------|
| **Correctness** | Are core facts correct? (names, numbers, conclusions) | 0.35 |
| **Completeness** | Are ALL key points from the gold answer covered? | 0.25 |
| **Attribution** | Is information attributed to the correct users? | 0.25 |
| **Hallucination** | Does it avoid fabricating facts? (5=none, 1=severe) | 0.15 |

**Overall score (0-1 scale):**
```
overall = (0.35 * correctness + 0.25 * completeness + 0.25 * attribution + 0.15 * hallucination) / 5.0
```

#### Binary Scoring (for `adversarial_confusion` category)

Tests whether the system resists deliberate misattribution:
- Question: "Did [WRONG_USER] do X?"
- Correct: Rejects the false attribution and identifies the real user
- Score: `1.0` (correct) or `0.0` (incorrect)

---

### Reporting

**File:** `evaluation/report.py`

Generates formatted comparison tables and structured JSON exports.

**Report types:**

1. **Scenario Report** (`print_scenario_report`): Detailed model x method grid for one scenario, with category breakdowns
2. **Comparison Report** (`print_comparison_report`): Cross-scenario averages, model summaries, method summaries, and dimension averages
3. **JSON Export** (`save_comparison_report`): Structured `comparison_report.json` with per-model, per-method, per-scenario breakdowns

---

## Prompts

### Answer Generation Prompt

**File:** `prompts/answer_gen.py`

The system prompt instructs the answer LLM to:
- Use ONLY the provided memory context (no general knowledge)
- Always attribute information to specific users
- Distinguish between different users' statements and beliefs
- State both positions when users disagree, with evidence-based resolution
- Track temporal evolution of beliefs across sessions
- Say "insufficient information" rather than guess

The user message includes the memory context, method name, and evaluation question.

### Judge Prompts

**File:** `prompts/judge.py`

Contains **category-specific judge guidance** for all 12 evaluation categories. Each category emphasizes different scoring priorities:

| Category | Critical Dimensions |
|----------|-------------------|
| user_attribution | Attribution |
| cross_user_synthesis | Completeness |
| conflict_resolution | Correctness + Attribution |
| information_gap | Correctness + Completeness |
| role_appropriate_briefing | Completeness + Correctness |
| document_coverage | Attribution + Completeness |
| cross_user_provenance | Correctness + Completeness |
| authority_hierarchy | Correctness |
| temporal_correction | Correctness + Completeness |
| adversarial_side_isolation | Correctness + Attribution |
| sequential_handoff | Completeness + Correctness |
| adversarial_confusion | Binary (correct/incorrect) |

---

## CLI Reference

The CLI is registered as `mum-eval` (defined in `cli.py`).

### Commands

#### `evaluate` — Run evaluation grid

```bash
# Single scenario, single method, single model
mum-eval evaluate --scenario 1 --method rag --answer-model Qwen3-14B

# All scenarios, all methods, all models (full 175-run grid)
mum-eval evaluate --all-scenarios --method all --answer-model all

# Multiple models, comma-separated
mum-eval evaluate --scenario 1 --method structured --answer-model "Qwen3-14B,Llama3.1-8B"

# Custom judge model and RAG parameters
mum-eval evaluate --scenario 1 --method rag --answer-model Qwen3-14B \
  --judge-model google/gemini-2.5-pro --top-k 20 --chunk-size 512

# Disable resume (start fresh)
mum-eval evaluate --scenario 1 --method rag --answer-model Qwen3-14B --no-resume
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--scenario / -s` | — | Scenario ID (1-5) |
| `--all-scenarios` | false | Run all 5 scenarios |
| `--method / -m` | `all` | Memory method: `no_memory`, `full_context`, `rag`, `summary`, `structured`, or `all` |
| `--answer-model / -a` | `all` | Model(s): `all`, or comma-separated names/IDs |
| `--judge-model` | `google/gemini-2.5-pro` | Judge model |
| `--benchmark-dir` | `MUMBench` | Benchmark data directory |
| `--no-resume` | false | Start from scratch (ignore partial results) |
| `--top-k` | `20` | Top-K for RAG retrieval |
| `--chunk-size` | `512` | Chunk size for RAG |

**Model name resolution** supports fuzzy matching: `"Qwen3"` matches `"Qwen/Qwen3-14B"`, and full IDs like `"Qwen/Qwen3-14B"` also work.

#### `report` — Print scenario report

```bash
mum-eval report --scenario 1
```

#### `compare` — Cross-model/method comparison + JSON export

```bash
mum-eval compare
```

#### `list-models` — Show registered models with pricing

```bash
mum-eval list-models
```

---

## Data Schemas

### QuestionResult

```python
@dataclass
class QuestionResult:
    question_id: str         # Unique identifier
    category: str            # Evaluation category (e.g., "user_attribution")
    difficulty: str          # "easy", "medium", or "hard"
    question: str            # The evaluation question
    gold_answer: str         # Reference answer
    predicted_answer: str    # LLM-generated answer
    judge_score: JudgeScore  # Dimensional or binary score
    context_tokens: int      # Tokens in the memory context
    answer_time_s: float     # Time to generate the answer
```

### JudgeScore

```python
@dataclass
class JudgeScore:
    question_id: str
    correctness: float       # 1-5
    completeness: float      # 1-5
    attribution: float       # 1-5
    hallucination: float     # 1-5
    reasoning: str           # Judge's explanation
    is_binary: bool          # True for adversarial_confusion
    binary_correct: bool     # True/False for binary scoring

    @property
    def overall(self) -> float:
        # Returns 0-1 scale
        if self.is_binary:
            return 1.0 if self.binary_correct else 0.0
        return (0.35*correctness + 0.25*completeness +
                0.25*attribution + 0.15*hallucination) / 5.0
```

### ScenarioResult

```python
@dataclass
class ScenarioResult:
    scenario_id: str
    method_name: str
    model: str
    question_results: list[QuestionResult]
    total_time_s: float

    # Computed properties:
    overall_score() -> float              # Aggregate 0-1 score
    scores_by_category() -> dict          # Per-category breakdown
    scores_by_difficulty() -> dict        # Per-difficulty breakdown
    dimension_averages() -> dict          # Avg correctness/completeness/attribution/hallucination
```

---

## Evaluation Grid

The benchmark evaluates across 12 categories:

**8 Universal** (all scenarios):

| Category | Tests |
|----------|-------|
| user_attribution | Who said X? |
| cross_user_synthesis | Combine knowledge across users |
| conflict_resolution | Identify and resolve disagreements |
| information_gap | What does user A know that user B doesn't? |
| role_appropriate_briefing | Tailor information to a user's role |
| adversarial_confusion | Resistance to deliberate misattribution (binary) |
| document_coverage | Which users reviewed which document sections? |
| cross_user_provenance | Trace a fact's history across users/sessions |

**4 Scenario-Specific:**

| Category | Scenarios |
|----------|-----------|
| authority_hierarchy | 2 (hierarchical), 3 (cross-functional), 4 (adversarial) |
| temporal_correction | 1 (symmetric), 2 (hierarchical), 5 (sequential) |
| adversarial_side_isolation | 4 (adversarial) |
| sequential_handoff | 5 (sequential) |

---

## Output Files

```
MUMBench/results/
├── eval_{method}__{model_short}__s{scenario_id}.json   # Per-run results
├── comparison_report.json                               # Full grid comparison
└── logs/
    └── eval_{timestamp}.log                             # Detailed execution log
```

Each result JSON contains the full `ScenarioResult` with per-question details, scores, predicted answers, gold answers, and judge reasoning.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `chromadb` | Vector store for RAGMemory |
| `tiktoken` | Token counting (`cl100k_base` encoding) |
| `rich` | Console formatting and progress bars |
| `openai` | OpenAI-compatible client (DeepInfra API) |
| `click` | CLI framework |
| `python-dotenv` | `.env` loading for API keys |

Requires `DEEPINFRA_API_KEY` in `.env` for all LLM calls.
