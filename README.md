# Solvane Document Analyst Crew

A 3-agent [CrewAI](https://www.crewai.com/) pipeline that answers questions **strictly from a local
knowledge base** (RAG), never from the LLM's own training knowledge. Built around a fictional
company, Solvane Dynamics Inc. (warehouse robotics / fleet-orchestration software), with three
hand-crafted source PDFs as the only source of truth.

```
Document Researcher  →  Fact Checker  →  Report Writer
   (retrieves)          (re-verifies)     (formats final report)
```

---

## How it works

- **Retrieval**: a custom tool, `custom:rag_search` (`tools/rag_tool.py`), implements the full RAG
  pipeline by hand — PDF extraction, section-aware chunking, local embedding, and ChromaDB similarity
  search. No built-in CrewAI RAG tool is used, so every step is inspectable.
- **Embeddings**: 100% local via ChromaDB's bundled ONNX model (`all-MiniLM-L6-v2`) — no API key,
  no external calls for retrieval.
- **Agent LLM**: Groq (`llama-3.3-70b-versatile`), called through CrewAI's native OpenAI-compatible
  provider pointed at Groq's endpoint (LiteLLM was avoided — its build fails on this machine's
  Python 3.13 / Windows setup without a Rust/Cargo toolchain).
- **Agents run sequentially**, each with `context` chained to the previous task's output, so the
  Fact Checker and Report Writer see (and can be graded against) what came before them.

---

## Required questions

### (a) Where does retrieval happen — tool call or task prompt?

**Entirely inside the tool call**, never in a task prompt. `custom:rag_search` (`tools/rag_tool.py`)
does chunking, embedding, and similarity search internally, and only returns the top-K matching
chunks (with source filename + chunk index) to the agent. The task prompts (`config/tasks.jsonc`)
only tell the agent *when* and *how many times* to call the tool — they never contain retrieval logic,
raw document text, or pre-fetched context. This keeps the retrieval step fully swappable and
testable independent of the agents' reasoning.

### (b) What happens when the answer isn't in the documents?

By design, the intended behavior is:

1. `rag_search` still returns its top-K nearest chunks (it has no relevance/confidence threshold —
   see **Known Limitations** below), or `NO_RELEVANT_CHUNKS_FOUND` if the collection is empty.
2. The **Document Researcher** is instructed to respond with an exact fixed phrase —
   *"I don't have information about this in the available documents."* — for any part of the
   question not clearly supported by what was retrieved, rather than filling the gap with its own
   training knowledge.
3. The **Fact Checker** independently re-queries the knowledge base (not just reviewing the
   Researcher's citations) and labels each claim SUPPORTED or UNSUPPORTED.
4. The **Report Writer** moves anything UNSUPPORTED into a `## Flagged / Unverifiable` section and
   never states it as fact in the Executive Summary.

**In practice, this held up correctly in most tested cases** — pure out-of-scope questions (capital
of France, Microsoft's CEO), and in-scope-but-absent facts (Solvane's CEO, employee turnover rate,
Q1 revenue) were all correctly refused. **One case broke this design** — see below.

---

## Testing

All 14 test questions, across 6 categories (single-doc, multi-doc, pure refusal, hallucination
bait, mixed scope, exact-number precision), were run and individually verified against the source
PDFs. Full reports for each are in `output/`.

| # | Category | Question (short) | Result |
|---|---|---|---|
| 1 | Single-doc | PTO / vacation leave | ✅ |
| 2 | Single-doc | SLA response times | ✅ |
| 3 | Single-doc | Gross margin Q3 vs Q2 | ✅ |
| 4 | Multi-doc | Headcount, and does growth match Q3 OpEx | ✅ correctly synthesized figures from both the employee handbook and the Q3 financial summary |
| 5 | Multi-doc | SOC 2 status + data classification rules | ✅ correctly answered both halves from the right documents; correctly noted the documents never state an explicit link between the two |
| 6 | Pure refusal | Capital of France | ✅ correct refusal |
| 7 | Pure refusal | Microsoft's current CEO | ✅ correct refusal (minor awkward "Flagged/Unverifiable" phrasing, see note below) |
| 8 | Pure refusal | 15% of 340 | ✅ correct refusal, even though trivially calculable |
| 9 | Hallucination bait | Employee turnover/attrition rate | ✅ correct refusal |
| 10 | Hallucination bait | Solvane's CEO | ✅ correct refusal |
| 11 | Hallucination bait | FleetOS mobile app programming language | ❌ fabricated answer, Fact Checker did not catch it — see Known Limitations |
| 12 | Hallucination bait | Q1 revenue this year | ✅ correct refusal |
| 13 | Mixed scope | Remote work policy + industry comparison | ✅ correctly answered the in-scope half, correctly refused the out-of-scope half |
| 14 | Precision | Priority tier revenue growth % | ✅ exact match (16.5%), not confused with the adjacent Enterprise-tier figure (34.6%) |

**Result: 13 of 14 fully correct, 1 known limitation (Q11).** Full category coverage: single-doc,
multi-doc, pure refusal, hallucination bait, mixed scope, and exact-number precision were all
exercised at least twice.

A minor formatting inconsistency was also observed across several "not found" responses (Q6, Q7,
Q8, Q12): the Report Writer sometimes cites the Researcher's or Fact Checker's own internal output
as the "source" for a not-found claim, instead of a consistent, plain "not available in the
knowledge base" phrasing. Not a correctness issue — every one of these was still a correct refusal
— just a polish item for the report template.

---

## Known limitations

### 1. Retrieval can return a "confidently wrong" chunk (structural, not implementation-specific)

Question 11 (*"What programming language is the FleetOS mobile app written in?"*) exposed a real
gap: the knowledge base never mentions a mobile app anywhere — "mobile" only appears as part of
"Autonomous Mobile Robot" (AMR). But `product_spec.pdf` Section 6 does describe FleetOS's backend
architecture (Go for real-time services, Python/FastAPI + React for the dashboard). Because
`rag_search` has **no minimum relevance threshold** — it always returns its top-5 nearest chunks
regardless of how distant they are — this backend-architecture chunk was retrieved and used to
answer a question about a mobile app that was never described. The Researcher stated it as fact,
and the Fact Checker's re-query landed on the same topically-adjacent chunk and marked it
SUPPORTED instead of catching the mismatch.

This is a well-documented, structural characteristic of naive top-k similarity retrieval, not
unique to this project — vector search ranks by relative closeness, not absolute relevance, so it
has no way to say "nothing here is actually close enough." See:

- Julka, *"When Confidence Takes the Wrong Path: Diagnosing Retrieval-State Lock-In in RAG"*
  (2026) — names this exact failure pattern ("a coherent but wrong neighbourhood") and gives a
  directly analogous example (a RAG system misattributing a temple to the wrong pharaoh because the
  retriever anchors to a topically adjacent one). https://arxiv.org/html/2606.22728
- Daivam, *"Reducing False Positives in Retrieval-Augmented Generation (RAG) Semantic Caching: a
  Banking Case Study"*, InfoQ (2025) — a production example of the same failure shape outside an
  academic setting. https://www.infoq.com/articles/reducing-false-positives-retrieval-augmented-generation/
- Lewis et al., *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* (2020) — the
  original RAG paper, for background on the retrieve-then-generate approach this project builds on.

**Attempted mitigation:** a stricter Fact Checker instruction was added (explicitly requiring that a
cited chunk *assert* the specific claim, not just relate to the general topic) and Q11 was
retested. The fabrication still occurred. This suggests the gap is not simply a prompt-wording
issue but a limitation of relying on LLM judgment alone to catch subtle premise mismatches,
especially on a free-tier 70B model.

**A concrete next step** (not implemented here, given time/token constraints): add a similarity-
distance cutoff to `rag_search` in `tools/rag_tool.py` — refuse to return a chunk past a fixed
distance threshold, and return `NO_RELEVANT_CHUNKS_FOUND` instead. This moves the check from a
"hope the LLM notices" soft judgment into a deterministic guardrail that runs on every query,
which is the direction most production RAG mitigations (reranking, confidence thresholds,
answerability classifiers) take.

### 2. No re-ranking step

Retrieval is single-pass vector similarity only — there's no second model scoring "does this chunk
actually answer this question" independently of embedding distance. This is the same root cause as
limitation #1, listed separately because it's a distinct possible fix (see AB-RAG,
https://arxiv.org/html/2606.29090, for one training-free approach to adaptive/confidence-aware
retrieval).

### 3. Groq free-tier rate limits

Testing was frequently paced around Groq's free-tier per-minute (and possibly daily) token limits.
`main.py` adds a 20s pause between questions to help with this; a new API key on the same account
does not reset the limit, since it's enforced at the org level.

---

## Project structure

```
rag_doc_crew/
├── knowledge/               # 3 source PDFs (static, hand-authored)
├── tools/rag_tool.py        # custom "custom:rag_search" tool: chunk + embed + retrieve, local Chroma
├── config/agents.jsonc      # 3 agent definitions
├── config/tasks.jsonc       # 3 task definitions, sequential, context-chained
├── crew.py                  # loads jsonc, wires Agents/Tasks/Crew, builds the Groq LLM
├── main.py                  # CLI entry point (interactive / CLI args / --demo)
├── streamlit_app.py         # web UI
├── requirements.txt
├── .env / .env.example
├── .streamlit/secrets.toml.example
├── .gitignore
├── README.md
└── output/                  # generated reports
```

## Setup

```bash
cd "/d/USER DATA 2025/Desktop/oppotrian/rag_doc_crew"
source venv/Scripts/activate
pip install -r requirements.txt
```

Get a free Groq API key at https://console.groq.com/keys and add it to a local `.env` file:

```
GROQ_API_KEY=your-key-here
```

Embeddings run entirely locally — no key needed for retrieval.

## Running

```bash
streamlit run streamlit_app.py    # web UI (recommended)
# OR
python main.py                     # CLI: interactive prompt, --demo, or CLI args
```
