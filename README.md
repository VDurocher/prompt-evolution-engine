# 🧬 Prompt Evolution Engine

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![OpenAI](https://img.shields.io/badge/Powered%20by-GPT--4o--mini-412991)](https://openai.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)

> **Automatically evolves prompts using a genetic algorithm.** Give it a task and an initial prompt — it generates variants, scores them, selects the best, and iterates until it finds the optimal prompt.

---

## The Problem It Solves

Prompt engineering is largely manual trial-and-error. Engineers spend hours tweaking wording, adding chain-of-thought, trying few-shot examples — with no systematic feedback loop.

**Prompt Evolution Engine automates this process:**

| Manual approach | This tool |
|----------------|-----------|
| Write a prompt, test it manually | Automatic generation of N variants per generation |
| Guess which technique works best | Systematic exploration: CoT, few-shot, persona, XML, constraints… |
| No visibility on improvement | Live score chart across generations |
| Black-box prompt quality | LLM-as-judge + deterministic metrics |

---

## How It Works

```
Initial Prompt
      │
      ▼
┌─────────────────────────────────────────┐
│  Generation 0 — Score initial prompt    │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Mutator (GPT-4o-mini)                  │  ← generates N variants
│  Techniques: CoT · few-shot · persona   │    via meta-prompt
│  XML · constraints · step-back · …     │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Executor — runs each variant on task   │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Evaluator                              │
│  • LLM-as-judge (GPT-4o-mini)          │
│  • Deterministic: keywords, JSON, len  │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  Tournament Selection → top K survive   │
└─────────────────────────────────────────┘
      │
      └──── Repeat for K generations ────▶ Best Prompt
```

**Tournament selection:** 3 random candidates compete per round — the best survives. Prevents premature convergence.

---

## Features

- **Live evolution dashboard** — score chart updates after each generation
- **Technique tagging** — each variant is labelled with the prompt engineering technique used
- **Genealogy tree** — Graphviz tree showing every prompt's lineage and score
- **Dual evaluator** — combine LLM-as-judge (qualitative) with deterministic metrics (keywords, JSON validity, length)
- **Cost estimation** — shows estimated API cost before running
- **~0.03€ per full run** with GPT-4o-mini (5 variants × 4 generations)

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/VDurocher/prompt-evolution-engine
cd prompt-evolution-engine
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your OpenAI API key to .env

# 3. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 5 | Variants generated per generation |
| `num_generations` | 4 | Number of evolution cycles |
| `num_survivors` | 2 | Best variants kept for next mutation |
| `tournament_size` | 3 | Candidates per tournament round |
| `MUTATION_TEMPERATURE` | 0.9 | LLM creativity for mutations |
| `JUDGE_TEMPERATURE` | 0.1 | LLM determinism for scoring |

All defaults in `config.py`.

---

## Mutation Techniques

| Technique | Description |
|-----------|-------------|
| `chain_of_thought` | "Let's think step by step" + structured reasoning |
| `few_shot` | 1-2 concrete input/output examples injected |
| `persona` | Expert role assignment ("You are a senior…") |
| `xml_structure` | Instructions wrapped in XML tags |
| `constraints` | Explicit format/length/style rules |
| `reformulation` | Clearer rewrite of the same instructions |
| `socratic` | Clarifying questions guide the response |
| `step_back` | High-level reflection before answering |

---

## Project Structure

```
prompt-evolution-engine/
├── core/
│   ├── genome.py       # PromptGenome + GenerationResult data structures
│   ├── mutator.py      # LLM-based variant generation
│   ├── evaluator.py    # Dual scoring (deterministic + LLM-judge)
│   ├── executor.py     # Runs a prompt against the task
│   └── evolution.py    # Main genetic algorithm loop (generator)
├── app.py              # Streamlit UI with live updates
├── config.py           # All tunable parameters
└── tests/              # Unit tests (genome, evaluator, evolution)
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Cost Estimation

With GPT-4o-mini (~$0.15/1M input tokens, ~$0.60/1M output tokens):

| Config | API calls | Estimated cost |
|--------|-----------|----------------|
| 5 variants × 4 generations | ~25 calls | ~€0.03 |
| 8 variants × 6 generations | ~55 calls | ~€0.06 |
| 10 variants × 8 generations | ~90 calls | ~€0.10 |

---

## License

MIT
