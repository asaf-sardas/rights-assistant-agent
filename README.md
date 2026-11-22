# rights-assistant-agent

AI agent that helps workers in Israel find out which rights they're entitled to. Uses RAG over [Kol Zchut (כל זכות)](https://www.kolzchut.org.il), asks clarifying questions when eligibility depends on personal details, and cites a source for every answer.

## What it does

- **Greets** the user as their personal assistant for workers' rights.
- **Searches** Kol Zchut content (stored in a vector database) for pages relevant to the user's question.
- **Asks clarifying questions** when the answer depends on personal details (e.g. years at the job, full/part-time, how the employment ended) — all missing details in one numbered message.
- **Answers only from the retrieved sources**, tailored to the user's situation, with numbered citations linking back to Kol Zchut.
- **Handles follow-up questions** while remembering what it already knows about the user.
- **Says so honestly** when the sources don't cover a question, and politely declines topics outside workers' rights in Israel.

> **Disclaimer:** This project provides general information only and is **not legal advice**. Rights depend on individual circumstances and may change. Always verify with [Kol Zchut](https://www.kolzchut.org.il), the Ministry of Labor, or a qualified lawyer.

> **Status:** early development. The architecture below is the design being implemented; source content is currently loaded from local `.txt` files, with a Kol Zchut fetcher planned.

## How it works

The agent is a [LangGraph](https://langchain-ai.github.io/langgraph/) state graph. **Each user message runs the graph once**, from `START` to `END`. Conversation state (messages, what we know about the user, current topic) is saved by a checkpointer under a `thread_id`, so the next message continues where the last one left off. The "ask → answer → ask again" loop therefore happens **across turns**, not inside a single run.

```
START ─┬─ (no messages yet) ──► greet ──► END
       │
       └─► understand_message ─┬─ out_of_scope ──► decline ──► END
                               │
                               └─ new_topic / clarification_answer / follow_up
                                              │
                                              ▼
                                   retrieve ──► assess_sufficiency
                                                   │
                                                   ├─ sources don't cover it ──────────► no_info ──► END
                                                   ├─ missing user info & rounds < MAX ─► ask_clarifying ──► END
                                                   └─ otherwise ───────────────────────► answer ──► END
```

### Nodes

| Node | What it does |
|---|---|
| `greet` | Sends the opening message. Chosen deterministically when the conversation is empty (no LLM call). |
| `understand_message` | One LLM call with structured output: classifies the **intent** (`new_topic`, `clarification_answer`, `follow_up`, `out_of_scope`), extracts **profile facts** about the user, and writes a **search query**. If the agent just asked a clarifying question, it prefers `clarification_answer` unless the user clearly changed topic. |
| `decline` | Politely explains the agent only covers workers' rights in Israel. Also handles attempts to override the agent's instructions. |
| `retrieve` | Searches the vector store and returns the top-k chunks, numbered `[1]..[k]`. |
| `assess_sufficiency` | Decides between two kinds of "missing": the **sources** don't cover the question, or **user details** are missing (returned as a list ordered by importance). |
| `no_info` | Says the sources don't cover the question and refers the user to Kol Zchut / the Ministry of Labor. |
| `ask_clarifying` | Asks **all** missing details in one numbered message, most important first. Limited by `MAX_CLARIFY_ROUNDS`. |
| `answer` | Answers **only** from the retrieved sources, tailored to the user's profile, citing sources as `[1]`, `[2]`. If the clarify limit was reached, it states its assumptions. |

### State

| Field | Meaning |
|---|---|
| `messages` | Full conversation history. |
| `user_profile` | Facts about the user (employment type, tenure, full/part-time, how employment ended, …). Merged every turn and **never reset**, so it carries over to new topics. |
| `intent`, `topic`, `search_query` | Output of `understand_message` for the current turn. |
| `retrieved_docs` | Chunks from the vector store with source id, title, URL and text. |
| `sources_cover_question`, `missing_info` | Output of `assess_sufficiency`. |
| `clarification_rounds` | How many times we asked for details on the current topic. Reset on `new_topic`. |
| `phase` | `new` → `clarifying` → `answered`. |

### Design principles

- **Grounded answers only.** The LLM sees sources as `[n] title + text` and must cite them. It never writes URLs; the code maps cited ids to the title and URL stored in the chunk metadata, so links can't be hallucinated.
- **Ask once, ask everything.** Clarifying questions come in a single numbered message instead of one at a time.
- **Bounded questioning.** After `MAX_CLARIFY_ROUNDS` the agent answers with explicit assumptions instead of asking again.
- **Swappable data source.** Retrieval sits behind a `Retriever` interface. Today it indexes `.txt` files; later a Kol Zchut fetcher feeds the same index without changing the graph.

## Tech stack

- **Python 3.12**, managed with [uv](https://docs.astral.sh/uv/)
- **LangGraph** — agent graph and conversation memory (checkpointer)
- **OpenAI** — chat model and `text-embedding-3-small` embeddings
- **Chroma** — local persistent vector store
- **pydantic-settings** — typed configuration

## Setup

```bash
# 1. Install dependencies (creates .venv from pyproject.toml + uv.lock)
uv sync

# 2. Create your .env and add your OpenAI key
cp .env.example .env
```

All other settings (models, `TOP_K`, `MAX_CLARIFY_ROUNDS`, paths) have defaults in `src/rights_assistant_agent/config.py`.

## Usage

```bash
# Build the vector index from the .txt files in data/raw/
uv run python scripts/ingest.py

# Chat with the agent in the terminal
uv run rights-assistant-agent

# Open LangGraph Studio to see the graph and step through state
uv run langgraph dev
```

> If you change the embedding model, delete `.chroma/` and run the ingest again — vectors from different models can't be compared.

### Source file format

Each file in `data/raw/` is one Kol Zchut page:

```
Title: פיצויי פיטורים
URL: https://www.kolzchut.org.il/he/פיצויי_פיטורים

<page text...>
```

The `Title` and `URL` lines become chunk metadata and are used for citations.

## Project structure

```
src/rights_assistant_agent/
  __init__.py            # entry point: main()
  config.py              # settings and defaults
  llm.py                 # chat model + embeddings factories
  prompts.py             # all system prompts
  cli.py                 # terminal chat loop
  graph/
    state.py             # graph state definition
    nodes.py             # node functions
    routing.py           # conditional-edge functions
    builder.py           # builds and compiles the graph
  retrieval/
    base.py              # Retriever interface, RetrievedDoc
    indexer.py           # chunking + writing to Chroma
    chroma_retriever.py  # search implementation
    loaders/
      txt_loader.py      # reads data/raw/*.txt
scripts/
  ingest.py              # builds the vector index
data/raw/                # source pages (.txt)
tests/
langgraph.json           # LangGraph Studio config
```

## Roadmap

- [ ] Core agent flow over local `.txt` files
- [ ] Kol Zchut fetcher (download and clean pages into `data/raw/`)
  - [ ] Check Kol Zchut terms of use / API access before automated downloads
  - [ ] Only workers'-rights categories, not the whole site
  - [ ] Store each page's MediaWiki revision id; re-fetch only pages that changed
  - [ ] Keep hand-written samples (`data/samples/`, committed) separate from fetched pages (`data/fetched/`, git-ignored)
- [ ] Incremental ingest instead of full rebuild
  - [ ] Chunk ids based on page URL instead of file name
  - [ ] `content_hash` per page in chunk metadata; skip unchanged pages
  - [ ] Changed page → delete all its chunks, then re-add (fixes stale chunks when a page gets shorter)
  - [ ] Deleted page → delete its chunks
- [ ] Evaluation set of realistic questions to measure answer quality
- [ ] Web UI / API
- [ ] Persistent conversation storage (replace in-memory checkpointer)
