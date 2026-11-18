# Let It Cook

An AI research agent that plans its approach, picks the right tool for each step, and checks its own answer before it replies. It shows all of its work as it goes.

<img width="688" height="910" alt="LetItCook-1" src="https://github.com/user-attachments/assets/ad79485c-e4a6-4c51-b674-6e1bd84b6e7b" />

**Live demo:** [letitcook-rl.vercel.app](https://letitcook-rl.vercel.app/)
**Related:** [Pull The Receipts](https://pullthereceipts-rl.vercel.app), my RAG project, which this one is the sequel to.

> The backend runs on a free Hugging Face Space that naps when nobody's using it. The page pokes it awake as soon as it loads, but the first question after a quiet spell can still take a minute or more. After that, it's quick.

Most chatbots answer in one shot. This one doesn't. You give it a question, and it writes a short plan, works out whether it needs to search the web, look through a set of documents, or do some maths, runs that step, looks at what came back, and keeps going until it has enough to answer properly. Then it reviews its own draft before handing it over. The interface shows every step while it happens, so you can see how it got there instead of taking the answer on trust.

I built it after Pull The Receipts, my retrieval project, as a deliberate step up from RAG into agents.

## Why it's an agent and not just RAG

RAG runs a fixed pipeline: take the question, fetch the closest matching chunks, answer from them. That's great when the answer is sitting in a document. It falls apart the moment a question needs more than one lookup, or a calculation, or even a decision about where to look.

An agent decides for itself. It plans the steps, picks whichever tool fits, runs it, and works out from the result whether it can answer yet or needs another step. That loop is the whole difference.

A question that shows it:

> What is the population difference between the two most populous countries in Southeast Asia, and what is that difference as a percentage of the larger one?

No single search or document answers that. The agent has to find two facts, do a calculation on them, and write it up. So it searches for the populations, passes the numbers to a calculator so the maths is right rather than guessed, and answers. You watch each step go by.

## The four agentic patterns

These four come up everywhere in agent work, and the project is built around them. Each one lives in a specific place in the code.

**Planning.** Before doing anything, the agent writes a short plan for the question. This is the `plan` node in `app/agent/graph.py`. It keeps the agent from reacting blindly and gives the later steps something to follow.

**Tool use.** The agent has three tools and decides which to call, and with what input, based on the task. The decision happens in the `reason` node through the model's function calling, and the chosen tool runs in the `act` node. Adding or swapping a tool doesn't touch the graph, because the executor looks tools up by name.

**Memory.** Two kinds, kept separate. Task memory is the running record of what the agent has tried within a single question, held in the `messages` field of the state. Conversation memory is remembering earlier turns in the same chat, which comes from a checkpointer that saves the state per conversation thread. (Long-term memory across different conversations would be a third kind, a store, which I didn't build.)

**Reflection.** After writing a draft, the agent grades its own answer in the `reflect` node and decides whether it's good enough or needs another go. If it falls short, it loops back with specific feedback and tries again, up to a limit. This is what lets it catch a weak first attempt instead of shipping it.

## How it works

The agent is a state graph: nodes, a shared state object that flows through them, and edges that decide what runs next. I built the loop by hand with LangGraph instead of using a prebuilt agent, because I wanted each step to be its own node I could see and explain, and because the visible reasoning in the UI comes straight out of that.

```
START -> plan -> reason -> (asked for a tool?)
                              |- yes -> act -> back to reason
                              |- no  -> reflect -> (good enough?)
                                                     |- yes -> finalize -> END
                                                     |- no  -> back to reason
```

`reason` is the hub. It looks at the plan and the conversation so far and either calls a tool or writes a draft answer. If it called a tool, `act` runs it and loops back. If it wrote an answer, `reflect` grades it and either finishes or sends it round again. Two counters cap the loops, one for tool steps and one for revisions, so it can't spin forever.

**State** (`app/agent/state.py`) is what every node reads and writes. It holds the question, the plan, the running `messages` list (the scratchpad, which becomes conversation memory once the checkpointer persists it), the two counters, the reflection verdict, and the final answer.

**Tools** (`app/agent/tools.py`, `calculator.py`, `retriever.py`) are three:

- Web search through Tavily, for current or general facts.
- Document search, retrieval over a ChromaDB vector store with local bge-small-en embeddings, the same retrieval setup as Pull The Receipts, wrapped here as a tool the agent can pick. It reads `.txt`, `.md`, and `.pdf`.
- A calculator that parses the expression into a syntax tree and only allows numbers and arithmetic operators, so the model can't sneak code execution in through the expression. It also caps how big a result can get (more on that below).

**The model** sits behind one `get_llm` function (`app/agent/llm.py`), so swapping Gemini for another provider is a one-place change. Every model call goes through a small retry helper, so a brief overload backs off and tries again instead of killing the run.

**The backend** (`app/api/`) is FastAPI with two endpoints: one that runs the agent and returns the finished answer, and one that streams each step as a Server-Sent Event. The streaming one is what the frontend reads to show the reasoning live. It's rate limited per IP, with a daily cap on top, so a public demo can't quietly run up the API bill.

**The frontend** (`frontend/`) is Next.js with TypeScript and Tailwind. It opens the streaming endpoint and lays the events out as a timeline: the plan, each tool step with what it found, the self-checks, and the answer. Showing the work is the main feature, so that's where most of the effort went.

I also wired in LangSmith for tracing, which turns on with an environment variable. With it on, every run becomes a tree you can open: each node, each model call with its prompt and tokens, each tool call, and any reflection retries.

## Keeping a public demo in one piece

Anyone can open the demo and type into it, and every question costs a few model calls. So the backend is built to fail politely and not run up a bill.

- **Retries, but only when they'll help.** Rate limits (429) and server errors (5xx) get up to five attempts, with exponential backoff and a bit of jitter. A bad API key or a malformed request fails straight away, because trying again won't change anything. I turned off the Google SDK's own retries, so there's only one layer doing it.
- **Rate limits.** Each IP gets 5 questions a minute, and the whole demo gets 100 a day, resetting at midnight UTC. Going over returns a 429 with a `Retry-After` header, and the page tells you whether to wait a few seconds or come back tomorrow. The daily cap is the real backstop on spending.
- **Finding the real visitor behind the proxy.** On Hugging Face, every request reaches the app through their proxy, so a naive per-IP limit would lump everyone into one bucket. uvicorn reads the visitor's address from `X-Forwarded-For`, but only when the request comes from a private-network proxy. It reads the header from the right, so an address someone writes into it themselves never gets picked.
- **Errors show up in the chat, not as an endless spinner.** If a run fails partway, the server logs the full error and sends the page a plain "something went wrong" event, with no internals. If the connection drops before the answer finishes, the page says so.
- **A broken tool doesn't end the run.** If Tavily or the vector store throws, or the model asks for a tool that doesn't exist, the error goes back to the model as that tool's result, and it can try something else.
- **The calculator has a ceiling.** Python integers have no size limit, so `9**9**9` would happily keep the server busy for ages. Whole-number results are capped at 10,000 bits (about 3,000 digits), and powers are checked before they're worked out, so that one gets refused instantly.
- **Every conversation gets its own memory.** The page sends a fresh thread ID per conversation, and a request without one gets a new one, so two visitors never end up sharing a chat by accident.
- **Logs without the personal bits.** The server log gets one line per request with the path, status and how long it took. Questions are logged by length only, never their text, and IPs only appear as a short salted hash that changes every time the server restarts. (LangSmith traces do keep the full question when tracing is on, since that's the whole point of a trace.)

## Testing

The suite has 37 tests and runs offline in about a second. Because the graph is built by a factory that takes the model and tools as arguments, the tests pass in fakes and run the whole agent with no API calls and nothing to spend. That seam is what makes it possible.

Beyond the agent loop itself, the tests cover:

- the retry rules: a transient error is retried and the run still finishes, while a bad key isn't retried;
- tool failures and made-up tool names;
- the calculator's limits;
- the API through FastAPI's `TestClient`: input validation, both rate limits (including the midnight reset), CORS settings, and the stream's error event not leaking any details.

Tracing and API keys are switched off before anything is imported, so a test run can't send a trace or spend anything by accident.

## The stack, and why

- **LangGraph** to build the agent as an explicit graph with a real loop and branching, so I could see and explain every step rather than hand it to a prebuilt agent that hides the loop.
- **LangChain** for the tool interfaces, the model calls, and the structured output the reflection step uses to return a clean yes or no.
- **Gemini 2.5 Flash-Lite**, because an agent makes several model calls per question, so cheap and fast matters more than top-end quality. Cost stays in fractions of a cent per question.
- **ChromaDB and bge-small-en** for the document tool's retrieval.
- **Tavily** for web search, since it's built for agents and returns clean text in one call.
- **LangSmith** for tracing and inspecting any run.
- **FastAPI** to serve the agent and stream the steps.
- **Next.js, React, TypeScript, Tailwind** for the frontend.
- **Docker** to containerise the backend, with the embedding model and vector store baked into the image so a cold start doesn't stall on a download.
- **pytest** for the offline test suite.

## Running it locally

You'll need Python 3.12 (3.13 works too), Node 20.9 or newer, a Google AI Studio API key, and a Tavily API key. Both keys are free to sign up for.

The backend is at the root, the Next.js frontend is in `frontend/`.

### Backend

```bash
git clone https://github.com/riskilinardi/let-it-cook.git
cd let-it-cook

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your two keys:

```
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
```

LangSmith is optional. If you're not using it, set `LANGSMITH_TRACING=false`.

`documents/` comes with two sample files. Add your own `.txt`, `.md` or `.pdf` files if you like, then build the vector store (the first run downloads the embedding model, around 130MB):

```bash
python ingest.py
```

Then either talk to the agent in the terminal:

```bash
python run.py
```

or run the API:

```bash
uvicorn app.api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
```

Make `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Then:

```bash
npm run dev
```

Open http://localhost:3000.

### Tests

From the root, with the venv active:

```bash
pytest
```

## Configuration

Keys and tracing go in environment variables:

- `GOOGLE_API_KEY` and `TAVILY_API_KEY` are required.
- `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` turn on tracing.
- `ALLOWED_ORIGINS` is a comma-separated list of frontend URLs allowed to call the API. Defaults to localhost:3000.
- `RATE_LIMIT_PER_MINUTE` (default 5, per IP) and `DAILY_QUESTION_CAP` (default 100, for everyone) set the demo's limits.

The rest is in `app/config.py`: the model name, the 60-second model timeout, the loop caps (`MAX_ITERATIONS`, `MAX_REVISIONS`), the chunking settings, and the default rate limits. To use a different Gemini model, change `MODEL_NAME` and nothing else moves, thanks to the `get_llm` seam. If you switch to a Gemini 3 model, the code leaves the temperature unset for you, since those misbehave at low temperatures.

## Deployment

The two halves deploy separately: the backend as a Docker container, the frontend on Vercel. I run the backend on Hugging Face Spaces, where the Space's README needs `app_port: 8000` in its front matter.

Order matters, because each side needs the other's URL:

1. Deploy the backend and note its URL.
2. Deploy the frontend with `NEXT_PUBLIC_API_URL` set to that backend URL (no trailing slash), and note the frontend's URL.
3. Set `ALLOWED_ORIGINS` on the backend to the frontend URL, then restart the backend.

On the backend host, set `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `ALLOWED_ORIGINS`, and the `LANGSMITH_*` keys if you want tracing. On Vercel, set `NEXT_PUBLIC_API_URL`. The Docker image already trusts private-network proxies for `X-Forwarded-For`. If your host's proxy sits somewhere else, override `FORWARDED_ALLOW_IPS`.

The demo is public, so on top of the rate limits, I keep the Gemini billing on a prepaid balance so spending has a hard ceiling no matter what.

## Limitations and what I'd do next

Where it stands:

- It's slower than a single answer. The loop means several model and tool calls per question, so there's a real wait. The streaming UI is partly there to make the wait feel like progress.
- It costs more than a single call. A few calls per question, and a public endpoint spends on every visitor. Tiny on Flash-Lite, but it's the reason for the rate limits and the prepaid cap.
- It still gets things wrong. It can pick the wrong tool, or the reflection step can wave through an answer it should have sent back. Reflection lowers the error rate, it doesn't remove it.
- Everything lives in memory. Conversations and the rate-limit counts sit in the server's RAM and reset on restart. Fine for one container, not for anything real.
- Web searches have no timeout. Model calls give up after 60 seconds, but the Tavily wrapper I use doesn't offer a timeout setting, so a search that hangs can stall a run.
- `ALLOWED_ORIGINS` doesn't do much on Hugging Face. Their proxy answers CORS for any origin by itself, so the setting only really matters if the backend moves to another host. The rate limits are what actually protect the demo.
- PDF text extraction is rough on tables and multi-column layouts, so figures buried in tables can come through messy.

Given more time I'd add a persistent checkpointer (SQLite or Redis) so conversations survive a restart, put my own timeout around web search, make the nodes fully async to handle more users at once, add prompt caching to cut the cost of the repeated system prompt, and put together a small evaluation set so I could tell whether a prompt or model change actually helps instead of going on feel.

---

Built by Riski Linardi. [Portfolio](https://riskilinardi.dev) · [Pull The Receipts](https://pullthereceipts-rl.vercel.app)
