# Agentic AI — hands-on workshop

**GDG Mississauga · Thursday 30 July 2026 · 45 minutes**

Build an agent, break it three ways on purpose, then measure it. No buzzwords,
real execution.

## API key: **not required**

You do **not** need an API key, a Google Cloud account, a credit card, or any
credentials. The workshop runs against a small deterministic stand-in model that
lives in this repo and makes **zero network calls**. That is the default and it
is the path everyone in the room will use.

**Setup should take under 3 minutes.** Please do it before you arrive — it is the
one thing that can eat into our 45 minutes.

---

## Pick one path

### Path 1 — Google Colab (easiest, nothing to install)

1. Click here: **[▶ Open the workshop in Colab](https://colab.research.google.com/github/YOUR-GITHUB-USERNAME/gdg-agentic-ai-workshop/blob/main/workshop.ipynb)**
2. Click the **first code cell** (labelled `CELL 0 — PREFLIGHT`).
3. Press **Shift+Enter** to run it. Wait about 30 seconds.
4. You should see `YOU'RE READY`. That's it — close the tab and come to the talk.

You need a Google account (any Gmail works). Colab is free.

### Path 2 — Local (if you'd rather run on your own machine)

Never used a terminal? Path 1 is genuinely fine — use that instead.

Open a terminal (macOS: **Terminal**; Windows: **PowerShell**) and run these
four commands, one at a time, pressing Enter after each:

```bash
# 1. Install uv, a fast Python package manager. It also installs Python for you.
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
# Windows PowerShell instead: irm https://astral.sh/uv/install.ps1 | iex

# 2. Download this repo
git clone https://github.com/YOUR-GITHUB-USERNAME/gdg-agentic-ai-workshop
cd gdg-agentic-ai-workshop

# 3. Install exactly the pinned versions this workshop was tested with
uv sync

# 4. Open the notebook
uv run jupyter lab
```

Step 4 opens your browser. Click **`workshop.ipynb`**, click the first code
cell, press **Shift+Enter**.

No `git`? Download the repo as a ZIP from the GitHub page (green **Code**
button → **Download ZIP**), unzip it, and `cd` into that folder for step 3.

---

## You're ready when you see this

After running Cell 0, the output ends with:

```
MODE=mock — deterministic fake model, no API calls.

==============================================================
  YOU'RE READY     MODE=mock   rows loaded=20
  Parts 2-4 (ADK): available
==============================================================
```

If you see anything else, the message will name the specific fix — there are no
raw tracebacks in that cell by design. If you're still stuck, come find me
before we start and I'll sort it out in 60 seconds. **Do not spend more than 5
minutes on this at home.**

---

## What you'll actually do

| Part | What happens |
|---|---|
| 1 | Read a complete agent loop in ~40 lines. No framework. |
| 2 | Rebuild it in Google ADK, then **write your own tool**. |
| 3 | Break it on purpose: vague docstrings, a tool that lies, a runaway loop. |
| 4 | Score it — answer accuracy vs. trajectory accuracy are not the same thing. |
| 5 | Where agents actually work, and where they don't. |

Some Python helps, but the exercises are near fill-in-the-blank and every one
has a worked example directly above it. If you haven't written Python in years,
you'll be fine. `solutions.ipynb` has completed answers for everything.

---

## A note on what the "model" is

Colab and the default local setup run in **mock mode**: a deterministic stand-in
that picks tools by keyword-matching your question against each tool's name and
docstring. Every run prints
`MODE=mock — deterministic fake model, no API calls.`

It faithfully shows the agent loop, tool wiring, the real failure modes, and the
eval harness. It does **not** show real language understanding — a real model
generalizes from what your docstring *means*, the mock only from words that
literally overlap. The notebook is explicit about this where it matters.

**Live mode (real Gemini via Vertex AI) is presenter-only and local-only.** It
needs Google Cloud credentials I'll have on my laptop. It is not available in
Colab and you don't need it. Take-home instructions for running against a real
model are below.

---

## Take-home: running it against a real Gemini model

Afterwards, if you want to see a real model handle these same tools:

1. Get a **free** API key — no credit card:
   - Go to **https://aistudio.google.com/apikey**
   - Sign in with your Google account
   - Click **Create API key** → **Create API key in new project**
   - Click the copy icon
2. In the repo folder, copy the example env file and paste your key in:
   ```bash
   cp .env.example .env
   ```
   Open `.env` in any text editor and set `GOOGLE_API_KEY=your-key-here`.
   (`.env` is git-ignored — don't commit it.)
3. In Cell 0, change `MODE = "mock"` to `MODE = "live"`, then re-run from the top.

Nothing else changes. `MODE` is the only line that differs between a fake model
and Gemini — that's the design.

Two things worth trying once you're live: re-run Exercise 1 with a docstring
written entirely in *synonyms* of the question's words (the mock misses it,
Gemini usually doesn't), and re-run Part 4 a few times to watch the scores move
even at temperature 0.

---

## Presenter notes

```bash
uv run python check_live.py     # preflight: resolves ADC, one real call, exits nonzero on failure
```

Live mode auto-detects Vertex AI via ADC first (`gcloud auth
application-default login` + `GOOGLE_CLOUD_PROJECT`), then falls back to an AI
Studio key, then falls back to mock rather than crashing. Requires
`roles/aiplatform.user` and the Vertex AI API enabled. Model IDs are pinned in
one constant at the top of Cell 0 and in `check_live.py`.

**Before you publish this repo:** replace `YOUR-GITHUB-USERNAME` in the Colab
badge link and the two `git clone` commands above, and in `REPO_URL` in Cell 0.
