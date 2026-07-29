"""A deterministic stand-in for a real LLM.

WHAT THIS IS
------------
This file lets ~50 people run a real agent loop on a bad conference wifi with no
API key. It is not a language model. It picks tools by counting words that
overlap between your question and the tool's name + docstring.

WHY NOT A CANNED-RESPONSE CACHE
-------------------------------
Because attendees invent their own tools in Exercise 1 and rewrite docstrings in
Exercise 2/3. A replay cache has no key for input it has never seen, so the
exercises would be theatre. This scores whatever tools you hand it, including
ones written 30 seconds ago.

WHY THE TOP HALF OF THIS FILE IS STDLIB-ONLY
--------------------------------------------
Part 1 of the workshop must run even if `pip install` fails outright. So the
decision engine below imports nothing but the standard library. The ADK adapter
at the bottom is behind a guarded import and is only needed from Part 2 onward.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Tuning knobs. These are the whole "model".
# --------------------------------------------------------------------------

# Words too common to carry signal. Kept short and hand-written on purpose: an
# attendee should be able to read this list and predict the model's behaviour.
STOPWORDS = frozenset("""
a about all am an and any are as at be been but by can could do does doing each
every find for from get give got had has have how i if in into is it its just
know let like make may me might much must my need of off on or our out please
should show so some tell than that the their them then there these they this
those to too us use want was we were what when where which who why will with
would you your
""".split())

# A token must survive stopword removal AND be at least this long to count.
# 3 keeps the words that carry meaning here ("sku", "oz" is lost but unused) and
# drops two-letter noise like the "hd" in "HD-1001". Pure digits are always
# kept, so the "1001" half of a SKU and quantities like "40" still register.
MIN_TOKEN_LEN = 3

# A word matching the tool's NAME is worth more than one matching only its
# docstring body — the name is the strongest signal a human gives a tool.
# These weights decide *which* eligible tool wins (the ranking).
NAME_WEIGHT = 2.0
DOC_WEIGHT = 1.0

# Score floor for calling any tool at all. Below this, the model stops and
# answers in text. Note this is *not* the binding constraint — MIN_DOC_MATCHES
# below almost always bites first, since 2 docstring matches already score 2.0.
# It is kept as an explicit floor so the stop condition is visible in one place.
MIN_SCORE_TO_CALL_TOOL = 2.0

# The gate that matters, and the reason this workshop works.
#
# A tool is only eligible to be called if at least this many content words from
# the question appear in its DOCSTRING. Matching the function *name* is not
# enough to authorize a call — it only affects ranking among eligible tools.
#
# That asymmetry is deliberate. It makes "the docstring IS the prompt"
# mechanically true instead of merely asserted:
#   - A good docstring naturally restates and expands the name's concepts, so it
#     clears the gate easily.
#   - A vague or empty docstring cannot clear it, no matter how well-named the
#     function is. The tool simply never gets called.
# Exercise 1 depends on exactly this, and so does the recovery in Exercise 3.
MIN_DOC_MATCHES = 2

_WORD_RE = re.compile(r"[a-z0-9]+")


def _singular(word: str) -> str:
    """Crudest possible stemmer: drop one trailing 's'.

    Exists so "products" in a question matches "product" in a docstring. Applied
    to BOTH sides, so it never makes matching asymmetric. Not linguistics — just
    enough that plurals don't silently break tool selection.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and short words.

    Returns a list (not a set) so callers can see duplicates, but all scoring
    below deduplicates. Deterministic: no hashing, no ordering surprises.
    """
    if not text:
        return []
    out = []
    for word in _WORD_RE.findall(text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) < MIN_TOKEN_LEN and not word.isdigit():
            continue
        out.append(_singular(word))
    return out


def score_tool(question: str, tool_name: str, tool_doc: str) -> dict:
    """Score how well one tool matches one question. Higher wins.

    Returns a dict rather than a bare number so the notebook can *show*
    attendees why a tool won or lost. Opaque scoring teaches nothing.
    """
    q = set(tokenize(question))
    name_tokens = set(tokenize(tool_name.replace("_", " ")))
    doc_tokens = set(tokenize(tool_doc or ""))

    # Words the question shares with the docstring. This is the eligibility
    # signal, and it deliberately includes words that also appear in the name —
    # a good docstring repeats the name's key nouns, and it should get credit.
    doc_hits = sorted(q & doc_tokens)

    # For ranking only, avoid double-counting: a shared word scores at the
    # higher name weight, and the rest score at the doc weight.
    name_hits = sorted(q & name_tokens)
    doc_only_hits = sorted(q & (doc_tokens - name_tokens))

    score = NAME_WEIGHT * len(name_hits) + DOC_WEIGHT * len(doc_only_hits)
    return {
        "score": score,
        "matched": name_hits + doc_only_hits,
        "name_hits": name_hits,
        "doc_hits": doc_hits,
        "eligible": len(doc_hits) >= MIN_DOC_MATCHES and score >= MIN_SCORE_TO_CALL_TOOL,
    }


def rank_tools(question: str, tools: list[dict]) -> list[dict]:
    """Rank every tool for a question, best first.

    `tools` is a list of {"name": str, "description": str} — the same shape the
    real model receives as function declarations.

    Sort key puts eligible tools first, then highest score, then tool name
    ascending. That last term is what makes ties deterministic across runs,
    across Python versions, and across dict insertion order.
    """
    ranked = []
    for tool in tools:
        result = score_tool(question, tool["name"], tool.get("description", ""))
        ranked.append({"name": tool["name"], **result})
    ranked.sort(key=lambda r: (not r["eligible"], -r["score"], r["name"]))
    return ranked


def _numbers_in_observations(history: list[dict]) -> dict:
    """Collect {field_name: number} from tool results, most recent last.

    This is what lets one tool's output feed the next tool's input — a genuine
    data dependency, which is the only reason this mock ever takes a second step.
    """
    found: dict = {}
    for turn in history:
        if turn.get("role") != "tool":
            continue
        for key, value in (turn.get("fields") or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                found[key.lower()] = value
    return found


def _content_phrase(question: str, all_tools: list[dict]) -> str:
    """The longest run of question words that no tool's own text claims.

    For "search the catalog for a cordless drill and tell me its SKU", every word
    except "cordless drill" also appears in some tool name or docstring. What is
    left over is what the user is actually asking *about*. Cheap, and it beats
    grabbing the last word.
    """
    vocabulary: set[str] = set()
    for tool in all_tools:
        vocabulary |= set(tokenize(tool["name"].replace("_", " ")))
        vocabulary |= set(tokenize(tool.get("description", "")))

    best: list[str] = []
    current: list[str] = []
    for word in tokenize(question):
        if word in vocabulary or word.isdigit():
            current = []
        else:
            current.append(word)
            # >= so that on a tie the LATER phrase wins. Questions tend to put
            # the thing being asked about after the framing words.
            if len(current) >= len(best):
                best = list(current)
    return " ".join(best)


def guess_args(
    question: str, tool: dict, history: list[dict] | None = None,
    all_tools: list[dict] | None = None,
) -> tuple[dict, bool]:
    """Fill in tool arguments from the question and from prior observations.

    Returns (args, used_observation). That second value drives the stop
    condition: a tool whose arguments all came straight from the question is
    just another lookup of something we already asked about, whereas a tool that
    needs a *previous result* is a real next step in a chain.

    A real model reads the parameter schema and reasons about what to pass. This
    does regex and dictionary lookups. It is the weakest part of the mock, and we
    say so out loud in the notebook.
    """
    history = history or []
    all_tools = all_tools or [tool]
    params = tool.get("parameters") or {}
    props = params.get("properties") or {}
    args: dict = {}
    used_observation = False

    sku_match = re.search(r"\b(HD-?\d{4})\b", question, re.IGNORECASE)
    sku = sku_match.group(1).upper().replace("HD", "HD-").replace("--", "-") if sku_match else None
    stripped = re.sub(r"HD-?\d{4}", "", question, flags=re.IGNORECASE)
    number_match = re.search(r"\b(\d+)\b", stripped)

    observed_numbers = _numbers_in_observations(history)
    categories = ("hand tools", "power tools", "paint", "fasteners", "electrical", "plumbing")

    for pname, pschema in props.items():
        ptype = str((pschema or {}).get("type", "string")).lower()
        lowered = pname.lower()

        if "sku" in lowered and sku:
            args[pname] = sku
        elif "category" in lowered:
            found = next((c for c in categories if c in question.lower()), None)
            args[pname] = found if found else "hand tools"
        elif ptype in ("integer", "number"):
            # Prefer a value a previous tool produced under a matching name —
            # that is how price flows from a lookup into a multiplication.
            match = next(
                (v for k, v in observed_numbers.items() if lowered in k or k in lowered),
                None,
            )
            if match is not None:
                args[pname] = int(match) if ptype == "integer" else match
                used_observation = True
            elif number_match:
                args[pname] = int(number_match.group(1))
            else:
                args[pname] = 1
        elif sku and ("product" not in lowered and "name" not in lowered and "query" not in lowered):
            args[pname] = sku
        else:
            args[pname] = _content_phrase(question, all_tools) or (sku or "")
    return args, used_observation


def already_observed(tool_name: str, args: dict, history: list[dict]) -> bool:
    """Has this exact call — same tool, same arguments — already been answered?

    Compares arguments, not just the name, so a tool called with genuinely new
    arguments may run again. That is what allows the pagination runaway in
    Part 3 to keep going, while still blocking pointless repeat lookups.
    """
    for turn in history:
        if turn.get("role") == "tool" and turn.get("name") == tool_name:
            if turn.get("args") == args:
                return True
    return False


def summarize(history: list[dict]) -> str:
    """Template a final answer out of whatever the tools actually returned.

    Deliberately plain. We do NOT fabricate reasoning prose here: writing fake
    chain-of-thought would teach attendees to trust a thing that isn't thinking.
    """
    observations = [t for t in history if t.get("role") == "tool"]
    if not observations:
        return "I don't have a tool that can answer that."
    parts = [f"Based on {t['name']}: {t['content']}" for t in observations]
    return " ".join(parts)


def decide(question: str, tools: list[dict], history: list[dict]) -> dict:
    """The whole model, in one function.

    Returns either
        {"kind": "tool_call", "name": ..., "args": {...}, "score": ..., "ranked": [...]}
    or
        {"kind": "final", "text": ..., "ranked": [...]}

    Same inputs always produce the same output. No randomness anywhere.
    """
    ranked = rank_tools(question, tools)
    by_name = {t["name"]: t for t in tools}
    have_result = any(turn.get("role") == "tool" for turn in history)

    for candidate in ranked:
        if not candidate["eligible"]:
            break  # ranked puts all eligible tools first, so we're done

        tool = by_name[candidate["name"]]
        args, used_observation = guess_args(question, tool, history, tools)

        # Two ways to stop, and between them they make the loop terminate:
        #
        # 1. Don't repeat an identical call. Its answer is already in history.
        if already_observed(candidate["name"], args, history):
            continue
        # 2. Once we have ANY result, only keep going for a tool that actually
        #    consumes a previous result. A tool whose arguments come straight
        #    from the question again is just a redundant second lookup of what
        #    we already asked — that is "all relevant results are in history",
        #    so we stop and answer. This is also exactly why the two multi-step
        #    demos in Part 3 keep going: their arguments DO come from prior
        #    observations.
        if have_result and not used_observation:
            break

        return {
            "kind": "tool_call",
            "name": candidate["name"],
            "args": args,
            "score": candidate["score"],
            "ranked": ranked,
        }

    return {"kind": "final", "text": summarize(history), "ranked": ranked}


def estimate_tokens(text: str) -> int:
    """A crude token estimate so the cost counter in Part 3 shows real numbers.

    ~4 characters per token is the usual English rule of thumb. This is an
    estimate for a fake model; it is labelled as such wherever it is printed.
    """
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# ADK adapter — everything below needs google-adk installed.
#
# Guarded so that `import mock_llm` still works with stdlib only, which is what
# makes Part 1 survive a failed pip install.
# --------------------------------------------------------------------------

try:
    from google.adk.models import BaseLlm, LlmRequest, LlmResponse
    from google.genai import types

    _ADK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on broken installs
    _ADK_AVAILABLE = False


if _ADK_AVAILABLE:

    class MockLlm(BaseLlm):
        """Plugs `decide()` into ADK wherever a real Gemini model would go.

        ADK's LlmAgent accepts `model: Union[str, BaseLlm]`, so this class is
        the seam that keeps every downstream notebook cell byte-identical
        between mock and live mode. Verified against google-adk 2.5.0.
        """

        model: str = "mock-deterministic-1"

        async def generate_content_async(
            self, llm_request: "LlmRequest", stream: bool = False
        ):
            """Yield exactly one LlmResponse, same shapes the real model yields.

            ADK hands us the full conversation plus generated tool schemas. We
            translate both into the plain dicts `decide()` understands, then
            translate its answer back into genai Parts.
            """
            tools = _tools_from_request(llm_request)
            question, history = _history_from_request(llm_request)

            decision = decide(question, tools, history)

            if decision["kind"] == "tool_call":
                part = types.Part(
                    function_call=types.FunctionCall(
                        name=decision["name"], args=decision["args"]
                    )
                )
                emitted = decision["name"] + str(decision["args"])
            else:
                part = types.Part(text=decision["text"])
                emitted = decision["text"]

            prompt_text = " ".join(
                p.text or "" for c in (llm_request.contents or []) for p in (c.parts or [])
            )
            prompt_tokens = estimate_tokens(prompt_text)
            output_tokens = estimate_tokens(emitted)

            # Populating usage_metadata is not cosmetic: ADK logs a warning
            # without it, and Part 3's cost counter reads these numbers.
            yield LlmResponse(
                content=types.Content(role="model", parts=[part]),
                partial=False,
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=prompt_tokens,
                    candidates_token_count=output_tokens,
                    total_token_count=prompt_tokens + output_tokens,
                ),
            )

    def _tools_from_request(llm_request: "LlmRequest") -> list[dict]:
        """Pull tool name/description/params out of ADK's generated schemas.

        Note what this proves: the mock reads the *same* function declarations
        ADK generates for Gemini from your type hints and docstring. Nothing is
        hardcoded, which is why a tool you write in Exercise 1 works.
        """
        tools: list[dict] = []
        config = getattr(llm_request, "config", None)
        for tool in getattr(config, "tools", None) or []:
            for decl in getattr(tool, "function_declarations", None) or []:
                tools.append(
                    {
                        "name": decl.name,
                        "description": decl.description or "",
                        "parameters": _params_of(decl),
                    }
                )
        return tools

    def _params_of(decl) -> dict:
        """Normalize a FunctionDeclaration's parameters to {"properties": {...}}.

        ADK 2.5.0 emits `parameters_json_schema` (a plain JSON Schema dict);
        older versions emitted `parameters` (a genai Schema object). We read
        whichever is present so this file survives an ADK upgrade.
        """
        raw = getattr(decl, "parameters_json_schema", None)
        if raw is None:
            schema = getattr(decl, "parameters", None)
            if schema is None:
                return {}
            raw = schema.model_dump(exclude_none=True)

        props = (raw or {}).get("properties") or {}
        return {
            "properties": {
                key: {"type": str((spec or {}).get("type", "string")).lower()}
                for key, spec in props.items()
            }
        }

    def _history_from_request(llm_request: "LlmRequest") -> tuple[str, list[dict]]:
        """Flatten ADK Contents into (first user question, history dicts).

        We keep the FIRST user text as "the question" so scoring stays stable as
        tool results pile up in later turns.
        """
        question = ""
        history: list[dict] = []
        for content in getattr(llm_request, "contents", None) or []:
            for part in getattr(content, "parts", None) or []:
                if part.text and content.role == "user" and not question:
                    question = part.text
                    history.append({"role": "user", "content": part.text})
                elif part.function_call:
                    history.append(
                        {
                            "role": "assistant",
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args or {}),
                        }
                    )
                elif part.function_response:
                    raw = part.function_response.response
                    history.append(
                        {
                            "role": "tool",
                            "name": part.function_response.name,
                            "content": _render(raw),
                            # Keep the structured result too: the next step's
                            # arguments may need a number out of it.
                            "fields": dict(raw) if isinstance(raw, dict) else {},
                            "args": _find_args_for(
                                part.function_response.name, history
                            ),
                        }
                    )
        return question, history

    def _find_args_for(name: str, history: list[dict]) -> dict:
        """Match a tool result back to the args of the call that produced it."""
        for turn in reversed(history):
            if turn.get("role") == "assistant" and turn.get("name") == name:
                return turn.get("args", {})
        return {}

    def _render(response) -> str:
        """Render a tool's return value into the flat text the mock summarizes."""
        if isinstance(response, dict):
            payload = {k: v for k, v in response.items() if k != "status"}
            if len(payload) == 1:
                return str(next(iter(payload.values())))
            return ", ".join(f"{k}={v}" for k, v in payload.items())
        return str(response)
