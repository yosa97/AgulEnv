"""InterCode tool-calling SFT format (PR #1201 protocol).

PR #1201 (commit e276d16b) rewrote validator/evaluation/eval_intercode.py from
ReAct "Action: execute[...]" text parsing to NATIVE tool calling:

  * Tools: execute_bash(command: str) and submit() — exactly one per turn.
  * PER-TURN-FRESH messages (like the PvP rewrite): every turn the model sees
    [system(INTERCODE_TOOL_SYSTEM_PROMPT), user("Question: {q}\\n\\nPrevious
    steps:\\n{history}\\n\\nTurn {n}: call exactly one tool now.")] — history is
    plain text lines "Tool {n}: execute_bash(command=\\"...\\")" /
    "Observation {n}: {obs}" (and "Thought {n}: ..." when the model emitted
    content). Nothing else carries across turns.
  * The episode ends when submit() is called (or max turns -> auto-submit).

So an InterCode task distills into N independent per-turn SFT examples, each
[system, user, assistant(<tool_call> execute_bash/submit)] — mirroring how the
eval rebuilds messages. This module holds the byte-exact system prompt + tool
schemas (verbatim from eval_intercode.py @ #1201 head 933a2682) and the
per-turn example builders. The legacy ReAct builders (intercode_format.py)
remain the bare_id-era surface.

The assistant emits the same Qwen <tool_call> serialized-text surface as the
PvP games (see pvp_tool_calling; converted per model family at tokenize time).
Assistant content carries no free-text thought — "may reason briefly" is
optional at eval, and pure tool calls keep the distilled policy compact.
"""

import json

from our_envs.pvp_tool_calling import _tool_call_content, _structured_tool_call


INTERCODE_EXECUTE_TOOL_NAME = "execute_bash"
INTERCODE_SUBMIT_TOOL_NAME = "submit"

# Verbatim from eval_intercode.py @ #1201 (INTERCODE_TOOL_SYSTEM_PROMPT).
INTERCODE_TOOL_SYSTEM_PROMPT = """Interact with a Bourne Shell system using Bash commands to answer a question.
You may reason briefly in normal assistant text, but every turn must call exactly one tool:
(1) execute_bash, which executes Bash code on the Bourne Shell system
(2) submit, which indicates that the previous observation is the answer
Do not write actions as plain text. Use the tools for every action.

Examples:

Question: Convert the first 16 characters in "/testbed/textfile7.txt" to a single hexadecimal value
Turn 1 tool: execute_bash(command="xxd -p -l 16 /testbed/textfile7.txt")
Observation 1: /bin/bash: line 1: xxd: command not found
Turn 2 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt")
Observation 2:  54 65 78 74 20 66 69 6c 65 20 37 20 77 69 74 68
Turn 3 tool: execute_bash(command="od -A n -t x1 -N 16 /testbed/textfile7.txt | awk '{$1=$1;print}' | tr -d ' '")
Observation 3: 546578742066696c6520372077697468
Turn 4 tool: submit()

Question: print disk usage in human readable format of files or folders in /workspace
Turn 1 tool: execute_bash(command="du /workspace")
Observation 1: 48\t/workspace/dir1
8\t/workspace/dir2/mysql
24\t/workspace/dir2
100\t/workspace
Turn 2 tool: execute_bash(command="du -h /workspace")
Observation 2: 48K\t/workspace/dir1
8.0K\t/workspace/dir2/mysql
24K\t/workspace/dir2
100K\t/workspace
Turn 3 tool: execute_bash(command="du -sh /workspace")
Observation 3: 100K\t/workspace
Turn 4 tool: submit()

Question: Count all the lines of all php files in the /testbed directory recursively
Turn 1 tool: execute_bash(command="find /testbed -name \"*.php\" | xargs wc -l")
Observation 1:  1 /testbed/dir1/info.php
 1 /testbed/hello.php
 2 total
Turn 2 tool: execute_bash(command="find /testbed -name \"*.php\" -type f -exec cat {} + | wc -l")
Observation 2: 2
Turn 3 tool: submit()
"""

# Byte-matched to eval_intercode.py build_intercode_action_tools().
INTERCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": INTERCODE_EXECUTE_TOOL_NAME,
            "description": "Execute one Bash command in the InterCode Bourne Shell environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The Bash command to execute."},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": INTERCODE_SUBMIT_TOOL_NAME,
            "description": "Submit the previous observation as the final answer and end the task.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


def _history_tool_line(turn: int, command: "str | None") -> str:
    """Mirror _tool_call_for_history: execute_bash(command=<json>) or submit()."""
    if command is None:
        return f"Tool {turn}: {INTERCODE_SUBMIT_TOOL_NAME}()"
    return f"Tool {turn}: {INTERCODE_EXECUTE_TOOL_NAME}(command={json.dumps(command)})"


def _user_prompt(query: str, history: "list[str]", turn: int) -> str:
    """Mirror _build_tool_messages' user prompt byte-for-byte."""
    hist = "\n".join(history) if history else "No commands have been executed yet."
    return (
        f"Question: {query}\n\n"
        f"Previous steps:\n{hist}\n\n"
        f"Turn {turn}: call exactly one tool now."
    )


def _execute_call(command: str) -> dict:
    # STRUCTURED tool_calls (fix #3) — same shape as the game envs so the merged
    # cross-env dataset has ONE uniform messages schema (Arrow can't concatenate
    # {role,content,tool_calls} game rows with {role,content} text-intercode rows).
    # apply_chat_template renders the family-native surface at tokenize.
    return {"role": "assistant", "content": None,
            "tool_calls": [_structured_tool_call(INTERCODE_EXECUTE_TOOL_NAME, {"command": command})]}


def _submit_call() -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [_structured_tool_call(INTERCODE_SUBMIT_TOOL_NAME, {})]}


def _turn_example(query: str, history: "list[str]", turn: int, assistant: dict) -> "list[dict]":
    return [
        {"role": "system", "content": INTERCODE_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(query, history, turn)},
        assistant,
    ]


def build_tool_examples_single(query: str, gold: str) -> "list[list[dict]] | None":
    """No-observation path: one turn-1 example (execute the gold)."""
    if not query or not gold:
        return None
    return [_turn_example(query, [], 1, _execute_call(gold))]


def build_tool_examples_multiturn(
    query: str, gold: str, observation: "str | None"
) -> "list[list[dict]] | None":
    """Gold -> observation -> submit, as 2 independent per-turn examples.

    Without a real observation falls back to the single-example shape (we never
    fabricate observations — the submit decision must be conditioned on real
    output).
    """
    if not query or not gold:
        return None
    if not observation:
        return build_tool_examples_single(query, gold)
    ex1 = _turn_example(query, [], 1, _execute_call(gold))
    hist = [_history_tool_line(1, gold), f"Observation 1: {observation}"]
    ex2 = _turn_example(query, hist, 2, _submit_call())
    # Upweight the submit-after-correct-observation turn (1:2 execute:submit) so the
    # model learns to submit decisively the moment the gold answer is on screen,
    # countering the few-shot bias toward extra exploratory commands that overwrite
    # the answer observation and collapse the eval's p3 (stdout/answer similarity).
    return [ex1, ex2, ex2]


def build_tool_examples_refine(
    query: str,
    gold: str,
    obs_imperfect: "str | None",
    obs_gold: "str | None",
    refine_head: str,
) -> "list[list[dict]] | None":
    """Refine-to-exact (IC-2) in tool form: imperfect head -> observe -> exact
    gold -> observe -> submit, as 3 independent per-turn examples. Requires both
    real observations; falls back to multiturn otherwise."""
    if not query or not gold or not refine_head:
        return None
    if not obs_imperfect or not obs_gold:
        return build_tool_examples_multiturn(query, gold, obs_gold)
    ex1 = _turn_example(query, [], 1, _execute_call(refine_head))
    hist2 = [_history_tool_line(1, refine_head), f"Observation 1: {obs_imperfect}"]
    ex2 = _turn_example(query, hist2, 2, _execute_call(gold))
    hist3 = hist2 + [_history_tool_line(2, gold), f"Observation 2: {obs_gold}"]
    ex3 = _turn_example(query, hist3, 3, _submit_call())
    return [ex1, ex2, ex3]
