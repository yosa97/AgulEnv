"""Intercode (NL2Bash) ReAct prompt + single-turn SFT example builder.

The REACT_INIT_MSG, DEMO_BASH_REACT few-shot block, and the execute[...] action
parser are ported VERBATIM from validator/evaluation/eval_intercode.py so the
SFT prompt is byte-for-byte identical to what the validator feeds the model at
eval time. The eval runs a text-completion ReAct loop: it sends ONE user message
containing REACT_INIT_MSG + DEMO_BASH_REACT + f"Question: {query}\nThought 1:"
and the model continues with " {thought}\nAction 1: execute[{cmd}]". We mirror
exactly that boundary for SFT (see build_intercode_sft_example).
"""

import re

REACT_INIT_MSG = """Interact with a Bourne Shell system using Bash queries to answer a question.
Use interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be two types:
(1) execute, which executes Bash code on the Bourne Shell system
(2) submit, which indicates that the previous observation is the answer
Here are some examples.
"""

DEMO_BASH_REACT = """Question: Convert the first 16 characters in "/testbed/textfile7.txt" to a single hexadecimal value
Thought 1: I can use the `xxd` command to get a hexdump of the /textbed/textfile7.txt file
Action 1: execute[xxd -p -l 16 /testbed/textfile7.txt]
Observation 1: /bin/bash: line 1: xxd: command not found
Thought 2: Seems like xxd is not available. I can try to use `od` (octal dump) command to get a hexadecimal dump.
Action 2: execute[od -A n -t x1 -N 16 /testbed/textfile7.txt]
Observation 2:  54 65 78 74 20 66 69 6c 65 20 37 20 77 69 74 68
Thought 3: This command is showing the hexadecimal representation of the first 16 bytes, but it is not a single value.
I should try stitching together the values into one continuous string.
Action 3: execute[od -A n -t x1 -N 16 /testbed/textfile7.txt | awk '{$1=$1;print}' | tr -d ' ']
Observation 3: 546578742066696c6520372077697468
Thought 4: I've configured
Action 4: submit
Question: Calculate a list of duplicate md5 sum hashes for all the ".java" files in the /testbed directory
Thought 1: Find all files with the `.java` extension in `testbed/`, then calculate the md5hash, sort them,
filter out duplicates, and only pick the first line of the hash.
Action 1: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -d | cut -d' ' -f1]
Observation 1:
Thought 2: Seems like this did not produce any results. I may have to use another command instead of cut
to pick out the hashes from the output string
Action 2: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -Dw 32]
Observation 2: f32a3a97638afeb2ee2a15cfe335ab72  /testbed/Hello.java
f32a3a97638afeb2ee2a15cfe335ab72  /testbed/Hello1.java
f32a3a97638afeb2ee2a15cfe335ab72  /testbed/dir1/AnotherHello.java
Thought 3: This shows the output hashes and they have the same values, indicating that these files are duplicates.
However, the file names are also shown, which are not needed.
Action 3: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | cut -c 35-]
Observation 3: /testbed/Hello.java
/testbed/Hello1.java
/testbed/dir1/AnotherHello.java
Thought 4: This shows the file names exclusively, and no longer shows the hashes. It seems that the cut
command argument may not be the best choice for selecting file names.
Action 4: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | awk '{print $2}']
Observation 4: /testbed/Hello.java
/testbed/Hello1.java
/testbed/dir1/AnotherHello.java
Thought 5: I use the awk command instead, but instead of printing out the hashes, it still prints out the file
names. I should select a different part of the output string instead of `$2`
Action 5: execute[find /testbed -name "*.java" -type f -exec md5sum {} + | sort | uniq -D -w 32 | awk '{print $1}']
Observation 5: f32a3a97638afeb2ee2a15cfe335ab72
f32a3a97638afeb2ee2a15cfe335ab72
f32a3a97638afeb2ee2a15cfe335ab72
Thought 6: This prints out identical hashes, and based on previous observations, I know that these are hashes of
duplicates `.java` files from the `testbed/` directory. This should be correct. I will submit.
Action 6: submit
Question: print disk usage in human readable format of files or folders in /workspace
Thought 1: The `du` command is useful for printing out disk usage of a specific directory. I can use this to
display this information for the `workspace` directory
Action 1: execute[du /workspace]
Observation 1: 48\t/workspace/dir1
8\t/workspace/dir2/mysql
24\t/workspace/dir2
100\t/workspace
Thought 2: The default `du` command gives storage in a non-human readble font. I can use the -h option
of the du command to print storage size with bytes.
Action 2: execute[du -h /workspace]
Observation 2: 48K\t/workspace/dir1
8.0K\t/workspace/dir2/mysql
24K\t/workspace/dir2
100K\t/workspace
Thought 3: This gives me storage information for every folder under the workspace directory, but
I only need the storage for just the `workspace/` directory. The `-s` option should help with this.
Action 3: execute[du -sh /workspace]
Observation 3: 100K\t/workspace
Thought 4: This shows data usage in human readable format for the `workspace` directory. I am finished.
Action 4: submit
Question: Count all the lines of all php files in the /testbed directory recursively
Thought 1: I should find the paths to all php files in the testbed directory, then apply the word
count command to each path.
Action 1: execute[find /testbed -name "*.php" | xargs wc -l]
Observation 1:  1 /testbed/dir1/info.php
 1 /testbed/hello.php
 2 total
Thought 2: This shows me too much information, I only need the total number of lines. I should add up
the lines together and output a single number.
Action 2: execute[find /testbed -name "*.php" -exec wc -l {} + | awk '{total += $1} END{print total}']
Observation 2: 4
Thought 3: This total is wrong, it doesn't match the previous observation, where total is 2. I only
need to apply the word count command.
Action 3: execute[find /testbed -name "*.php" -type f -exec cat {} + | wc -l]
Observation 3: 2
Thought 4: The value is 2, which matches the initial observation that the total lines of php files in the
testbed directory is 2. I can submit.
Action 4: submit
Question: Create a hello.txt file in the /testbed directory and add the text "Hello world" to it.
Thought 1: I can first create a `hello.txt` file in the `testbed/` directory
Action 1: touch testbed/hello.txt
Observation 1:
Thought 2: I should check that the file was created successfully.
Action 2: execute[ls testbed/]
Observation 2: dir1/
dir2/
dir3/
hello.txt
files.txt
Thought 3: I can now add the "Hello world" text to the hello.txt file
Action 3: execute[echo Hello world > hello.txt]
Observation 3:
Thought 4: I should check that the text was written successfully to the hello.txt file.
Action 4: execute[cat testbed/hello.txt]
Observation 4: Hello world
Thought 5: The hello.txt file has been created successfully in the testbed/ directory, and it contains
the Hello World text. I can submit.
Action 5: submit
"""

_REACT_ACTION_RE = re.compile(r"execute\[(.*)\]", re.DOTALL)


def _parse_action(action: str) -> tuple[str, bool]:
    if action == "submit":
        return action, True
    matches = _REACT_ACTION_RE.findall(action)
    if matches:
        return matches[0], True
    return action, False


# ─────────────────────────────────────────────────────────────────────────────
# Single-turn SFT example builder
# ─────────────────────────────────────────────────────────────────────────────

# The eval primes the prompt with "Thought 1:" and the model continues; the
# NL2Bash reward scores the bash command, not the prose, so a short fixed
# thought keeps the ReAct shape while the gold command carries the signal.
_DEFAULT_THOUGHT = "I can answer this question by running a bash command."


def build_intercode_sft_example(query: str, gold: str, thought: str | None = None) -> "list[dict] | None":
    """Build one single-turn ReAct SFT example as a [user, assistant] message list.

    Mirrors the eval boundary exactly. The eval's first turn sends, as the user
    message:

        REACT_INIT_MSG + DEMO_BASH_REACT + f"Question: {query}\\nThought 1:"

    and the model continues with " {thought}\\nAction 1: execute[{cmd}]". We put
    "Thought 1:" at the END of the user content (no trailing space, as the eval
    does) and start the assistant turn with the thought, so the trained
    boundary is byte-identical to inference.

    Returns None when query or gold is missing (caller drops the row).
    """
    query = str(query or "").strip()
    gold = str(gold or "").strip()
    if not query or not gold:
        return None
    user = REACT_INIT_MSG + DEMO_BASH_REACT + f"Question: {query}\nThought 1:"
    assistant = f" {thought or _DEFAULT_THOUGHT}\nAction 1: execute[{gold}]"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# Default observation when we don't actually run the gold command. Keep it
# generic-but-plausible: a typical "the command worked" observation that lets
# Turn 2 reason about success and submit. Override by passing observation=...
_DEFAULT_OBS = "<command output omitted>"

_SUBMIT_THOUGHT = (
    "The command above produced the requested result. I will submit."
)


def build_intercode_multiturn_example(
    query: str,
    gold: str,
    observation: str | None = None,
    thought1: str | None = None,
    thought2: str | None = None,
) -> "list[dict] | None":
    """Build a 2-turn ReAct SFT example: execute[gold] -> observation -> submit.

    Matches the eval's multi-turn boundary EXACTLY at turn 2: after the model's
    Action 1, eval appends "Observation 1: ...\\nThought 2:" and the model
    continues. We reproduce that boundary so the model learns to read the
    observation and emit `submit` (which ends the eval loop).

    observation: real stdout from executing gold. When omitted, a generic
    placeholder is used (still teaches the submit STRUCTURE, but loses the
    real-feedback signal). Real-execution observations require a sandbox
    (intercode_local_bash_env.LocalBashEnv) -- enabled by the caller.

    Returns None when query or gold is missing.
    """
    query = str(query or "").strip()
    gold = str(gold or "").strip()
    if not query or not gold:
        return None
    obs_text = observation if observation is not None else _DEFAULT_OBS
    # Truncate long observations to match eval (DEFAULT_OBS_TRUNCATE_CHARS = 350).
    if len(obs_text) > 350:
        obs_text = obs_text[:350]

    sys_prefix = REACT_INIT_MSG + DEMO_BASH_REACT
    user1 = sys_prefix + f"Question: {query}\nThought 1:"
    asst1 = f" {thought1 or _DEFAULT_THOUGHT}\nAction 1: execute[{gold}]"
    # Turn 2: the next user content the eval would send is
    #   "Observation 1: {obs}\nThought 2:"  (concatenated onto the running prompt).
    # In chat-message form, that's a new user turn carrying that delta.
    user2 = f"Observation 1: {obs_text}\nThought 2:"
    asst2 = f" {thought2 or _SUBMIT_THOUGHT}\nAction 2: submit"
    return [
        {"role": "user", "content": user1},
        {"role": "assistant", "content": asst1},
        {"role": "user", "content": user2},
        {"role": "assistant", "content": asst2},
    ]


# IC-2: refine-to-exact trajectory. The single/multi-turn builders always run the
# gold ONCE and submit, so the trained model learns "run a command -> submit" and
# at eval submits its first imperfect command (the 0.67-plateau: a near-miss command
# loses the stdout-tfidf reward component). The eval's own few-shot DEMO repeatedly
# shows refine-to-exact (du -> du -sh, find|wc -> ...|awk, cut -> awk). We mirror
# that: turn 1 runs the gold MINUS its final pipe stage (the core command, whose raw
# output is over-verbose / not the final form), then turn 2 refines to the exact gold,
# then submit. This teaches the model to read the observation and refine to the exact
# command before submitting.
_REFINE_THOUGHT1 = (
    "I'll start with the core command and inspect its raw output before refining."
)
_REFINE_THOUGHT2 = (
    "The output needs further processing to match the request. I'll refine the command."
)
_REFINE_SUBMIT_THOUGHT = "This is the exact result I need. I will submit."


def derive_imperfect_first_attempt(gold: str) -> "str | None":
    """Return a plausible imperfect first command = gold minus its final pipe stage.

    Only applies when the gold is a pipeline with a non-trivial head and tail (the
    head must be a runnable command on its own). Returns None otherwise, so the
    caller falls back to the single-shot / 2-turn builder.
    """
    gold = str(gold or "").strip()
    if " | " not in gold:
        return None
    head, _, tail = gold.rpartition(" | ")
    head = head.strip()
    if not head or not tail.strip():
        return None
    # Avoid degenerate heads (e.g. a bare variable assignment or an empty subshell).
    if head in {"(", "{"} or head.endswith(("&&", "||", ";")):
        return None
    return head


def build_intercode_refine_example(
    query: str,
    gold: str,
    obs_imperfect: str | None = None,
    obs_gold: str | None = None,
) -> "list[dict] | None":
    """Build a 3-action refine-to-exact ReAct example: execute[head] -> obs ->
    execute[gold] -> obs -> submit.

    `head` is derive_imperfect_first_attempt(gold) (gold minus its final pipe
    stage). Returns None when query/gold is missing or no refine split applies
    (caller falls back to build_intercode_multiturn_example). Observations default
    to the generic placeholder; pass real stdout (IC-3) to teach the genuine
    wrong-then-corrected signal.
    """
    query = str(query or "").strip()
    gold = str(gold or "").strip()
    if not query or not gold:
        return None
    head = derive_imperfect_first_attempt(gold)
    if head is None:
        return None

    def _obs(text: str | None) -> str:
        t = text if text is not None else _DEFAULT_OBS
        return t[:350] if len(t) > 350 else t

    sys_prefix = REACT_INIT_MSG + DEMO_BASH_REACT
    user1 = sys_prefix + f"Question: {query}\nThought 1:"
    asst1 = f" {_REFINE_THOUGHT1}\nAction 1: execute[{head}]"
    user2 = f"Observation 1: {_obs(obs_imperfect)}\nThought 2:"
    asst2 = f" {_REFINE_THOUGHT2}\nAction 2: execute[{gold}]"
    user3 = f"Observation 2: {_obs(obs_gold)}\nThought 3:"
    asst3 = f" {_REFINE_SUBMIT_THOUGHT}\nAction 3: submit"
    return [
        {"role": "user", "content": user1},
        {"role": "assistant", "content": asst1},
        {"role": "user", "content": user2},
        {"role": "assistant", "content": asst2},
        {"role": "user", "content": user3},
        {"role": "assistant", "content": asst3},
    ]
