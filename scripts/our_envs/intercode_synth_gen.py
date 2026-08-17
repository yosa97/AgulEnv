"""Eval-aligned SYNTHETIC NL2Bash task generator for InterCode SFT.

Why this exists: the whitelisted dataset gradients-io-tournaments/intercode_
bigcode_combined_12k is a poor proxy for the validator's InterCode eval — half is
BigCode Python (off-task, dropped by the bash filter) and the NL2Bash half is
generic Tellina commands NOT grounded on the eval's specific filesystems (so
their golds produce empty/garbage output when run on /testbed//system). The eval
grades by RUNNING the gold on a SPECIFIC baked filesystem and comparing the
agent's fs-diff + stdout to the gold's (validator/evaluation/eval_intercode.py).

This generator instead builds tasks GROUNDED on the REAL baked eval filesystems
(/intercode_fs/fs{1,2}.tar = /testbed, /system; fs4 = string/stdin tasks), with
golds that actually run, then distills each into a multi-turn ReAct trajectory via
REAL execution — exactly the eval's distribution. fs3 (/workspace) is skipped
(it manages the trainer's own working dir).

Legal: we already bake /intercode_fs (the env, like the game env-servers — see
intercode_local_bash_env docstring); the tasks + golds are GENERATED here
(synthetic, like the LD/LP/GR experts), not a dataset. No 12k, no non-whitelist
data, no internet, never-raises.

Usage:
    INTERCODE_FS_ROOT=/intercode_fs python -m our_envs.intercode_synth_gen \\
        --output_path /workspace/scripts/datasets/intercode_synth --per_fs 4000
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict

from our_envs.intercode_local_bash_env import LocalBashEnv, is_safe_for_real_exec
from our_envs.intercode_tool_format import build_tool_examples_multiturn


VALIDATION_RATIO = 0.01

# fs we can generate on: 1=/testbed, 2=/system (real files); 4=string/stdin (no
# fs). fs3=/workspace is excluded — it manages the trainer's own working dir.
_FS_ROOTS: dict[int, str] = {1: "/testbed", 2: "/system"}
_GEN_FS = (1, 2, 4)

# Word pool for fs4 string-manipulation tasks (kept single-quote-free so the
# string embeds safely inside echo '...').
_WORDS = (
    "hello", "world", "foo", "bar", "data", "file", "system", "value", "result",
    "input", "output", "config", "test", "alpha", "beta", "gamma", "node", "edge",
    "Hello", "World", "Backup", "Report", "Summary", "Final", "Draft",
)
_SEPS = (":", ",", "-", ";", "|", " ", ".")


# ---------------------------------------------------------------------------
# Filesystem inventory (built once per fs from the restored snapshot)
# ---------------------------------------------------------------------------

@dataclass
class Inv:
    root: str
    files: list[str]
    dirs: list[str]
    by_ext: dict[str, list[str]]
    text_files: list[str]
    ext_dirs: dict[str, list[str]]  # ext -> dirs that contain >=1 file of it


def _walk_inventory(root: str) -> "Inv | None":
    if not os.path.isdir(root):
        return None
    files: list[str] = []
    dirs: list[str] = []
    by_ext: dict[str, list[str]] = defaultdict(list)
    ext_dirs: dict[str, set[str]] = defaultdict(set)
    text_files: list[str] = []
    for cur, subdirs, names in os.walk(root):
        dirs.append(cur)
        for n in names:
            full = os.path.join(cur, n)
            files.append(full)
            base = os.path.basename(full)
            if "." in base:
                ext = base.rsplit(".", 1)[-1].lower()
                if ext and len(ext) <= 6 and ext.isalnum():
                    by_ext[ext].append(full)
                    ext_dirs[ext].add(cur)
                    ext_dirs[ext].add(root)
            if base.endswith((".txt", ".log", ".csv", ".md", ".sql")):
                text_files.append(full)
    if not files:
        return None
    return Inv(
        root=root,
        files=files,
        dirs=dirs,
        by_ext=dict(by_ext),
        text_files=text_files,
        ext_dirs={k: sorted(v) for k, v in ext_dirs.items()},
    )


# ---------------------------------------------------------------------------
# Task templates — each returns (query, gold) grounded on the inventory, or None.
# Non-mutating (read/string) unless mutates=True. Golds use ABSOLUTE paths only
# (the LocalBashEnv cwd is a throwaway sandbox; absolute paths hit the restored fs).
# ---------------------------------------------------------------------------

@dataclass
class Template:
    name: str
    fs: set[int]
    mutates: bool
    fn: object  # (inv, rng) -> (query, gold) | None


def _pick_ext(inv, rng):
    return rng.choice(list(inv.by_ext)) if inv.by_ext else None


def _pick_dir(inv, rng):
    return rng.choice(inv.dirs) if inv.dirs else inv.root


def _rand_string(rng) -> str:
    body = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 5)))
    return " " * rng.randint(1, 3) + body + " " * rng.randint(1, 3)


# --- fs1/fs2 read templates ------------------------------------------------

def t_count_lines_ext(inv, rng):
    ext = _pick_ext(inv, rng)
    if not ext:
        return None
    d = rng.choice(inv.ext_dirs.get(ext, [inv.root]))
    return (
        f"Count the total number of lines across all {ext} files under '{d}' (recursively).",
        f"find '{d}' -type f -name '*.{ext}' -exec cat {{}} + | wc -l",
    )


def t_count_files_ext(inv, rng):
    ext = _pick_ext(inv, rng)
    if not ext:
        return None
    d = rng.choice(inv.ext_dirs.get(ext, [inv.root]))
    return (
        f"Count how many {ext} files there are under '{d}' (recursively).",
        f"find '{d}' -type f -name '*.{ext}' | wc -l",
    )


def t_find_files_ext(inv, rng):
    ext = _pick_ext(inv, rng)
    if not ext:
        return None
    d = rng.choice(inv.ext_dirs.get(ext, [inv.root]))
    return (
        f"List the paths of all {ext} files under '{d}' (recursively).",
        f"find '{d}' -type f -name '*.{ext}'",
    )


def t_find_latest(inv, rng):
    d = _pick_dir(inv, rng)
    return (
        f"Recursively find the most recently modified file in '{d}'.",
        f"find '{d}' -type f -printf '%T@ %p\\n' | sort -n | tail -1 | cut -d' ' -f2-",
    )


def t_find_largest(inv, rng):
    d = _pick_dir(inv, rng)
    return (
        f"Find the largest file (by size) under '{d}'.",
        f"find '{d}' -type f -printf '%s %p\\n' | sort -n | tail -1 | cut -d' ' -f2-",
    )


def t_count_extensions(inv, rng):
    d = _pick_dir(inv, rng)
    return (
        f"Count the number of unique file extensions under '{d}'.",
        f"find '{d}' -type f -name '*.*' | awk -F. '{{print $NF}}' | sort -u | wc -l",
    )


def t_count_files(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"Count the total number of files under '{d}'.", f"find '{d}' -type f | wc -l")


def t_count_dirs(inv, rng):
    d = _pick_dir(inv, rng)
    return (
        f"Count the number of subdirectories under '{d}'.",
        f"find '{d}' -mindepth 1 -type d | wc -l",
    )


def t_find_empty(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"Find all empty files under '{d}'.", f"find '{d}' -type f -empty")


def t_files_with_spaces(inv, rng):
    return (
        f"Find all files whose names contain a space under '{inv.root}'.",
        f"find '{inv.root}' -type f -name '* *'",
    )


def t_list_subdirs(inv, rng):
    d = _pick_dir(inv, rng)
    return (
        f"List the immediate subdirectories of '{d}'.",
        f"find '{d}' -mindepth 1 -maxdepth 1 -type d",
    )


def t_find_by_name(inv, rng):
    if not inv.files:
        return None
    name = os.path.basename(rng.choice(inv.files))
    return (
        f"Find the full path(s) of the file named '{name}' under '{inv.root}'.",
        f"find '{inv.root}' -name '{name}'",
    )


# --- fs1/fs2 file-content read templates -----------------------------------

def t_last_line(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (
        f"Print the last non-empty line of '{f}'.",
        f"grep -v '^[[:space:]]*$' '{f}' | tail -1",
    )


def t_first_line(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Print the first line of '{f}'.", f"head -1 '{f}'")


def t_line_count_file(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Count the number of lines in '{f}'.", f"wc -l < '{f}'")


def t_word_count_file(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Count the number of words in '{f}'.", f"wc -w < '{f}'")


# --- fs4 string / stdin templates ------------------------------------------

def t_trim_spaces(inv, rng):
    s = _rand_string(rng)
    return (
        f"Remove the leading and trailing whitespace from the string '{s}'.",
        f"echo '{s}' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'",
    )


def t_seq_sep(inv, rng):
    n = rng.randint(5, 20)
    sep = rng.choice(_SEPS)
    return (
        f"Print the numbers 1 through {n} separated by '{sep}'.",
        f"seq 1 {n} | paste -sd '{sep}' -",
    )


def t_upper(inv, rng):
    s = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 4)))
    return (f"Convert the string '{s}' to uppercase.", f"echo '{s}' | tr '[:lower:]' '[:upper:]'")


def t_reverse(inv, rng):
    s = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 4)))
    return (f"Reverse the characters of the string '{s}'.", f"echo '{s}' | rev")


def t_count_words_str(inv, rng):
    s = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 7)))
    return (f"Count the number of words in the string '{s}'.", f"echo '{s}' | wc -w")


# --- string-search OUTPUT SHAPE (fs1/fs2) ----------------------------------
# The eval scores a read-only task as 0.01 + 0.33 (fs-diff, free) + 0.33 (content,
# free) + 0.33 * tfidf_cosine(our stdout, gold stdout) — so on these tasks the ONLY
# thing being graded is the printed TEXT (validator/evaluation/evaluators/intercode.py).
# That makes "which shape of answer" a cliff, not a nuance: asked for file NAMES,
# printing matching LINES shares almost no tokens with the gold and scores ~0 on the
# only component in play (0.67 total), even though the search itself was right. Our
# eval log shows exactly that split — "show matched lines with line numbers" scored
# 1.00 while "print all filenames ... containing 'hello', case-insensitive" scored
# 0.67. We had no grep -l template at all. These teach the query->shape mapping as
# contrasting pairs so the distinction is learnable rather than guessed.

def _grep_dir_and_needle(inv, rng):
    if not inv.text_files:
        return None, None
    f = rng.choice(inv.text_files)
    try:
        with open(f, "r", errors="replace") as fh:
            words = [w for w in re.findall(r"[A-Za-z]{3,}", fh.read(4096))]
    except OSError:
        return None, None
    if not words:
        return None, None
    d = rng.choice(inv.dirs or [inv.root])
    return d, rng.choice(words)


def t_grep_filenames(inv, rng):
    """NAMES of matching files -> grep -l (the shape we had no template for)."""
    d, needle = _grep_dir_and_needle(inv, rng)
    if not d:
        return None
    return (
        f"Print the names of all files under '{d}' that contain the string '{needle}'.",
        f"grep -rl '{needle}' '{d}'",
    )


def t_grep_filenames_icase(inv, rng):
    """Same, case-insensitive -> the -i flag changes WHICH files print."""
    d, needle = _grep_dir_and_needle(inv, rng)
    if not d:
        return None
    return (
        f"Print the names of all files under '{d}' containing '{needle.lower()}', case-insensitively.",
        f"grep -ril '{needle.lower()}' '{d}'",
    )


def t_grep_lines(inv, rng):
    """The CONTRAST: matching LINES, not names -> no -l."""
    d, needle = _grep_dir_and_needle(inv, rng)
    if not d:
        return None
    return (
        f"Print the lines containing '{needle}' in the files under '{d}'.",
        f"grep -r '{needle}' '{d}'",
    )


def t_grep_lines_numbered(inv, rng):
    d, needle = _grep_dir_and_needle(inv, rng)
    if not d:
        return None
    return (
        f"Search for '{needle}' case-insensitively under '{d}' and show the matched lines with their line numbers.",
        f"grep -rin '{needle}' '{d}'",
    )


def t_grep_count_matches(inv, rng):
    """A COUNT prints one number — a third distinct shape from names/lines."""
    d, needle = _grep_dir_and_needle(inv, rng)
    if not d:
        return None
    return (
        f"Count how many lines under '{d}' contain '{needle}'.",
        f"grep -r '{needle}' '{d}' | wc -l",
    )


def t_list_all_incl_hidden(inv, rng):
    """Hidden entries change the printed set — our eval log scored this family 0.77."""
    d = rng.choice(inv.dirs or [inv.root])
    return (
        f"Recursively print all files and directories in the '{d}' directory tree, including hidden ones.",
        f"find '{d}'",
    )


# --- system info + shell variables (fs4: filesystem-agnostic) ---------------
# We emitted ZERO of these, and the eval's "Store system load average number in the
# 'proc_load_average' variable and print it" scored 0.67 = sim 0.00: with no example
# of "assign a command's output to a variable, then echo it" the model cannot produce
# the gold's shape (a bare value) at all. These teach the CAPABILITY with varied names
# and sources — deliberately NOT the eval's specific task, which we must not fit to.
# All are read-only (is_safe_for_real_exec allows reads of absolute paths), and the
# sources are stable ones: a value that churns between the gold's run and the agent's
# would cost similarity even when the command is right.

_VAR_NAMES = ("result", "value", "output", "count_var", "info_var", "sys_value", "current_val", "data_var")

_SYS_SOURCES = (
    ("the system load average (1 minute)", "cut -d' ' -f1 /proc/loadavg"),
    ("the system's hostname", "cat /proc/sys/kernel/hostname"),
    ("the kernel version", "cut -d' ' -f3 /proc/version"),
    ("the number of CPU cores", "grep -c '^processor' /proc/cpuinfo"),
    ("the total memory in kB", "grep MemTotal /proc/meminfo | awk '{print $2}'"),
    ("the system uptime in seconds", "cut -d' ' -f1 /proc/uptime"),
)


def t_sysinfo_print(inv, rng):
    what, cmd = rng.choice(_SYS_SOURCES)
    return (f"Print {what}.", cmd)


def t_sysinfo_to_variable(inv, rng):
    """The missing shape: capture into a named variable, then print the variable."""
    what, cmd = rng.choice(_SYS_SOURCES)
    var = rng.choice(_VAR_NAMES)
    return (
        f"Store {what} in the '{var}' variable and print it.",
        f"{var}=$({cmd}); echo ${var}",
    )


def t_var_from_count(inv, rng):
    """Same variable mechanic, but over a filesystem count (bridges both families)."""
    d = rng.choice(inv.dirs or [inv.root]) if inv.dirs or inv.root else None
    if not d:
        return None
    var = rng.choice(_VAR_NAMES)
    return (
        f"Store the number of files under '{d}' in the '{var}' variable and print it.",
        f"{var}=$(find '{d}' -type f | wc -l); echo ${var}",
    )


def t_date_format(inv, rng):
    fmt, desc = rng.choice(
        (("+%Y", "the current year"), ("+%Y-%m", "the current year and month"), ("+%A", "the current day of the week")),
    )
    return (f"Print {desc}.", f"date '{fmt}'")


def t_arith_expr(inv, rng):
    a, b = rng.randint(11, 99), rng.randint(2, 9)
    op, word = rng.choice((("+", "sum"), ("*", "product"), ("-", "difference")))
    return (f"Print the {word} of {a} and {b}.", f"echo $(({a} {op} {b}))")


# --- mutating (fs1/fs2; the env is restored before each) -------------------

def t_strip_trailing_ws(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (
        f"Remove trailing whitespace from every line of '{f}' (edit the file in place).",
        f"sed -i 's/[[:space:]]*$//' '{f}'",
    )


# --- expansion v1.1: more fs operation families (size/time/content/structure) +
#     more per-file/per-string variety, to scale grounded volume + match the eval
#     distribution (find-by-perm/time/size, content edits, symlinks). ------------

def t_total_size(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"Compute the total size in bytes of all files under '{d}'.",
            f"find '{d}' -type f -printf '%s\\n' | awk '{{s+=$1}} END {{print s}}'")


def t_smallest_file(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"Find the smallest non-empty file under '{d}'.",
            f"find '{d}' -type f ! -empty -printf '%s %p\\n' | sort -n | head -1 | cut -d' ' -f2-")


def t_find_larger(inv, rng):
    d = _pick_dir(inv, rng)
    n = rng.choice((1, 10, 50, 100))
    return (f"Find all files larger than {n} bytes under '{d}'.",
            f"find '{d}' -type f -size +{n}c")


def t_newest_n(inv, rng):
    d = _pick_dir(inv, rng)
    n = rng.randint(2, 4)
    return (f"List the {n} most recently modified files under '{d}'.",
            f"find '{d}' -type f -printf '%T@ %p\\n' | sort -rn | head -{n} | cut -d' ' -f2-")


def t_find_symlinks(inv, rng):
    return (f"Find all symbolic links under '{inv.root}'.", f"find '{inv.root}' -type l")


def t_find_executable(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"List all files with the execute permission set under '{d}'.",
            f"find '{d}' -type f -perm -u+x")


def t_count_lines_all_txt(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"Count the total number of lines across all .txt files under '{d}' (recursively).",
            f"find '{d}' -name '*.txt' -type f -exec cat {{}} + | wc -l")


def t_count_words_all_txt(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"Count the total number of words across all .txt files under '{d}'.",
            f"find '{d}' -name '*.txt' -type f -exec cat {{}} + | wc -w")


def t_list_extensions(inv, rng):
    d = _pick_dir(inv, rng)
    return (f"List the unique file extensions present under '{d}', sorted.",
            f"find '{d}' -type f -name '*.*' | awk -F. '{{print $NF}}' | sort -u")


def t_deepest_file(inv, rng):
    return (f"Find the file with the deepest (most nested) path under '{inv.root}'.",
            f"find '{inv.root}' -type f | awk -F/ '{{print NF, $0}}' | sort -rn | head -1 | cut -d' ' -f2-")


def t_count_at_depth(inv, rng):
    d = _pick_dir(inv, rng)
    n = rng.randint(1, 3)
    return (f"Count the number of files exactly {n} level(s) deep under '{d}'.",
            f"find '{d}' -mindepth {n} -maxdepth {n} -type f | wc -l")


def t_char_count_file(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Count the number of characters (bytes) in '{f}'.", f"wc -c < '{f}'")


def t_nonempty_lines(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Count the number of non-empty lines in '{f}'.", f"grep -c '[^[:space:]]' '{f}'")


def t_longest_line(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Print the longest line in '{f}'.",
            f"awk '{{ if (length > m) {{ m = length; l = $0 }} }} END {{ print l }}' '{f}'")


def t_basename(inv, rng):
    if not inv.files:
        return None
    f = rng.choice(inv.files)
    return (f"Print only the file name (strip the directory path) of '{f}'.", f"basename '{f}'")


def t_dirname(inv, rng):
    if not inv.files:
        return None
    f = rng.choice(inv.files)
    return (f"Print the directory that contains '{f}'.", f"dirname '{f}'")


# --- more fs4 string templates ---------------------------------------------

def t_replace_word(inv, rng):
    words = [rng.choice(_WORDS) for _ in range(rng.randint(3, 5))]
    a = rng.choice(words)
    b = rng.choice(_WORDS)
    s = " ".join(words)
    return (f"In the string '{s}', replace every occurrence of '{a}' with '{b}'.",
            f"echo '{s}' | sed 's/{a}/{b}/g'")


def t_sort_words(inv, rng):
    s = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 6)))
    return (f"Sort the words in the string '{s}' alphabetically.",
            f"echo '{s}' | tr ' ' '\\n' | sort | paste -sd ' ' -")


def t_first_word(inv, rng):
    s = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 5)))
    return (f"Print the first word of the string '{s}'.", f"echo '{s}' | awk '{{print $1}}'")


def t_last_word(inv, rng):
    s = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 5)))
    return (f"Print the last word of the string '{s}'.", f"echo '{s}' | awk '{{print $NF}}'")


def t_char_count_str(inv, rng):
    s = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 5)))
    return (f"Count the number of characters in the string '{s}'.", f"printf '%s' '{s}' | wc -c")


# --- more mutating (fs1/fs2; env restored before each) ----------------------

def t_remove_blank_lines(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Remove all blank lines from '{f}' (edit the file in place).",
            f"sed -i '/^[[:space:]]*$/d' '{f}'")


def t_uppercase_file(inv, rng):
    if not inv.text_files:
        return None
    f = rng.choice(inv.text_files)
    return (f"Convert the contents of '{f}' to uppercase (edit the file in place).",
            f"tr '[:lower:]' '[:upper:]' < '{f}' > /tmp/ic_x && mv /tmp/ic_x '{f}'")


_TEMPLATES: list[Template] = [
    # Output-SHAPE contrasts: names vs lines vs count for the same search (see the
    # block above — shape is the whole score on a read-only task).
    Template("grep_filenames", {1, 2}, False, t_grep_filenames),
    Template("grep_filenames_icase", {1, 2}, False, t_grep_filenames_icase),
    Template("grep_lines", {1, 2}, False, t_grep_lines),
    Template("grep_lines_numbered", {1, 2}, False, t_grep_lines_numbered),
    Template("grep_count_matches", {1, 2}, False, t_grep_count_matches),
    Template("list_all_incl_hidden", {1, 2}, False, t_list_all_incl_hidden),
    # System info + shell variables — the family we emitted none of.
    Template("sysinfo_print", {4}, False, t_sysinfo_print),
    Template("sysinfo_to_variable", {4}, False, t_sysinfo_to_variable),
    Template("var_from_count", {1, 2}, False, t_var_from_count),
    Template("date_format", {4}, False, t_date_format),
    Template("arith_expr", {4}, False, t_arith_expr),
    Template("count_lines_ext", {1, 2}, False, t_count_lines_ext),
    Template("count_files_ext", {1, 2}, False, t_count_files_ext),
    Template("find_files_ext", {1, 2}, False, t_find_files_ext),
    Template("find_latest", {1, 2}, False, t_find_latest),
    Template("find_largest", {1, 2}, False, t_find_largest),
    Template("count_extensions", {1, 2}, False, t_count_extensions),
    Template("count_files", {1, 2}, False, t_count_files),
    Template("count_dirs", {1, 2}, False, t_count_dirs),
    Template("find_empty", {1, 2}, False, t_find_empty),
    Template("files_with_spaces", {1, 2}, False, t_files_with_spaces),
    Template("list_subdirs", {1, 2}, False, t_list_subdirs),
    Template("find_by_name", {1, 2}, False, t_find_by_name),
    Template("last_line", {1, 2}, False, t_last_line),
    Template("first_line", {1, 2}, False, t_first_line),
    Template("line_count_file", {1, 2}, False, t_line_count_file),
    Template("word_count_file", {1, 2}, False, t_word_count_file),
    Template("strip_trailing_ws", {1, 2}, True, t_strip_trailing_ws),
    Template("trim_spaces", {4}, False, t_trim_spaces),
    Template("seq_sep", {4}, False, t_seq_sep),
    Template("upper", {4}, False, t_upper),
    Template("reverse", {4}, False, t_reverse),
    Template("count_words_str", {4}, False, t_count_words_str),
    # --- expansion v1.1 ---
    Template("total_size", {1, 2}, False, t_total_size),
    Template("smallest_file", {1, 2}, False, t_smallest_file),
    Template("find_larger", {1, 2}, False, t_find_larger),
    Template("newest_n", {1, 2}, False, t_newest_n),
    Template("find_symlinks", {1, 2}, False, t_find_symlinks),
    Template("find_executable", {1, 2}, False, t_find_executable),
    Template("count_lines_all_txt", {1, 2}, False, t_count_lines_all_txt),
    Template("count_words_all_txt", {1, 2}, False, t_count_words_all_txt),
    Template("list_extensions", {1, 2}, False, t_list_extensions),
    Template("deepest_file", {1, 2}, False, t_deepest_file),
    Template("count_at_depth", {1, 2}, False, t_count_at_depth),
    Template("char_count_file", {1, 2}, False, t_char_count_file),
    Template("nonempty_lines", {1, 2}, False, t_nonempty_lines),
    Template("longest_line", {1, 2}, False, t_longest_line),
    Template("basename", {1, 2}, False, t_basename),
    Template("dirname", {1, 2}, False, t_dirname),
    Template("replace_word", {4}, False, t_replace_word),
    Template("sort_words", {4}, False, t_sort_words),
    Template("first_word", {4}, False, t_first_word),
    Template("last_word", {4}, False, t_last_word),
    Template("char_count_str", {4}, False, t_char_count_str),
    Template("remove_blank_lines", {1, 2}, True, t_remove_blank_lines),
    Template("uppercase_file", {1, 2}, True, t_uppercase_file),
]


# ---------------------------------------------------------------------------
# Output quality gate
# ---------------------------------------------------------------------------

_ERROR_MARKERS = (
    "command not found", "No such file", "Permission denied", "syntax error",
    "Execution error", "Command timed out", "cannot ", "illegal ", "invalid ",
)


def _meaningful(obs: str, mutates: bool = False) -> bool:
    """Keep tasks whose gold ran cleanly + produced a usable result. READ tasks
    need a non-empty observation; MUTATING tasks (in-place edits) carry their
    result in the FILESYSTEM change, so an empty stdout is fine as long as the
    command did not error."""
    s = (obs or "").strip()
    low = s.lower()
    if any(m.lower() in low for m in _ERROR_MARKERS):
        return False
    return True if mutates else bool(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", required=True)
    p.add_argument("--per_fs", type=int, default=4000, help="tasks attempted per fs variant.")
    p.add_argument("--max_per_fs", type=int, default=0,
                   help="cap PRODUCED tasks per fs (0=no cap); balances fs4 string tasks vs fs1/fs2.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot_root", default=os.environ.get("INTERCODE_FS_ROOT", "/intercode_fs"))
    args = p.parse_args()

    snapshot_root = Path(args.snapshot_root)
    rng = random.Random(args.seed)
    walltime_budget = float(os.environ.get("INTERCODE_GEN_WALLTIME_SEC", "2700"))
    start_ts = time.monotonic()

    examples: list[dict] = []
    built = 0
    dropped_unsafe = 0
    dropped_empty = 0

    def _save(label: str) -> None:
        if not examples:
            return
        try:
            ds = Dataset.from_list(examples)
            splits = ds.train_test_split(test_size=VALIDATION_RATIO, seed=args.seed)
            dd = DatasetDict({"train": splits["train"], "validation": splits["test"]})
            dd.save_to_disk(args.output_path)
            print(
                f"[intercode_synth] {label} save: {len(examples)} examples -> {args.output_path} "
                f"train={len(dd['train'])} val={len(dd['validation'])}",
                flush=True,
            )
        except Exception as exc:
            print(f"[intercode_synth] {label} save FAILED: {exc}", flush=True)

    import atexit
    atexit.register(_save, "atexit-final")

    for fs_v in _GEN_FS:
        if time.monotonic() - start_ts > walltime_budget:
            print(f"[intercode_synth] walltime {walltime_budget}s exceeded; stopping at fs{fs_v}.", flush=True)
            break
        templates = [t for t in _TEMPLATES if fs_v in t.fs]
        if not templates:
            continue

        try:
            env = LocalBashEnv(fs_v, snapshot_root)
            env.reset()
        except Exception as exc:
            print(f"[intercode_synth] fs{fs_v}: LocalBashEnv unavailable ({exc}); skipping.", flush=True)
            continue

        inv = _walk_inventory(_FS_ROOTS[fs_v]) if fs_v in _FS_ROOTS else Inv(
            root="", files=[], dirs=[], by_ext={}, text_files=[], ext_dirs={}
        )
        if fs_v in _FS_ROOTS and inv is None:
            print(f"[intercode_synth] fs{fs_v}: empty inventory at {_FS_ROOTS[fs_v]}; skipping.", flush=True)
            continue

        seen: set[tuple] = set()
        dirty = False
        produced = 0
        for i in range(args.per_fs):
            if i and i % 500 == 0:
                elapsed = time.monotonic() - start_ts
                print(
                    f"[intercode_synth] fs{fs_v} progress {i}/{args.per_fs} elapsed={elapsed:.0f}s "
                    f"built={built}",
                    flush=True,
                )
            if time.monotonic() - start_ts > walltime_budget:
                break
            t = rng.choice(templates)
            try:
                out = t.fn(inv, rng)
            except Exception:
                out = None
            if out is None:
                continue
            query, gold = out
            key = (query, gold)
            if key in seen:
                continue
            seen.add(key)
            if not is_safe_for_real_exec(gold):
                dropped_unsafe += 1
                continue
            try:
                if t.mutates or dirty:
                    env.reset()
                    dirty = False
                obs = env.step(gold)
                if t.mutates:
                    dirty = True
            except Exception as exc:
                print(f"[intercode_synth] fs{fs_v} exec fail ({t.name}): {exc}", flush=True)
                continue
            if not _meaningful(obs, t.mutates):
                dropped_empty += 1
                continue
            try:
                tool_exs = build_tool_examples_multiturn(query, gold, obs)
            except Exception:
                tool_exs = None
            if not tool_exs:
                continue
            examples.extend({"messages": m} for m in tool_exs)
            built += 1
            produced += 1
            if len(examples) and len(examples) % 2000 < len(tool_exs):
                _save(f"checkpoint@{len(examples)}")
            if args.max_per_fs and produced >= args.max_per_fs:
                break
        print(
            f"[intercode_synth] fs{fs_v}: produced {produced} grounded tasks "
            f"({len(_FS_ROOTS.get(fs_v, '')) and len(inv.files)} source files).",
            flush=True,
        )

    print(
        f"[intercode_synth] DONE built={built} tasks -> {len(examples)} examples "
        f"(dropped_unsafe={dropped_unsafe}, dropped_empty/err={dropped_empty})",
        flush=True,
    )
    if not examples:
        raise RuntimeError(
            "No synthetic intercode examples built. Check that /intercode_fs/fs{1,2}.tar "
            f"exist (snapshot_root={snapshot_root}) and restore to /testbed//system."
        )
    _save("final")


if __name__ == "__main__":
    main()
