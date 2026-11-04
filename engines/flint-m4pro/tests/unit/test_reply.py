"""A reply to a request with tools, read as Flint's SGLang parsers read it, against the cases SGLang's
and vLLM's GLM-4.7 parsers are tested on: the reasoning that opens it, the value types a schema
declares, values kept verbatim, text around calls, malformed and unclosed calls; and a stream split
anywhere giving what the whole text gives."""

import json
import random
import time
from pathlib import Path

import pytest

from jitengine_flint_m4pro.reply import ReplyStream, parse

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "count": {"type": "integer"},
                    "edits": {"type": "array"},
                    "id": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                },
            },
        },
    }
]


def call(name: str, **args: str) -> str:
    pairs = "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args.items())
    return f"<tool_call>{name}{pairs}</tool_call>"


def args(c: dict) -> dict:
    return json.loads(c["function"]["arguments"])


@pytest.mark.parametrize(
    ("arg", "raw", "want"),
    [
        ("path", " a.py\n", " a.py\n"),  # a string keeps its text exactly
        ("path", '"a.py"', '"a.py"'),  # and is not unquoted, as SGLang would
        ("count", "2", 2),
        ("count", "2.0", 2),
        ("count", "null", None),
        ("edits", "[1]", [1]),
        ("edits", "{}", "{}"),  # an object is not an array
        ("id", "7", 7),
        ("undeclared", "true", "true"),  # no declared type: the text
    ],
)
def test_a_value_takes_the_first_type_its_schema_admits(arg, raw, want):
    _, _, (c,) = parse(call("edit", **{arg: raw}), TOOLS)
    assert args(c) == {arg: want}


def test_reasoning_content_and_calls():
    text = "<think>Read it first.</think>\n\nOne edit.\n" + call("edit", path="a.py", count="2")
    reasoning, content, (c,) = parse(text, TOOLS)
    assert reasoning == "Read it first." and content == "One edit."
    assert c["function"]["name"] == "edit" and args(c) == {"path": "a.py", "count": 2}
    # whitespace between the name and the arguments, a tool no request offered, no arguments
    text = "<tool_call>edit\n<arg_key>edits</arg_key>\n<arg_value>[1]</arg_value>\n</tool_call>"
    assert args(parse(text, TOOLS)[2][0]) == {"edits": [1]}
    _, _, (other, ls) = parse(call("other", flag="true") + call("ls"), TOOLS)
    assert args(other) == {"flag": "true"} and ls["function"] == {"name": "ls", "arguments": "{}"}


def test_a_value_may_hold_the_close_tags_of_the_format():
    """A file that quotes the format must not be cut at the quoted tag."""
    for value in (
        'x = "</arg_value>"\nrest',
        "a </arg_value> b",
        'x = "</tool_call>"\nrest',
        "a <arg_key>k</arg_key> y",
    ):
        text = "Writing.\n" + call("edit", path=value, count="2") + "\nDone."
        _, content, (c,) = parse(text, TOOLS)
        assert args(c) == {"path": value, "count": 2}, value
        assert content == "Writing.\nDone."


def test_what_stays_reasoning_or_content():
    # a call inside the reasoning is reasoning; a reply that ends first is all reasoning
    assert parse("<think>maybe " + call("edit"), TOOLS) == ("maybe " + call("edit"), "", [])
    # no <think> at the start: no reasoning
    assert parse("Done. <think>x</think>", TOOLS) == (None, "Done. <think>x</think>", [])
    # text between and after calls is content, less the whitespace before each call
    text = "Two.\n" + call("edit", path="a") + " Between.\n" + call("edit", path="b") + " Done.\n"
    _, content, calls = parse(text, TOOLS)
    assert content == "Two. Between. Done." and [args(c)["path"] for c in calls] == ["a", "b"]
    # a call that names no function is dropped; one never closed stays as Flint wrote it
    _, content, calls = parse("Two." + call("") + call("edit", path="a"), TOOLS)
    assert content == "Two." and len(calls) == 1
    unclosed = "Editing.<tool_call>edit<arg_key>path</arg_key><arg_value>a"
    assert parse(unclosed, TOOLS) == (None, unclosed, [])


def test_a_stream_split_anywhere_gives_the_whole_text_s_reading():
    texts = [
        "<think>Read it first.</think>\n\nTwo edits. \n" + call("edit", path="a") + call("ls"),
        "Two.\n" + call("edit", path="a") + " Between.\n" + call("ls") + " Done.\n",
        "Two." + call("") + call("edit", path="a"),
        "  <think>long thought, cut off before </th",
        "<think>x</think>   ",
        "Plain text ending in a partial marker <tool_ca",
        "<thi",
        "Editing.  <tool_call>edit<arg_key>path</arg_key><arg_value>a",
        call("edit", path="a"),
    ]
    for text in texts:
        reasoning, content, calls = parse(text, TOOLS)
        for step in (1, 2, 3, 7, len(text)):
            s, got = ReplyStream(TOOLS), {}
            for i in range(0, len(text), step):
                for k, v in s.push(text[i : i + step]).items():
                    got[k] = got.get(k, "") + v
            rest, got_calls = s.finish()
            for k, v in rest.items():
                got[k] = got.get(k, "") + v
            assert got.get("reasoning_content") == reasoning, (text, step)
            assert got.get("content", "") == content, (text, step)
            assert [c["function"] for c in got_calls] == [c["function"] for c in calls]


# --- malformed and quoting replies that a regex reading got wrong ---


def _read(text: str, tools: list[dict] = TOOLS) -> tuple[str, list[dict]]:
    _, content, calls = parse(text, tools)
    return content, [args(c) for c in calls]


QUOTED = call("edit", path="x")  # a whole example call, as a test file or a doc would hold
MALFORMED = [
    # first call lacks its last </arg_value>: two calls, not one merged call and a lost one
    (
        "first_call_lacks_last_close_arg_value",
        "<tool_call>edit<arg_key>path</arg_key><arg_value>a</tool_call>" + call("edit", path="b"),
        ("", [{"path": "a"}, {"path": "b"}]),
    ),
    (
        "first_call_lacks_close_tool_call",
        "<tool_call>edit<arg_key>path</arg_key><arg_value>a</arg_value>" + call("edit", path="b"),
        ("", [{"path": "a"}, {"path": "b"}]),
    ),
    (
        "middle_arg_lacks_close_arg_value_and_does_not_swallow_the_next",
        "<tool_call>edit<arg_key>path</arg_key><arg_value>a<arg_key>count</arg_key>"
        "<arg_value>2</arg_value></tool_call>",
        ("", [{"path": "a", "count": 2}]),
    ),
    (
        "arg_without_a_value_is_skipped_and_both_calls_kept",
        "<tool_call>edit<arg_key>path</arg_value></tool_call>" + call("edit", path="b"),
        ("", [{}, {"path": "b"}]),
    ),
    (
        "junk_after_last_close_arg_value_is_not_in_the_value",
        "<tool_call>edit<arg_key>path</arg_key><arg_value>a</arg_value> Done.</tool_call>",
        ("", [{"path": "a"}]),
    ),
    (
        "junk_after_a_close_arg_value_does_not_merge_two_calls",
        "<tool_call>edit<arg_key>path</arg_key><arg_value>a</arg_value> Done.</tool_call>"
        + call("edit", path="b"),
        ("", [{"path": "a"}, {"path": "b"}]),
    ),
    (
        "value_quoting_a_whole_call_is_kept",
        call("edit", path=QUOTED),
        ("", [{"path": QUOTED}]),
    ),
    (
        "value_quoting_a_whole_call_is_kept_before_another_arg",
        call("edit", path=QUOTED, count="2"),
        ("", [{"path": QUOTED, "count": 2}]),
    ),
    (
        "value_quoting_two_whole_calls_is_kept",
        call("edit", path=QUOTED + " and " + QUOTED, count="2"),
        ("", [{"path": QUOTED + " and " + QUOTED, "count": 2}]),
    ),
    (
        "value_quoting_a_whole_pair_is_kept",
        call("edit", path="<arg_key>k</arg_key><arg_value>v</arg_value>", count="2"),
        ("", [{"path": "<arg_key>k</arg_key><arg_value>v</arg_value>", "count": 2}]),
    ),
    (
        "value_holding_only_a_close_tool_call_is_kept",
        call("edit", path="a </tool_call> b", count="2"),
        ("", [{"path": "a </tool_call> b", "count": 2}]),
    ),
    (
        "value_holding_only_a_close_arg_value_is_kept",
        call("edit", path="a </arg_value> b", count="2"),
        ("", [{"path": "a </arg_value> b", "count": 2}]),
    ),
    (
        "prose_mentioning_the_tag_in_a_value_is_kept",
        call("edit", path="use <tool_call> to call", count="2"),
        ("", [{"path": "use <tool_call> to call", "count": 2}]),
    ),
    (
        "call_with_every_close_missing_before_the_next_call_stays_content",
        "<tool_call>edit<arg_key>path</arg_key><arg_value>a" + call("edit", count="2"),
        ("<tool_call>edit<arg_key>path</arg_key><arg_value>a", [{"count": 2}]),
    ),
]


@pytest.mark.parametrize(("name", "text", "want"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_replies_keep_every_call_and_every_value_s_text(name, text, want):
    assert _read(text) == want
    s, got = ReplyStream(TOOLS), {}
    for i in range(0, len(text), 3):
        for k, v in s.push(text[i : i + 3]).items():
            got[k] = got.get(k, "") + v
    rest, calls = s.finish()
    for k, v in rest.items():
        got[k] = got.get(k, "") + v
    assert (got.get("content", ""), [args(c) for c in calls]) == want


def _best_of(n: int, fn) -> float:
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _grows_linearly(build, n: int) -> None:
    """Parsing a reply four times as long takes about four times as long: at most eight here, where
    a quadratic parser takes sixteen. A fixed bound alone fails on a slow CI runner; the full-size
    reply must still parse well under a second."""
    small, large = build(n // 4), build(n)
    parse(small, TOOLS), parse(large, TOOLS)  # warm up
    t_small = _best_of(3, lambda: parse(small, TOOLS))
    t_large = _best_of(3, lambda: parse(large, TOOLS))
    assert t_large < 1.0 and t_large <= 8 * max(t_small, 0.002), (t_small, t_large)


@pytest.mark.parametrize(
    ("head", "unit", "n", "tail"),
    [
        ("<tool_call>f", "<arg_key>k</arg_key><arg_value>x", 400, "</tool_call>"),
        ("<tool_call>f", "<arg_key>k</arg_key><arg_value>x", 4000, "</tool_call>"),
        ("<tool_call>f<arg_key>k</arg_key><arg_value>", "</tool_call>", 16000, ""),
        ("<tool_call>f<arg_key>k</arg_key><arg_value>", "</arg_value> ", 16000, ""),
        ("<tool_call>f<arg_key>k</arg_key><arg_value>", "<tool_call>f", 16000, ""),
        ("<tool_call>f<arg_key>k</arg_key><arg_value>", "<arg_key>k</arg_key>", 8000, ""),
        ("", "<tool_call>f<arg_key>k</arg_key><arg_value>x", 4000, ""),
        ("", "<tool_call>f<arg_key>k</arg_key><arg_value>x</tool_call>", 4000, ""),
        ("", "<tool_call>f<arg_key>k</arg_key><arg_value>x</arg_value> y", 4000, ""),
    ],
    ids=[
        "unclosed_pairs_400",
        "unclosed_pairs_4000",
        "stray_close_tool_calls",
        "stray_close_arg_values",
        "nested_opens",
        "pair_starts",
        "unclosed_calls",
        "calls_closed_by_tool_call_only",
        "calls_with_junk",
    ],
)
def test_parsing_takes_time_linear_in_the_reply(head, unit, n, tail):
    _grows_linearly(lambda k: head + unit * k + tail, n)


@pytest.mark.parametrize(
    ("head", "unit", "n"),
    [
        (
            "<tool_call>edit<arg_key>path</arg_key><arg_value>",
            "<arg_key>count</arg_key><arg_value>x</arg_value>",
            8000,
        ),
        ("<tool_call>edit<arg_key>path</arg_key><arg_value>", call("edit", path="x"), 4000),
        ("<tool_call>edit<arg_key>path</arg_key><arg_value>", "<tool_call>edit<arg_key>", 6000),
        ("<tool_call>edit<arg_key>path</arg_key><arg_value>x", "</arg_value></tool_call>", 6000),
    ],
    ids=["quoted_pairs", "quoted_calls", "lone_opens", "stray_closes"],
)
def test_parsing_a_quoting_value_takes_time_linear_in_the_reply(head, unit, n):
    _grows_linearly(lambda k: head + unit * k, n)


ATOMS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "edit",
    "ls",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "path",
    "hello ",
    " ",
    "\n",
    "2",
    "<",
    "</",
    "<tool",
]


def _stream(pieces: list[str], tools: list[dict]):
    s, got = ReplyStream(tools), {}
    for p in pieces:
        for k, v in s.push(p).items():
            got[k] = got.get(k, "") + v
    rest, calls = s.finish()
    for k, v in rest.items():
        got[k] = got.get(k, "") + v
    return got.get("reasoning_content"), got.get("content", ""), [c["function"] for c in calls]


def test_streamed_and_whole_readings_agree_on_random_replies():
    rng = random.Random(243)
    for _ in range(1500):
        text = "".join(rng.choice(ATOMS) for _ in range(rng.randrange(1, 40)))
        pieces, i = [], 0
        while i < len(text):
            j = i + rng.choice([1, 1, 2, 3, 7])
            pieces.append(text[i:j])
            i = j
        for tools in ([], TOOLS):
            reasoning, content, calls = parse(text, tools)
            want = (reasoning, content, [c["function"] for c in calls])
            got = _stream(pieces, tools)
            assert (got[0] or None, got[1], got[2]) == (reasoning or None, content, want[2]), text


VALUES = [
    "plain",
    "two\nlines",
    "a </tool_call> b",
    'x = "</arg_value>"\nrest',
    "a </arg_value> b",
    "a <arg_key>k</arg_key> y",
    "use <tool_call> to call",
    "<arg_key>k</arg_key><arg_value>v</arg_value>",
    QUOTED,
    "see " + QUOTED + " above",
]


WIDE = [  # the tool of TOOLS with the arguments that the generated calls give
    {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "properties": {k: {"type": "string"} for k in ("path", "other", "label")}
            },
        },
    }
]


def test_well_formed_replies_with_quoting_values_read_back_exactly():
    rng = random.Random(7)
    for _ in range(400):
        sent = [
            {
                k: rng.choice(VALUES)
                for k in rng.sample(["path", "other", "label"], rng.randint(0, 3))
            }
            for _ in range(rng.randint(1, 3))
        ]
        text = "Go.\n" + "".join(call("edit", **a) for a in sent)
        assert _read(text, WIDE) == ("Go.", sent), text


# --- a value that quotes the format, read with and without the request's tool schemas (round 3) ---

BARE = [{"type": "function", "function": {"name": "other"}}]  # a request with no schemas
WRITE = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "properties": {k: {"type": "string"} for k in ("path", "content", "note")}
            },
        },
    }
]
ORDERS = {
    "first": ("content", "path", "note"),
    "middle": ("path", "content", "note"),
    "last": ("path", "note", "content"),
}
OTHERS = {"path": "x.py", "note": "n"}


def _round_trips(content: str, tools: list[dict]) -> bool:
    for order in ORDERS.values():
        want = {k: content if k == "content" else OTHERS[k] for k in order}
        _, rest, calls = parse(call("write", **want), tools)
        if rest != "" or [args(c) for c in calls] != [want]:
            return False
    return True


@pytest.mark.parametrize(
    "content",
    [
        "Models emit a <tool_call> tag to call tools.",
        "x\n<tool_call>\ny",
        "<tool_call>",
        "if '<tool_call>' in text:\n    parse(text)",
        "<tool_call>write",
        "a </tool_call> b <tool_call> c",
    ],
    ids=["prose", "line", "only_open", "code", "open_and_name", "closed_and_open"],
)
def test_a_value_with_a_lone_tool_call_keeps_its_call_with_or_without_schemas(content):
    assert _round_trips(content, WRITE)
    assert _round_trips(content, BARE)


@pytest.mark.parametrize(
    "content",
    [
        "<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value>",
        'S = "<arg_key>a</arg_key><arg_value>1</arg_value>"',
        "see <arg_key>a</arg_key> and <arg_value>2</arg_value>",
    ],
    ids=["two_pairs", "python_string", "separate_tags"],
)
def test_a_value_that_quotes_whole_pairs_is_kept_with_or_without_schemas(content):
    assert _round_trips(content, WRITE)
    assert _round_trips(content, BARE)


def test_a_declared_argument_that_was_not_given_yet_is_what_starts_a_new_one():
    """The schemas break a tie only where the text is ambiguous: a quoted pair that more value
    text follows before the real close."""
    quoted = "x</arg_value><arg_key" + ">zzz</arg_key><arg_value>q</arg_value> and more"  # split
    text = call("write", path="x.py", content=quoted, note="n")
    # no schemas: the quoted close ends the value and zzz is an argument of the call
    _, _, calls = parse(text, BARE)
    assert args(calls[0])["content"] == "x"
    _, _, calls = parse(text, WRITE)
    assert [args(c) for c in calls] == [{"path": "x.py", "content": quoted, "note": "n"}]
    # an argument that the call has given already is text as well
    again = "x</arg_value><arg_key" + ">path</arg_key><arg_value>q"
    _, _, calls = parse(call("write", path="x.py", content=again, note="n"), WRITE)
    assert [args(c) for c in calls] == [{"path": "x.py", "content": again, "note": "n"}]
    # and a declared one that is still to come starts the next argument
    _, _, calls = parse(call("write", content="x", path="x.py", note="n"), WRITE)
    assert [args(c) for c in calls] == [{"content": "x", "path": "x.py", "note": "n"}]


BASH = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "properties": {
                    k: {"type": "string"}
                    for k in ("command", "timeout", "description", "run_in_background")
                }
            },
        },
    }
]


def test_claude_codes_bash_call_with_an_undeclared_workdir_keeps_its_command():
    """Found live: the command came back with the next argument's tags glued to it. Claude Code refuses the `workdir`
    Flint sends (21 of 21 named the project directory), so it is folded into the command."""
    wire = call("Bash", command="python3 test_calc.py", workdir="/Users/someone/calc-demo")
    want = {"command": "cd /Users/someone/calc-demo && python3 test_calc.py"}
    assert _read(wire, BASH)[1] == [want]
    streamed = _stream([wire[i : i + 5] for i in range(0, len(wire), 5)], BASH)[2]
    assert streamed == [{"name": "Bash", "arguments": json.dumps(want)}]


CLAUDE = [  # three of Claude Code's tools, whose names it declares capitalised
    {
        "type": "function",
        "function": {"name": n, "parameters": {"properties": {"x": {"type": "integer"}}}},
    }
    for n in ("Read", "Edit", "Bash")
]


@pytest.mark.parametrize("written", ["read", "READ", "rEaD", "Read"])
def test_a_tool_named_in_another_case_is_the_declared_tool(written):
    """Found live: Flint called `read`, `edit` and `bash` where Claude Code declared `Read`, `Edit` and `Bash`, and
    Claude Code refused each call. The one declared name that matches but for case names the call, and its schema
    types the arguments; streamed the same."""
    wire = call(written, x="7")
    assert _read(wire, CLAUDE)[1] == [{"x": 7}]
    assert [c["function"]["name"] for c in parse(wire, CLAUDE)[2]] == ["Read"]
    assert _stream([wire[i : i + 4] for i in range(0, len(wire), 4)], CLAUDE)[2] == [
        {"name": "Read", "arguments": json.dumps({"x": 7})}
    ]


def test_a_lowercase_call_after_an_unclosed_value_is_still_a_second_call():
    """The check for a call inside an unclosed value (`<tool_call>` there starts a call only when it names a declared
    tool) takes the declared name too, so a lowercase second call is not swallowed into the first one's value."""
    wire = call("Read", x="1").replace("</arg_value>", "") + call(
        "bash", x="2"
    )  # the first lacks its </arg_value>
    _, content, calls = parse(wire, CLAUDE)
    assert content == "" and [(c["function"]["name"], args(c)) for c in calls] == [
        ("Read", {"x": 1}),
        ("Bash", {"x": 2}),
    ]


def test_a_name_two_declared_tools_match_but_for_case_is_kept_as_written():
    tools = CLAUDE + [{"type": "function", "function": {"name": "READ", "parameters": {}}}]
    for written, want in (("Read", "Read"), ("READ", "READ"), ("read", "read"), ("edit", "Edit")):
        (c,) = parse(call(written, x="7"), tools)[2]
        assert c["function"]["name"] == want, written


def _fn(name: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


STR, INT = {"type": "string"}, {"type": "integer"}
READ_EDIT = [  # Claude Code's Read and Edit, their properties trimmed
    _fn("Read", {"file_path": STR, "offset": INT, "limit": INT}, ["file_path"]),
    _fn(
        "Edit",
        {"file_path": STR, "old_string": STR, "new_string": STR},
        ["file_path", "old_string", "new_string"],
    ),
]


def test_an_undeclared_argument_named_for_a_missing_required_one_takes_its_name():
    """Found live: Flint read files with `path` where Claude Code's Read requires `file_path`, 22 of the 52 calls
    Claude Code refused for their arguments. The value keeps its type from the declared name; streamed the same."""
    wire = call("Read", path="/p/calc.py", limit="40")
    want = {"file_path": "/p/calc.py", "limit": 40}
    assert _read(wire, READ_EDIT)[1] == [want]
    assert _stream([wire[i : i + 5] for i in range(0, len(wire), 5)], READ_EDIT)[2] == [
        {"name": "Read", "arguments": json.dumps(want)}
    ]


def test_an_undeclared_argument_that_repeats_a_declared_one_is_dropped():
    """Found live: Flint's Edit calls carried a `path` beside the same `file_path` (10 refused calls)."""
    wire = call("Edit", file_path="/p/calc.py", path="/p/calc.py", old_string="a", new_string="b")
    assert _read(wire, READ_EDIT)[1] == [
        {"file_path": "/p/calc.py", "old_string": "a", "new_string": "b"}
    ]


def test_other_undeclared_arguments_stay_for_the_client_to_judge():
    """A `path` that names another file than `file_path`, and a `workdir`, could mean something: they are kept."""
    wire = call("Edit", file_path="/p/a.py", path="/p/b.py", old_string="a", new_string="b")
    assert _read(wire, READ_EDIT)[1][0]["path"] == "/p/b.py"
    assert _read(call("Read", file_path="/p/a.py", workdir="/p"), READ_EDIT)[1] == [
        {"file_path": "/p/a.py", "workdir": "/p"}
    ]


SHELL = [_fn("Bash", {"command": STR, "timeout": INT}, ["command"])]


def test_an_undeclared_workdir_is_folded_into_the_command():
    """The directory is quoted, so a space in it stays in one word; streamed the same."""
    wire = call("Bash", command="python3 test_calc.py", workdir="/p/calc demo")
    want = {"command": "cd '/p/calc demo' && python3 test_calc.py"}
    assert _read(wire, SHELL)[1] == [want]
    assert _stream([wire[i : i + 5] for i in range(0, len(wire), 5)], SHELL)[2] == [
        {"name": "Bash", "arguments": json.dumps(want)}
    ]


def test_an_undeclared_cwd_is_folded_into_the_command_too():
    wire = call("Bash", command="ls", cwd=" Users/someone/calc ")
    assert _read(wire, SHELL)[1] == [{"command": "cd Users/someone/calc && ls"}]


def test_a_declared_workdir_stays_an_argument():
    tools = [_fn("Run", {"command": STR, "workdir": STR}, ["command"])]
    wire = call("Run", command="ls", workdir="/p")
    assert _read(wire, tools)[1] == [{"command": "ls", "workdir": "/p"}]


def test_an_undeclared_workdir_stays_when_the_tool_has_no_command():
    tools = [_fn("Run", {"script": STR}, ["script"])]
    wire = call("Run", script="a.sh", workdir="/p")
    assert _read(wire, tools)[1] == [{"script": "a.sh", "workdir": "/p"}]


def test_an_empty_workdir_is_left_alone():
    wire = call("Bash", command="ls", workdir="  ")
    assert _read(wire, SHELL)[1] == [{"command": "ls", "workdir": "  "}]


def test_a_workdir_and_a_cwd_together_are_left_alone():
    wire = call("Bash", command="ls", workdir="/p", cwd="/q")
    assert _read(wire, SHELL)[1] == [{"command": "ls", "workdir": "/p", "cwd": "/q"}]


def test_a_rename_needs_exactly_one_missing_required_argument_it_fits():
    tools = [_fn("Copy", {"src_path": STR, "dst_path": STR}, ["src_path", "dst_path"])]
    assert _read(call("Copy", path="/p/a"), tools)[1] == [{"path": "/p/a"}]  # fits both
    assert _read(call("Copy", src_path="/p/a", path="/p/b"), tools)[1] == [
        {"src_path": "/p/a", "dst_path": "/p/b"}
    ]


@pytest.mark.parametrize(
    "pairs",
    [
        (("path", "x.py"), ("workdir", "/tmp")),
        (("workdir", "/tmp"), ("path", "x.py")),
        (("path", "x.py"), ("workdir", "/tmp"), ("note", "n")),
        (("path", "x.py"), ("workdir", "/tmp"), ("env", "A=1")),
        (("workdir", "/tmp"),),
    ],
    ids=["at_the_end", "first", "in_the_middle", "two_undeclared", "only"],
)
def test_a_well_formed_undeclared_argument_is_an_argument_of_the_call(pairs):
    text = "Running." + call("write", **dict(pairs))
    _, content, calls = parse(text, WRITE)
    assert content == "Running." and [args(c) for c in calls] == [dict(pairs)]
    streamed = _stream([text[i : i + 4] for i in range(0, len(text), 4)], WRITE)
    assert streamed[1] == "Running." and [json.loads(c["arguments"]) for c in streamed[2]] == [
        dict(pairs)
    ]


def test_the_stream_with_tools_reads_what_the_whole_reply_reads():
    text = "Go. " + call("write", path="x.py", content="<tool_call>q<arg_key>k</arg_key>", note="n")
    reasoning, content, calls = parse(text, WRITE)
    want = (reasoning, content, [c["function"] for c in calls])
    for step in (1, 4, 9):
        got = _stream([text[i : i + step] for i in range(0, len(text), step)], WRITE)
        assert (got[0] or None, got[1], got[2]) == (reasoning, content, want[2])


def _corpus() -> list[tuple[str, str]]:
    """Every text file of this engine's src/ and tests/, as (relative path, text)."""
    engine = Path(__file__).resolve().parents[2]
    out = []
    for folder in ("src", "tests"):
        for path in sorted((engine / folder).rglob("*")):
            if path.suffix not in {
                ".py",
                ".md",
                ".jinja",
                ".json",
                ".toml",
                ".txt",
                ".yaml",
                ".yml",
            }:
                continue
            if "__pycache__" in path.parts or not path.is_file() or path.stat().st_size > 300_000:
                continue
            try:
                out.append((str(path.relative_to(engine)), path.read_text(encoding="utf-8")))
            except UnicodeDecodeError:
                continue
    return out


@pytest.mark.parametrize("tools", [WRITE, BARE], ids=["with_schemas", "without_schemas"])
def test_every_file_of_the_engine_round_trips_as_an_argument_value(tools):
    """The engine's own sources and tests (and its chat template) quote the format in every way
    (whole calls, lone tags, close tags in strings, unclosed examples): each, as the first,
    middle or last argument and with or without a trailing newline, must come back exactly."""
    corpus = _corpus()
    assert len(corpus) > 50 and any(name.endswith(".jinja") for name, _ in corpus)
    failed = []
    for name, text in corpus:
        for content in (text, text.rstrip("\n")):
            if not _round_trips(content, tools):
                failed.append(name)
                break
    assert not failed, failed


def test_a_name_with_no_argument_and_no_close_before_the_next_call_stays_content():
    text = "<tool_call>edit" + call("edit", path="a")
    _, content, calls = parse(text, TOOLS)
    assert content == "<tool_call>edit" and [args(c) for c in calls] == [{"path": "a"}]
