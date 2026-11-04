"""Tool calls read from a reply in the template's format, against the cases vLLM's Qwen3-Coder parser
is tested on: the value types a schema declares, values kept verbatim, text around calls, parameters
left open, malformed and unclosed calls; and a stream split anywhere giving what the whole text gives."""

import json
import random
import time
from pathlib import Path

import pytest

from jitengine_qwen3_6_35b_a3b_m4pro.server.tool_calls import ToolCallText, parse

PROPERTIES = {
    "text": {"type": "string"},
    "count": {"type": "integer"},
    "size": {"type": "number"},
    "enabled": {"type": "boolean"},
    "payload": {"type": "object"},
    "items": {"type": "array"},
    "maybe": {"type": "string", "nullable": True},
    "id": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
    "label": {"type": ["string", "integer"]},
    "level": {"enum": [1, 2, 3]},
    "free": {},
}
TOOLS = [
    {"type": "function", "function": {"name": "f", "parameters": {"properties": PROPERTIES}}},
    {"type": "function", "function": {"name": "g", "parameters": {"properties": {}}}},
]


def call(name: str, **params: str) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def args(c: dict) -> dict:
    return json.loads(c["function"]["arguments"])


def one(text: str) -> dict:
    _, calls = parse(text, TOOLS)
    assert len(calls) == 1, calls
    return args(calls[0])


@pytest.mark.parametrize(
    ("param", "raw", "want"),
    [
        ("text", "42", "42"),
        ("text", "null", "null"),  # a string parameter keeps the text "null"
        ("count", "3", 3),
        ("count", "5.0", 5),  # a whole number is an integer to JSON Schema
        ("count", "1.5", "1.5"),  # not one: the text
        ("size", "5.0", 5.0),
        ("size", "nan", "nan"),
        ("enabled", "True", True),
        ("enabled", "0", False),
        ("payload", '{"nested": true}', {"nested": True}),
        ("payload", "[1]", "[1]"),  # an array is not an object
        ("payload", "", {}),
        ("items", "[1, 2]", [1, 2]),
        ("count", "null", None),
        ("count", "None", None),
        ("maybe", "null", None),
        ("id", "42", 42),  # the first type of anyOf that the text spells
        ("id", "abc", "abc"),
        ("label", "42", "42"),  # string comes first here
        ("level", "2", 2),
        ("free", "42", "42"),  # no declared type: the text
        ("undeclared", "[1]", "[1]"),
    ],
)
def test_a_value_takes_the_first_type_its_schema_admits(param, raw, want):
    assert one(call("f", **{param: raw})) == {param: want}


def test_values_are_kept_verbatim_inside_the_template_s_line_breaks():
    code = "    def foo():\n        return 1\n"  # vLLM's whitespace case: one wrapping newline each side
    html = '<div class="test"><span>Hello</span></div>'
    escaped = "杭州 &lt;/parameter&gt;&lt;/function&gt;&lt;/tool_call&gt;"
    assert one(call("f", text=code)) == {"text": code}
    assert one(call("f", text=html, maybe=escaped)) == {"text": html, "maybe": escaped}
    # a tool no request offered, and a call without arguments
    assert one(call("other", flag="true")) == {"flag": "true"}
    assert one(call("g")) == {}


def test_a_value_may_hold_the_close_tags_of_the_format():
    """A file that quotes the format (this module's own docstring) must not be cut at the quoted tag."""
    quoted = "s = '<parameter=x>1</parameter>'\nprint(s)"
    assert one(call("f", text=quoted)) == {"text": quoted}
    for value in (
        "a </parameter> b\nmore lines",
        "x = '<function=foo>'\n</function>\nrest",
        "line1\n</function>",
        "a </tool_call> b",
        "<tool_call>\n<function=g>\n</function>\n</tool_call> tail",
    ):
        assert one(call("f", text=value, count="2")) == {"text": value, "count": 2}, value
    content, calls = parse("Writing.\n" + call("f", text="a </tool_call> b") + "\nDone.", TOOLS)
    assert content == "Writing.\nDone." and len(calls) == 1


def test_the_content_is_the_text_outside_the_calls():
    assert parse("no calls here ", TOOLS) == ("no calls here ", [])
    content, calls = parse("Reading.\n\n" + call("f", text="a") + "\n" + call("g"), TOOLS)
    assert content == "Reading." and [c["function"]["name"] for c in calls] == ["f", "g"]
    assert len({c["id"] for c in calls}) == 2
    # vLLM's text between calls: kept, less the whitespace before each call
    text = "I'll check two.\n" + call("g") + "Between calls.\n" + call("g") + "Done.\n"
    assert parse(text, TOOLS)[0] == "I'll check two.Between calls.Done."


def test_a_parameter_left_open_ends_at_the_next_one_or_the_function():
    text = "<tool_call>\n<function=f>\n<parameter=text>\nDallas\n<parameter=count>\n3\n</function>\n</tool_call>"
    assert one(text) == {"text": "Dallas", "count": 3}
    # and a function left open ends with the call
    assert one("<tool_call>\n<function=f>\n<parameter=count>\n3\n</parameter>\n</tool_call>") == {
        "count": 3
    }


def test_malformed_and_unclosed_calls_are_dropped():
    nameless = "<tool_call>\n<bad>\n</tool_call>"
    no_gt = "<tool_call>\n<function=f\n<parameter=text>x</parameter>\n</function>\n</tool_call>"
    content, calls = parse("Two.\n" + nameless + no_gt + call("f", count="1"), TOOLS)
    assert content == "Two." and [args(c) for c in calls] == [{"count": 1}]
    # a call the reply never closed: its text is not content, which the template would refuse
    assert (
        parse("Reading.\n" + call("g") + "\n<tool_call>\n<function=f>\n<para", TOOLS)[0]
        == "Reading."
    )
    assert parse("Reading.\n<tool_call>\n<function=f>", TOOLS) == ("Reading.", [])


def test_a_stream_split_anywhere_gives_the_whole_text_s_reading():
    texts = [
        "Reading.\n\n" + call("f", count="2", text=" x \n") + "\n" + call("g"),
        "I'll check two.\n" + call("g") + "Between.\n" + call("g") + "Done.",
        "Plain text ending in a space and a partial marker <tool ",
        "Reading.\n<tool_call>\n<function=f>\n<parameter=text>\na",
        "Two.\n<tool_call>\n<bad>\n</tool_call>" + call("f", count="1"),
        "x <tool_call y",
    ]
    for text in texts:
        want_content, want_calls = parse(text, TOOLS)
        for step in (1, 2, 5, 11, len(text)):
            s = ToolCallText(TOOLS)
            sent = "".join(s.push(text[i : i + step]) for i in range(0, len(text), step))
            rest, calls = s.finish()
            assert sent + rest == want_content, (text, step)
            assert [c["function"] for c in calls] == [c["function"] for c in want_calls]


def test_a_stream_holds_back_only_what_may_begin_a_call():
    s = ToolCallText(TOOLS)
    assert s.push("Reading it. ") == "Reading it."
    assert s.push("<tool") == ""
    assert s.push("s are next") == " <tools are next"
    assert s.finish() == ("", [])


# --- malformed and quoting replies that a regex reading got wrong ---


def _wrap(body: str) -> str:
    return f"<tool_call>\n<function=f>\n{body}</function>\n</tool_call>"


def _read(text: str) -> tuple[str, list[dict]]:
    content, calls = parse(text, TOOLS)
    return content, [args(c) for c in calls]


QUOTED = call("g", text="x")  # a whole example call, as a test file or a doc would hold
MALFORMED = [
    # first call lacks its last </parameter>: two calls, not one merged call and a lost one
    (
        "first_call_lacks_last_close_parameter",
        "Go\n<tool_call>\n<function=f>\n<parameter=text>\na.py\n</function>\n</tool_call>\n"
        + call("f", text="b.py"),
        ("Go", [{"text": "a.py"}, {"text": "b.py"}]),
    ),
    (
        "first_call_lacks_close_function",
        "<tool_call>\n<function=f>\n<parameter=text>\na.py\n</parameter>\n</tool_call>\n"
        + call("f", text="b.py"),
        ("", [{"text": "a.py"}, {"text": "b.py"}]),
    ),
    (
        "first_call_lacks_close_tool_call",
        "<tool_call>\n<function=f>\n<parameter=text>\na.py\n</parameter>\n</function>\n"
        + call("f", text="b.py"),
        ("", [{"text": "a.py"}, {"text": "b.py"}]),
    ),
    (
        "middle_parameter_lacks_close_parameter_and_does_not_swallow_the_next",
        _wrap("<parameter=text>\na.py\n<parameter=free>\nhello\n</parameter>\n"),
        ("", [{"text": "a.py", "free": "hello"}]),
    ),
    (
        "junk_after_last_close_parameter_is_not_in_the_value",
        _wrap("<parameter=text>\na.py\n</parameter>\nDone.\n"),
        ("", [{"text": "a.py"}]),
    ),
    (
        "junk_after_a_close_parameter_does_not_merge_two_calls",
        _wrap("<parameter=text>\na.py\n</parameter>\nDone.\n") + "\n" + call("f", text="b.py"),
        ("", [{"text": "a.py"}, {"text": "b.py"}]),
    ),
    (
        "value_quoting_a_whole_call_is_kept_last_parameter",
        call("f", count="1", text=QUOTED),
        ("", [{"count": 1, "text": QUOTED}]),
    ),
    (
        "value_quoting_a_whole_call_is_kept_first_parameter",
        call("f", text=QUOTED, count="1"),
        ("", [{"text": QUOTED, "count": 1}]),
    ),
    (
        "value_quoting_two_whole_calls_is_kept",
        call("f", text=QUOTED + "\nand\n" + QUOTED, count="1"),
        ("", [{"text": QUOTED + "\nand\n" + QUOTED, "count": 1}]),
    ),
    (
        "value_holding_function_and_tool_call_close_tags_is_kept",
        call("f", count="1", text="</function>\n</tool_call> tail"),
        ("", [{"count": 1, "text": "</function>\n</tool_call> tail"}]),
    ),
    (
        "value_holding_only_a_close_parameter_is_kept",
        call("f", count="1", text="x = '</parameter>' + y"),
        ("", [{"count": 1, "text": "x = '</parameter>' + y"}]),
    ),
    (
        "prose_mentioning_the_tag_is_content_and_the_call_after_it_is_kept",
        "Use <tool_call> to call.\n" + call("f", count="1"),
        ("Use <tool_call> to call.", [{"count": 1}]),
    ),
    (
        "call_with_every_close_missing_before_the_next_call_is_dropped_and_the_next_kept",
        "<tool_call>\n<function=f>\n<parameter=text>\nx\n" + call("f", count="2"),
        ("", [{"count": 2}]),
    ),
    (
        "inline_close_tags_after_an_open_value",
        "<tool_call>\n<function=f>\n<parameter=text>\nx</function></tool_call>",
        ("", [{"text": "x"}]),
    ),
]


@pytest.mark.parametrize(("name", "text", "want"), MALFORMED, ids=[m[0] for m in MALFORMED])
def test_malformed_replies_keep_every_call_and_every_value_s_text(name, text, want):
    assert _read(text) == want
    s = ToolCallText(TOOLS)
    sent = "".join(s.push(text[i : i + 3]) for i in range(0, len(text), 3))
    rest, calls = s.finish()
    assert (sent + rest, [args(c) for c in calls]) == want


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
        ("<tool_call>\n<function=f>\n<parameter=text>\n", "</tool_call>", 16000, ""),
        ("<tool_call>\n<function=f>\n<parameter=text>\n", "\n</tool_call>", 16000, ""),
        ("<tool_call>\n<function=f>\n<parameter=text>\n", "</parameter> ", 16000, ""),
        ("<tool_call>\n<function=f>\n<parameter=text>\n", "<tool_call>\n", 16000, ""),
        ("", "<function=", 16000, "<tool_call>"),
        ("", "<tool_call>\n<function=f>\n<parameter=a>\nx\n", 4000, ""),
        ("", "<tool_call><function=f><parameter=a>x", 4000, ""),
        ("", "<tool_call>\n<function=f>\n<parameter=a>\nx\n</tool_call>\n", 4000, ""),
        ("", "<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\ny\n", 4000, ""),
        ("<tool_call>\n<function=f>\n", "<parameter=a>\nx\n", 8000, ""),
    ],
    ids=[
        "stray_close_tool_calls",
        "stray_close_tool_calls_on_lines",
        "stray_close_parameters",
        "nested_opens",
        "function_opens",
        "unclosed_calls",
        "unclosed_calls_on_one_line",
        "calls_closed_by_tool_call_only",
        "calls_with_junk",
        "unclosed_parameters",
    ],
)
def test_parsing_takes_time_linear_in_the_reply(head, unit, n, tail):
    _grows_linearly(lambda k: head + unit * k + tail, n)


@pytest.mark.parametrize(
    ("head", "unit", "n"),
    [
        (
            "<tool_call>\n<function=f>\n<parameter=text>\n",
            "<parameter=count>\nx\n</parameter>\n",
            8000,
        ),
        ("<tool_call>\n<function=f>\n<parameter=text>\n", call("g", text="x"), 4000),
        ("<tool_call>\n<function=f>\n<parameter=text>\n", "<tool_call>\n<function=f>\n", 6000),
        ("<tool_call>\n<function=f>\n<parameter=text>\nx\n", "</parameter>\n</function>\n", 6000),
    ],
    ids=["quoted_parameters", "quoted_calls", "lone_function_opens", "stray_closes"],
)
def test_parsing_a_quoting_value_takes_time_linear_in_the_reply(head, unit, n):
    _grows_linearly(lambda k: head + unit * k, n)


ATOMS = [
    "<tool_call>",
    "</tool_call>",
    "<function=f>",
    "</function>",
    "<parameter=text>",
    "<parameter=count>",
    "</parameter>",
    "\n",
    "\n",
    "hi ",
    " ",
    "2",
    "<",
    "</tool",
    "<param",
]


def _whole_and_streamed(text: str, tools: list[dict], rng: random.Random):
    pieces, i = [], 0
    while i < len(text):
        j = i + rng.choice([1, 1, 2, 3, 7])
        pieces.append(text[i:j])
        i = j
    s = ToolCallText(tools)
    sent = "".join(s.push(p) for p in pieces)
    rest, calls = s.finish()
    content, whole = parse(text, tools)
    return (content, [c["function"] for c in whole]), (sent + rest, [c["function"] for c in calls])


def test_streamed_and_whole_readings_agree_on_random_replies():
    rng = random.Random(243)
    for _ in range(1500):
        text = "".join(rng.choice(ATOMS) for _ in range(rng.randrange(1, 40)))
        for tools in ([], TOOLS):
            whole, streamed = _whole_and_streamed(text, tools, rng)
            assert whole == streamed, (text, bool(tools))


VALUES = [
    "plain",
    "two\nlines",
    "a </tool_call> b",
    "x = '</parameter>' + y",
    "</function>\nrest",
    "line\n</function>",
    "s = '<parameter=x>1</parameter>'\nprint(s)",
    "a </parameter> b\nmore lines",
    QUOTED,
    "see " + QUOTED + " above",
]


def test_well_formed_replies_with_quoting_values_read_back_exactly():
    rng = random.Random(7)
    for _ in range(400):
        sent = [
            {
                k: rng.choice(VALUES)
                for k in rng.sample(["text", "free", "label"], rng.randint(0, 3))
            }
            for _ in range(rng.randint(1, 3))
        ]
        text = "Go.\n" + "\n".join(call("f", **a) for a in sent)
        assert _read(text) == ("Go.", sent), text


# --- a value that quotes the format, read with and without the request's tool schemas ---

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


def _write_call(order: tuple[str, ...], content: str) -> str:
    return call("write", **{k: content if k == "content" else OTHERS[k] for k in order})


def _round_trips(content: str, tools: list[dict]) -> bool:
    for order in ORDERS.values():
        reply, calls = parse(_write_call(order, content), tools)
        want = {k: content if k == "content" else OTHERS[k] for k in order}
        if reply != "" or [args(c) for c in calls] != [want]:
            return False
    return True


@pytest.mark.parametrize(
    "content",
    [
        "Start a call with:\n<tool_call>\nthen name the function.",
        "x\n<tool_call>\n<function=f>\ny",
        "<tool_call>",
        "before\n<function=f>\nafter",
        "a\n</function>\nb\n<tool_call>\nc",
    ],
    ids=["lone_open", "open_and_function", "only_open", "function_open", "mixed"],
)
def test_a_value_with_a_lone_tool_call_line_keeps_its_call_with_or_without_schemas(content):
    """An unmatched <tool_call> in a value is text; the value's own close later in the call wins
    (it made the whole call vanish, with finish_reason stop)."""
    assert _round_trips(content, WRITE)
    if "parameter" + "=b" not in content:  # without schemas that is the call's next parameter
        assert _round_trips(content, [])


QUOTED_PARAMETERS = 'S = (\n    "<parameter=a>\\n1\\n</parameter>\\n"\n    "<parameter=b>\\n2\\n</parameter>\\n"\n)\n'


@pytest.mark.parametrize(
    "content",
    [
        QUOTED_PARAMETERS,
        "<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n",
        "<parameter=a>\n1\n</parameter>\n",
        "see <parameter=a>1</parameter> and <parameter=b>2</parameter>",
    ],
    ids=["python_strings", "lines", "one_pair", "inline_pairs"],
)
def test_a_value_that_quotes_whole_parameters_is_kept_with_or_without_schemas(content):
    assert _round_trips(content, WRITE)
    assert _round_trips(content, [])


def test_a_declared_parameter_that_was_not_given_yet_is_what_starts_a_new_one():
    """The schemas break a tie only where the text is ambiguous: a quoted parameter that more value
    text follows before the real close."""
    quoted = (
        "x\n</parameter>\n<parameter" + "=zzz>\nq\n</parameter>\nand more"
    )  # split: no tag here
    text = _write_call(("path", "content", "note"), quoted)
    _, calls = parse(text, [])  # no schemas: the quoted close ends the value
    assert args(calls[0])["content"] == "x"
    _, calls = parse(text, WRITE)
    assert [args(c) for c in calls] == [{"path": "x.py", "content": quoted, "note": "n"}]
    # a parameter that the call has given already is text as well
    again = "x\n</parameter>\n<parameter" + "=path>\nq"
    _, calls = parse(_write_call(("path", "content", "note"), again), WRITE)
    assert [args(c) for c in calls] == [{"path": "x.py", "content": again, "note": "n"}]
    # and a declared one that is still to come starts the next parameter
    _, calls = parse(_write_call(("content", "path", "note"), "x"), WRITE)
    assert [args(c) for c in calls] == [{"content": "x", "path": "x.py", "note": "n"}]


def _wire(*pairs: tuple[str, str]) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in pairs)
    return f"Running.\n<tool_call>\n<function=write>\n{body}</function>\n</tool_call>"


@pytest.mark.parametrize(
    "pairs",
    [
        (("path", "x.py"), ("workdir", "/Users/someone/demo")),
        (("workdir", "/tmp"), ("path", "x.py")),
        (("path", "x.py"), ("workdir", "/tmp"), ("note", "n")),
        (("path", "x.py"), ("workdir", "/tmp"), ("env", "A=1")),
        (("workdir", "/tmp"),),
    ],
    ids=["at_the_end", "first", "in_the_middle", "two_undeclared", "only"],
)
def test_a_well_formed_undeclared_parameter_is_an_argument_of_the_call(pairs):
    """A request's schema does not hide a parameter that the model gave properly: the previous
    value stays intact (a shell command is not the command plus the next parameter's tags)."""
    text = _wire(*pairs)
    content, calls = parse(text, WRITE)
    assert content == "Running." and [args(c) for c in calls] == [dict(pairs)]
    s = ToolCallText(WRITE)
    sent = "".join(s.push(text[i : i + 4]) for i in range(0, len(text), 4))
    rest, streamed = s.finish()
    assert (sent + rest, [args(c) for c in streamed]) == ("Running.", [dict(pairs)])


def test_the_parser_streamed_with_tools_reads_what_it_reads_whole():
    text = "Go.\n" + _write_call(("path", "content", "note"), QUOTED_PARAMETERS + "<tool_call>\nx")
    want = parse(text, WRITE)
    s = ToolCallText(WRITE)
    sent = "".join(s.push(text[i : i + 5]) for i in range(0, len(text), 5))
    rest, calls = s.finish()
    assert (sent + rest, [args(c) for c in calls]) == (want[0], [args(c) for c in want[1]])


def _corpus() -> list[tuple[str, str]]:
    """Every text file of this engine's src/ and tests/, as (relative path, text)."""
    engine = Path(__file__).resolve().parents[3]
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


@pytest.mark.parametrize("tools", [WRITE, []], ids=["with_schemas", "without_schemas"])
def test_every_file_of_the_engine_round_trips_as_a_parameter_value(tools):
    """The engine's own sources and tests quote the format in every way (whole calls, lone tags,
    close tags in strings, unclosed examples): each, as the first, middle or last parameter and
    with or without a trailing newline, must come back exactly."""
    corpus = _corpus()
    assert len(corpus) > 50
    failed = []
    for name, text in corpus:
        for content in (text, text.rstrip("\n")):
            if not _round_trips(content, tools):
                failed.append(name)
                break
    assert not failed, failed


def test_a_function_prefix_with_no_closed_parameter_and_no_close_function_is_dropped():
    """A reply cut after `<function=NAME>` is not a parameterless call, even when a call follows."""
    prefix = "<tool_call>\n<function=f>\n"
    assert _read(prefix + call("g", text="ls")) == ("", [{"text": "ls"}])
    assert _read("Go.\n" + prefix) == ("Go.", [])
    # parameterless calls that were closed stay calls, with or without the final </tool_call>
    bare = "<tool_call>\n<function=g>\n</function>"
    assert _read(bare)[1] == [{}]
    assert _read(bare + "\n</tool_call>")[1] == [{}]
    assert _read(bare + "\n</tool_call>\n" + call("g", text="ls"))[1] == [{}, {"text": "ls"}]
    assert _read(bare + "\n" + call("g", text="ls"))[1] == [{}, {"text": "ls"}]
    # and a call whose parameter closed but whose </function> is missing is a call
    assert _read(prefix + "<parameter=text>\na\n</parameter>\n" + call("g"))[1] == [
        {"text": "a"},
        {},
    ]
