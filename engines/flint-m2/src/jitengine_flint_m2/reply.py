"""A chat reply to a request with tools, read as Recursive's `flint` parsers for SGLang read it: an
optional `<think>…</think>` that opens the reply, returned as `reasoning_content`; the text after it;
and the tool calls the template asks for,

    <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>

whose VALUE it renders as it is for a string and as JSON otherwise. A `<tool_call>` inside the
reasoning stays reasoning. Text outside the calls is content; a call that names no function is
dropped, and one the reply never closed stays in the content as Flint wrote it, which the template
takes back. A VALUE is converted by the type its parameter's schema declares, as vLLM converts it,
and stays text when the schema declares none or the text does not convert. `parse` reads a whole
reply; `ReplyStream` reads one as it streams, holding back what may still begin a marker and
everything from the first `<tool_call>` on.

Grammar of a call. The text is cut once into tags (`<tool_call>`, `</tool_call>`, `<arg_key>`,
`</arg_key>`, `<arg_value>`, `</arg_value>`) and the text between them, in one pass that cannot
backtrack; everything after works on that list with a bounded lookahead, so the cost is linear in the
reply.

    call = OPEN NAME { ARG_KEY key ARG_KEY_END ARG_VALUE value ARG_VALUE_END } CLOSE

A pair starts at an `<arg_key>` that is followed by `</arg_key>` and `<arg_value>` with only
whitespace between those two. A value may quote the format, as a file that documents it does, so a
tag inside a value ends it only when what follows is structure. From the start of a value we read
tags until one of these decides it:

 1. A `</arg_value>` followed, after whitespace, by `<arg_key>`, `</tool_call>`, `<tool_call>` or the
    end of the reply ends the value. This is the template's own shape.
 2. A `</arg_value>` followed by anything else is a weak end. A later strong end (rule 1) within the
    lookahead wins, so `x = "</arg_value>"` keeps its text; otherwise the weak one ends the value
    and the junk after it is ignored.
 3. A pair start with an `</arg_value>` to close it, or a `<tool_call>` that is followed by a name
    and has a `</tool_call>` to close it (a stack pairs them), is a quotation: it is skipped whole,
    so a value may hold a complete example pair or call. A `<tool_call>` that is no call, as in prose
    about the format, is ignored.
 4. A value with no end of those kinds is left open and ends at the first of: a pair start, a
    `</tool_call>`, or a `<tool_call>`. This is what a model that forgets a close tag produces, and
    it keeps the calls around it: a call missing its last `</arg_value>` is followed by its own
    `</tool_call>` and then by the next call.

Past a candidate end, a reading looks on for a better one, and the whole reply is allowed 8 such
tags per tag it holds, which keeps the work linear. A call missing its `</tool_call>` is kept when the next call starts and its
last value was closed; any other call that is not closed stays in the content. What stays ambiguous,
and is read as the template would write it: a value that quotes only the tail of a pair or call
(`</arg_value>` followed by `<arg_key>` or `</tool_call>`) ends at that quoted `</arg_value>`. A
whole quoted pair or call, open tag included, is kept.
"""

from __future__ import annotations

import json
import math
import re
import uuid

THINK, END_THINK = "<think>", "</think>"
OPEN = "<tool_call>"
CLOSE = "</tool_call>"
_TAG = re.compile(r"<tool_call>|</tool_call>|<arg_key>|</arg_key>|<arg_value>|</arg_value>")
_KINDS = {
    "<tool_call>": 0,
    "</tool_call>": 1,
    "<arg_key>": 2,
    "</arg_key>": 3,
    "<arg_value>": 4,
    "</arg_value>": 5,
}
_OPEN, _CLOSE, _KEY, _KEY_END, _VALUE, _VALUE_END = range(6)
_STRONG_NEXT = (_KEY, _CLOSE, _OPEN)  # what a closing `</arg_value>` is followed by


class _Allowed:
    """The arguments a call may still be given: `open` are the declared ones not given yet, and
    `declared` all the declared ones."""

    __slots__ = ("declared", "open")

    def __init__(self, declared: frozenset[str], given: set[str]):
        self.declared, self.open = declared, declared - given


_NAME = re.compile(r"[\w.:-]{1,128}")  # what a function's name looks like
_WORK = 8  # tags read past a candidate end, per tag of the reply, while looking for a better one
_NULLS = ("null", "none")  # JSON's and Python's spelling
_JSON_TYPES = (
    (bool, "boolean"),  # before int, which bool is
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
)
_MISSING = object()


def _param_schemas(tools: list[dict]) -> dict[str, dict]:
    """Each tool's parameter schemas, by tool name and parameter name."""
    out: dict[str, dict] = {}
    for tool in tools:
        # the template takes a tool's function, or the function itself
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        out[fn["name"]] = props if isinstance(props, dict) else {}
    return out


def _list(value: object) -> list:
    return value if isinstance(value, list) else []


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    return next((name for kind, name in _JSON_TYPES if isinstance(value, kind)), "string")


def _schema_types(schema: object) -> list[str]:
    """The JSON types a parameter's schema admits, in its order: `type` (one or a list), the
    options of `anyOf` and `oneOf`, the types of `enum` and `const` values, and null when it is
    `nullable`. Empty when the schema does not constrain the type."""
    if not isinstance(schema, dict):
        return []
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else _list(declared)
    for key in ("anyOf", "oneOf"):
        for option in _list(schema.get(key)):
            types += _schema_types(option)
    literals = [schema["const"]] if "const" in schema else []
    types += [_json_type(v) for v in [*literals, *_list(schema.get("enum"))]]
    if schema.get("nullable") is True:
        types.append("null")
    return [t for i, t in enumerate(types) if isinstance(t, str) and t not in types[:i]]


def _integer(raw: str) -> object:
    value = _number(raw)
    if isinstance(value, float):  # a whole number written as a float is an integer to JSON Schema
        return int(value) if value.is_integer() else _MISSING
    return value


def _number(raw: str) -> object:
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        value = float(raw)
    except ValueError:
        return _MISSING
    return value if math.isfinite(value) else _MISSING


def _boolean(raw: str) -> object:
    return {"true": True, "1": True, "false": False, "0": False}.get(raw.strip().lower(), _MISSING)


def _json_of(kind: type, empty: object):
    def convert(raw: str) -> object:
        if not raw.strip():
            return empty
        try:
            value = json.loads(raw)
        except ValueError:
            return _MISSING
        return value if isinstance(value, kind) else _MISSING

    return convert


_CONVERT = {
    "string": lambda raw: raw,
    "integer": _integer,
    "number": _number,
    "boolean": _boolean,
    "object": _json_of(dict, {}),
    "array": _json_of(list, []),
    "null": lambda raw: None if raw.strip().lower() in _NULLS else _MISSING,
}


def _value(raw: str, schema: object) -> object:
    """An argument's value: its text converted to the first type its schema admits that the
    text spells, else the text. `null` (or `None`) is null unless the parameter is a string."""
    types = _schema_types(schema)
    if raw.strip().lower() in _NULLS and types != ["string"]:
        return None
    for t in types:
        value = _CONVERT[t](raw) if t in _CONVERT else _MISSING
        if value is not _MISSING:
            return value
    return raw


class _Tags:
    """The tags of a text, with what the grammar asks about each: its kind and position, whether
    only whitespace separates it from the tag before, the index of the first `</arg_value>` at or
    after each tag, and for each `<tool_call>` the index of the `</tool_call>` that closes it (-1
    when none does)."""

    def __init__(self, text: str, schemas: dict[str, dict] | None = None):
        self.text = text
        self.schemas = schemas or {}
        # a name the model wrote in another case than the request declared it ("read" for Claude Code's "Read",
        # which Claude Code refuses) is that tool, when only one declared name folds to it
        folds: dict[str, list[str]] = {}
        for declared in self.schemas:
            folds.setdefault(declared.casefold(), []).append(declared)
        self.folded = {f: names[0] for f, names in folds.items() if len(names) == 1}
        self.declared: dict[str, frozenset[str]] = {}
        kinds, starts, ends, spaced = [], [], [], []
        prev = 0
        for m in _TAG.finditer(text):
            kinds.append(_KINDS[m.group(0)])
            starts.append(m.start())
            ends.append(m.end())
            spaced.append(not text[prev : m.start()].strip())
            prev = m.end()
        self.kinds, self.starts, self.ends, self.spaced = kinds, starts, ends, spaced
        self.tail_spaced = not text[prev:].strip()
        self.count = n = len(kinds)
        self.work = _WORK * (n + 8)  # tags the whole reading may read past a candidate
        self.match = [-1] * n
        stack: list[int] = []
        for k, kind in enumerate(kinds):
            if kind == _OPEN and self._opens_a_call(k):
                stack.append(k)
            elif kind == _CLOSE and stack and self._ends_a_call(k):
                self.match[stack.pop()] = k
        # whether what follows each `</arg_value>` is only well-formed pairs up to the call's end
        self.clean = [False] * n
        for k in range(n - 1, -1, -1):
            if kinds[k] != _VALUE_END:
                continue
            if k + 1 >= n:
                self.clean[k] = self.tail_spaced
            elif self.spaced[k + 1]:
                nxt = kinds[k + 1]
                self.clean[k] = nxt in (_CLOSE, _OPEN) or (
                    self.pair_at(k + 1)
                    and k + 4 < n
                    and kinds[k + 4] == _VALUE_END
                    and self.clean[k + 4]
                )
        self.next_value_end = [-1] * (n + 1)
        for k in range(n - 1, -1, -1):
            self.next_value_end[k] = k if kinds[k] == _VALUE_END else self.next_value_end[k + 1]

    def _ends_a_call(self, k: int) -> bool:
        """Whether the `</tool_call>` at tag `k` follows an `</arg_value>` or a name, as a call's
        does."""
        if k == 0 or not self.spaced[k]:
            return k > 0 and self.kinds[k - 1] == _OPEN and self._is_name(k - 1, k)
        return self.kinds[k - 1] == _VALUE_END

    def _is_name(self, open_tag: int, next_tag: int) -> bool:
        name = self.text[self.ends[open_tag] : self.starts[next_tag]].strip()
        return _NAME.fullmatch(name) is not None

    def tool_name(self, name: str) -> str:
        """`name` as the request declared it: the name itself when declared, else the one declared name it
        matches but for case, else unchanged."""
        return name if name in self.schemas else self.folded.get(name.casefold(), name)

    def _declared(self, k: int) -> bool:
        """Whether the call opened at tag `k` is one the request offered, when it offered any."""
        name = self.text[self.ends[k] : self.starts[k + 1]].strip()
        return not self.schemas or self.tool_name(name) in self.schemas

    def _opens_a_call(self, k: int) -> bool:
        """Whether the `<tool_call>` at tag `k` is followed by a name and then an argument or the
        `</tool_call>`, as a quoted call is; prose that mentions the tag is not."""
        if k + 1 >= self.count or self.kinds[k + 1] not in (_KEY, _CLOSE):
            return False
        return self._is_name(k, k + 1)

    def pair_at(self, k: int) -> bool:
        """Whether tag `k` starts a pair: `<arg_key>`, `</arg_key>`, then `<arg_value>`."""
        return (
            self.kinds[k] == _KEY
            and k + 2 < self.count
            and self.kinds[k + 1] == _KEY_END
            and self.kinds[k + 2] == _VALUE
            and self.spaced[k + 2]
        )

    def key_at(self, k: int, allowed: _Allowed | None) -> bool:
        """Whether tag `k` starts a pair that is an argument of the call: any pair when `allowed` is
        None. Otherwise a declared argument that was not given yet, or an undeclared one whose
        pair is well-formed and followed only by well-formed pairs up to the call's end."""
        if not self.pair_at(k):
            return False
        if allowed is None:
            return True
        name = self.text[self.ends[k] : self.starts[k + 1]].strip()
        if name in allowed.open:
            return True
        return (
            name not in allowed.declared
            and k + 3 < self.count
            and (self.kinds[k + 3] == _VALUE_END and self.clean[k + 3])
        )

    def closes_a_value(self, k: int, allowed: _Allowed | None) -> bool:
        """Whether the `</arg_value>` at tag `k` is followed by the template's next tag."""
        if k + 1 >= self.count:
            return self.tail_spaced
        if not self.spaced[k + 1]:
            return False
        if self.kinds[k + 1] == _KEY and allowed is not None:
            return self.key_at(k + 1, allowed)
        return self.kinds[k + 1] in _STRONG_NEXT

    def value_end(self, first: int, allowed: _Allowed | None) -> tuple[int | None, int]:
        """Where the value whose text starts before tag `first` ends: (the tag index it stops at,
        or None when it runs to the end of the text; the tag index to read on from). With
        `allowed`, the names of the arguments that may still be given, an `<arg_key>` of another
        name is text. A quoted pair or call is skipped on the first reading; when that finds no
        proper end the value is read again with every pair and call as the next one."""
        stop, resume, strong, skipped = self._scan(first, allowed, True)
        if strong or not skipped:
            return stop, resume
        return self._scan(first, allowed, False)[:2]

    def _scan(
        self, first: int, allowed: _Allowed | None, skip_quotes: bool
    ) -> tuple[int | None, int, bool, bool]:
        """One reading of a value: (where it stops, where to read on, whether it ends at a proper
        `</arg_value>`, whether it skipped a quotation)."""
        kinds, match = self.kinds, self.match
        weak = cand = None
        closer, skipped, k = False, False, first
        while k < self.count:
            if weak is not None or cand is not None:
                self.work -= 1
                if self.work < 0:
                    break
            kind = kinds[k]
            if kind == _VALUE_END:
                if self.closes_a_value(k, allowed):
                    return k, k + 1, True, skipped
                weak = k if weak is None else weak
            elif kind == _KEY and self.key_at(k, allowed):
                cand = k if cand is None else cand
                end = self.next_value_end[k + 2]
                if skip_quotes and end >= 0 and self.next_value_end[end + 1] >= 0:
                    skipped = True
                    k = end  # a quoted pair: skip it whole
                else:
                    break
            elif kind == _CLOSE:
                closer = True
                cand = k if cand is None else cand
            elif kind == _OPEN:
                cand = k if cand is None else cand
                if match[k] >= 0:
                    if skip_quotes and self.next_value_end[match[k] + 1] >= 0:
                        skipped = True
                        k = match[k]  # a quoted call: skip it whole
                    elif (closer or weak is not None or not skip_quotes) and self._declared(k):
                        break  # the next call
            k += 1
        if weak is not None:
            return weak, weak + 1, False, skipped
        return (
            (cand, cand, False, skipped) if cand is not None else (None, self.count, False, skipped)
        )

    def call(self, i: int) -> tuple[str, int, int, tuple[str, dict[str, str]]]:
        """Read the call whose `<tool_call>` is tag `i`: (what it is, where its text ends, the tag
        to read on from, its name and raw arguments). What it is: "call", "drop" for a call to
        leave out of the content and the calls, or "text" for a call the reply never closed."""
        text, kinds, starts, ends = self.text, self.kinds, self.starts, self.ends
        n, j = self.count, i + 1
        while j < n and kinds[j] not in (_KEY, _CLOSE, _OPEN):
            j += 1  # a stray tag in the name makes it no name
        name_end = starts[j] if j < n else len(text)
        name = self.tool_name(text[ends[i] : name_end].strip())
        named = bool(name) and "<" not in name
        params: dict[str, str] = {}
        if name not in self.declared:
            self.declared[name] = frozenset(self.schemas.get(name) or ())
        declared = self.declared[name]
        last_closed, k = True, j
        while k < n:
            kind = kinds[k]
            if kind == _KEY and self.pair_at(k):
                key = text[ends[k] : starts[k + 1]].strip()
                allowed = _Allowed(declared, params.keys() | {key}) if declared else None
                stop, resume = self.value_end(k + 3, allowed)
                end = len(text) if stop is None else starts[stop]
                params[key] = text[ends[k + 2] : end]
                last_closed = stop is not None and kinds[stop] == _VALUE_END
                k = resume
                continue
            if kind == _CLOSE:
                return ("call" if named else "drop"), ends[k], k + 1, (name, params)
            if kind == _OPEN:
                done = named and params and last_closed
                return ("call" if done else "text"), starts[k], k, (name, params)
            k += 1
        return "text", len(text), n, (name, params)


def _call(name: str, params: dict[str, str], schemas: dict[str, dict]) -> dict:
    """One call as an OpenAI tool call."""
    declared = schemas.get(name, {})
    args = {key: _value(raw, declared.get(key)) for key, raw in params.items()}
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _calls(text: str, tools: list[dict]) -> tuple[str, list[dict]]:
    """The text's content and its calls: the content is the text outside the closed calls, less
    the whitespace before each and at the end; text without `<tool_call>` is unchanged, and so is
    any text when no tools are offered."""
    if not tools or OPEN not in text:
        return text, []
    schemas = _param_schemas(tools)
    tags = _Tags(text, schemas)
    kept, calls, pos, i = [], [], 0, 0
    while i < tags.count:
        if tags.kinds[i] != _OPEN:
            i += 1
            continue
        what, end, resume, (name, params) = tags.call(i)
        if what == "text":
            i += 1
            continue
        kept.append(text[pos : tags.starts[i]].rstrip())
        if what == "call":
            calls.append(_call(name, params, schemas))
        pos, i = end, resume
    kept.append(text[pos:])
    return "".join(kept).rstrip(), calls


def split_reasoning(text: str) -> tuple[str | None, str]:
    """The reasoning of a reply that opens with `<think>` (up to `</think>`, or all of it when the
    reply ends first) and the rest, less the whitespace that starts it; (None, text) otherwise."""
    head = text.lstrip()
    if not head.startswith(THINK):
        return None, text
    body = head[len(THINK) :]
    end = body.find(END_THINK)
    if end < 0:
        return body, ""
    return body[:end], body[end + len(END_THINK) :].lstrip()


def parse(text: str, tools: list[dict]) -> tuple[str | None, str, list[dict]]:
    """A whole reply's reasoning (None when it does not reason), content and tool calls. With an
    empty `tools` the reply is read for its reasoning alone."""
    reasoning, rest = split_reasoning(text)
    content, calls = _calls(rest, tools)
    return reasoning, content, calls


def _partial(text: str, marker: str) -> int:
    """The length of the longest tail of `text` that begins `marker`."""
    return next((n for n in range(len(marker) - 1, 0, -1) if text.endswith(marker[:n])), 0)


class ReplyStream:
    """A streamed reply to a request with tools. `push` takes the next text and returns the delta
    that is safe to send, `reasoning_content` and `content` as the reply has them; `finish` returns
    the rest and the tool calls. Together they give what `parse` gives for the whole text."""

    def __init__(self, tools: list[dict]):
        self.tools = tools
        self.state = "start"  # start, reasoning, lead (right after </think>), content or calls
        self.pending = ""  # text held back
        self.gap = ""  # the whitespace before the first `<tool_call>`
        self.held = ""  # the text from the first `<tool_call>` on

    def push(self, piece: str) -> dict:
        text, self.pending = self.pending + piece, ""
        delta: dict = {}
        if self.state == "calls":
            self.held += text
            return delta
        if self.state == "start":
            head = text.lstrip()
            if len(head) < len(THINK) and THINK.startswith(head):
                self.pending = text  # it may still open with <think>
                return delta
            self.state = "reasoning" if head.startswith(THINK) else "content"
            text = head[len(THINK) :] if self.state == "reasoning" else text
        if self.state == "reasoning":
            end = text.find(END_THINK)
            cut = len(text) - _partial(text, END_THINK) if end < 0 else end
            if text[:cut]:
                delta["reasoning_content"] = text[:cut]
            if end < 0:
                self.pending = text[cut:]
                return delta
            self.state, text = "lead", text[end + len(END_THINK) :]
        if self.state == "lead":
            text = text.lstrip()
            if not text:
                return delta
            self.state = "content"
        start = text.find(OPEN) if self.tools else -1  # no tools offered: no calls to hold back
        keep = (_partial(text, OPEN) if self.tools else 0) if start < 0 else len(text) - start
        safe = len(text[: len(text) - keep].rstrip())
        if start >= 0:
            self.state, self.gap, self.held = "calls", text[safe:start], text[start:]
        else:
            self.pending = text[safe:]
        if safe:
            delta["content"] = text[:safe]
        return delta

    def finish(self) -> tuple[dict, list[dict]]:
        """The delta still held back at the end of the reply, and its tool calls."""
        pending, self.pending = self.pending, ""
        if self.state == "calls":
            content, calls = _calls(self.gap + self.held, self.tools)
            return ({"content": content} if content else {}), calls
        if not pending:
            return {}, []
        return {"reasoning_content" if self.state == "reasoning" else "content": pending}, []
