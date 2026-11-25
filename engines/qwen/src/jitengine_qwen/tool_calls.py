"""Tool calls in a chat reply. The checkpoint's template asks for each call as

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

after any text, and renders a call in history the same way: a string VALUE as it is, any other as
JSON. `parse` reads a reply into its content and OpenAI's `tool_calls`; `ToolCallText` does the same
for a stream, holding back the text from the first `<tool_call>` on.

The reading follows vLLM's Qwen3-Coder parser: a VALUE is converted by the type its parameter's
schema declares, and stays text when the schema declares none or the text does not convert; text
between and after calls is content; a call that is malformed or unclosed (the reply ran out of
tokens or hit a stop sequence) is dropped rather than left in the content.

Grammar. The reply is cut once into tags (`<tool_call>`, `</tool_call>`, `<function=NAME>`,
`</function>`, `<parameter=KEY>`, `</parameter>`) and the text between them, in one pass that cannot
backtrack; everything after works on that list with a bounded lookahead, so the cost is linear in the
reply. A tag is "anchored" when only spaces and tabs separate it from the start of its line or from
the tag before it, which is where the template puts every tag.

    call   = OPEN  FUNC  { PARAM value }  [ FUNC_END ]  CLOSE
    value  = text up to the PARAM_END that ends it

A value may quote the format, as a file that documents it does, so a tag inside a value ends it only
when what follows is structure. From the start of a value we read tags until one of these decides it:

 1. A `</parameter>` followed, after whitespace, by `<parameter=`, `</function>`, `</tool_call>`,
    `<tool_call>` or the end of the reply ends the value. This is the template's own shape.
 2. A `</parameter>` followed by anything else is a weak end. A later strong end (rule 1) within the
    lookahead wins, so `x = '</parameter>' + y` keeps its text; otherwise the weak one ends the value
    and the junk after it is ignored.
 3. A `<tool_call>` that opens a function and has an anchored `</tool_call>` to match (a stack pairs
    them) is a quoted call: it is skipped whole, so a value may hold a complete example call.
    A `<tool_call>` that opens no function, as in prose about the format, is ignored.
 4. A value with no end of those kinds is left open. It ends at the first of: an anchored
    `<parameter=` (any `<parameter=` once a weak end has been seen), a `</function>`, a
    `</tool_call>`, or an anchored `<tool_call>` that is not a quoted call. An unanchored
    `<parameter=` counts only when nothing else does. This is what a model that forgets a close tag
    produces, and it keeps the calls around it: a call missing its last `</parameter>` or its
    `</function>` is followed by its own `</tool_call>` and then by the next call.

Past a candidate end, a reading looks on for a better one, and the whole reply is allowed 8 such
tags per tag it holds, which keeps the work linear. A call missing its `</tool_call>` is kept when the next call starts and everything
before that was closed; any other call that the reply does not close is dropped.

What stays ambiguous is read as the template would write it. A value that quotes only the tail of a
call (`</parameter>` then `</function>`, or `</parameter>` then `<parameter=`) ends at that quoted
`</parameter>`. A whole quoted call, open tag included, is kept.
"""

from __future__ import annotations

import json
import math
import re
import uuid

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
# The tags, found in one pass. The names are bounded so that a run of `<function=` cannot make
# the scan quadratic.
_TAG = re.compile(
    r"<tool_call>|</tool_call>|<function=([^<>\n]{1,256})>|</function>"
    r"|<parameter=([^<>\n]{1,256})>|</parameter>"
)
_OPEN, _CLOSE, _FUNC, _FUNC_END, _PARAM, _PARAM_END = range(6)
_FIXED = {
    "<tool_call>": _OPEN,
    "</tool_call>": _CLOSE,
    "</function>": _FUNC_END,
    "</parameter>": _PARAM_END,
}
_STRONG_NEXT = (_PARAM, _FUNC_END, _CLOSE, _OPEN)  # what a closing `</parameter>` is followed by


class _Allowed:
    """The parameters a call may still be given: `open` are the declared ones not given yet, and
    `declared` all the declared ones."""

    __slots__ = ("declared", "open")

    def __init__(self, declared: frozenset[str], given: set[str]):
        self.declared, self.open = declared, declared - given


_NAME = re.compile(r"[\w.:/-]{1,128}")  # what a function's name looks like
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
        fn = tool.get("function") if isinstance(tool, dict) else None
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
    """A parameter's value: its text converted to the first type its schema admits that the
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
    """The tags of a text, with what the grammar asks about each: its kind and position, the
    parameter or function name it carries, whether it is anchored, whether only whitespace
    separates it from the tag before, and for an anchored `<tool_call>` the index of the
    `</tool_call>` that closes it (-1 when none does)."""

    def __init__(self, text: str, schemas: dict[str, dict] | None = None):
        self.text = text
        self.schemas = schemas or {}
        self.declared: dict[str, frozenset[str]] = {}
        kinds, starts, ends, names, anchored, spaced = [], [], [], [], [], []
        prev = 0
        for m in _TAG.finditer(text):
            if m.group(1) is not None:
                kind, name = _FUNC, m.group(1)
            elif m.group(2) is not None:
                kind, name = _PARAM, m.group(2)
            else:
                kind, name = _FIXED[m.group(0)], ""
            gap_start = text.rfind("\n", prev, m.start()) + 1 or prev
            kinds.append(kind)
            starts.append(m.start())
            ends.append(m.end())
            names.append(name)
            anchored.append(not text[gap_start : m.start()].strip(" \t"))
            spaced.append(not text[prev : m.start()].strip())
            prev = m.end()
        self.kinds, self.starts, self.ends, self.names = kinds, starts, ends, names
        self.anchored, self.spaced = anchored, spaced
        self.tail_spaced = not text[prev:].strip()
        self.count = len(kinds)
        self.work = _WORK * (self.count + 8)  # tags the whole reading may read past a candidate
        # whether what follows each `</parameter>` is only well-formed parameters up to the call's end
        n = len(kinds)
        self.clean = [False] * n
        for k in range(n - 1, -1, -1):
            if kinds[k] != _PARAM_END:
                continue
            if k + 1 >= n:
                self.clean[k] = self.tail_spaced
            elif spaced[k + 1]:
                nxt = kinds[k + 1]
                if nxt == _FUNC_END:
                    self.clean[k] = (
                        self.tail_spaced
                        if k + 2 >= n
                        else spaced[k + 2] and kinds[k + 2] in (_CLOSE, _OPEN)
                    )
                else:
                    self.clean[k] = nxt in (_CLOSE, _OPEN) or (
                        nxt == _PARAM
                        and k + 2 < n
                        and kinds[k + 2] == _PARAM_END
                        and self.clean[k + 2]
                    )
        self.match = [-1] * len(kinds)
        self.next_value_end = [-1] * len(kinds)  # first `</parameter>` after each tag
        ahead = -1
        for k in range(len(kinds) - 1, -1, -1):
            self.next_value_end[k] = ahead
            if kinds[k] == _PARAM_END:
                ahead = k
        stack: list[int] = []
        for k, kind in enumerate(kinds):
            if (
                kind == _OPEN
                and kinds[k + 1 : k + 2] == [_FUNC]
                and spaced[k + 1]
                and self._is_name(k + 1)
            ):
                stack.append(k)
            elif kind == _CLOSE and anchored[k] and stack:
                self.match[stack.pop()] = k

    def _is_name(self, k: int) -> bool:
        """Whether the `<function=` at tag `k` names a function, as opposed to a placeholder."""
        name = self.names[k].strip()
        return _NAME.fullmatch(name) is not None

    def _declared(self, k: int) -> bool:
        """Whether the call opened at tag `k` is one the request offered, when it offered any."""
        return not self.schemas or self.names[k + 1].strip() in self.schemas

    def param_at(self, k: int, allowed: _Allowed | None) -> bool:
        """Whether the `<parameter=` at tag `k` is a parameter of the call: any when `allowed` is
        None. Otherwise a declared one that was not given yet, or an undeclared one that is closed
        at once and followed only by well-formed parameters up to the call's end."""
        if allowed is None:
            return True
        name = self.names[k].strip()
        if name in allowed.open:
            return True
        return (
            name not in allowed.declared
            and k + 1 < self.count
            and self.kinds[k + 1] == _PARAM_END
            and self.clean[k + 1]
        )

    def closes_a_value(self, k: int, allowed: _Allowed | None) -> bool:
        """Whether the `</parameter>` at tag `k` is followed by the template's next tag."""
        if k + 1 < self.count:
            kind = self.kinds[k + 1]
            if kind == _PARAM and allowed is not None:
                return self.spaced[k + 1] and self.param_at(k + 1, allowed)
            return self.spaced[k + 1] and kind in _STRONG_NEXT
        return self.tail_spaced

    def value_end(self, first: int, allowed: _Allowed | None) -> tuple[int | None, int]:
        """Where the value whose text starts before tag `first` ends: (the tag index it stops at,
        or None when it runs to the end of the text; the tag index to read on from). With
        `allowed`, the names of the parameters that may still be given, a `<parameter=` of another
        name is text. A quoted call is skipped on the first reading; when that finds no proper end
        the value is read again with every call that starts at a line as the next call."""
        stop, resume, strong, skipped = self._scan(first, allowed, True)
        if strong or not skipped:
            return stop, resume
        return self._scan(first, allowed, False)[:2]

    def _scan(
        self, first: int, allowed: _Allowed | None, skip_quotes: bool
    ) -> tuple[int | None, int, bool, bool]:
        """One reading of a value: (where it stops, where to read on, whether it ends at a proper
        `</parameter>`, whether it skipped a quotation)."""
        kinds, anchored, match = self.kinds, self.anchored, self.match
        weak = good = soft = None
        closer, skipped, k = False, False, first
        while k < self.count:
            if weak is not None or good is not None or soft is not None:
                self.work -= 1
                if self.work < 0:
                    break
            kind = kinds[k]
            if kind == _PARAM_END:
                if self.closes_a_value(k, allowed):
                    return k, k + 1, True, skipped
                weak = k if weak is None else weak
            elif kind in (_FUNC_END, _CLOSE):
                closer = True
                good = k if good is None else good
            elif (kind == _PARAM and self.param_at(k, allowed)) or (
                kind == _OPEN and (anchored[k] or match[k] >= 0)
            ):
                if anchored[k]:
                    good = k if good is None else good
                else:
                    soft = k if soft is None else soft
                jump = self._quote(k, skip_quotes, weak is not None or closer, weak is not None)
                if jump < 0:
                    break
                skipped = skipped or jump != k
                k = jump
            k += 1
        if weak is not None:
            return weak, weak + 1, False, skipped
        stop = good if good is not None else soft
        return (
            (stop, stop, False, skipped) if stop is not None else (None, self.count, False, skipped)
        )

    def _quote(self, k: int, skip_quotes: bool, after_close: bool, after_weak: bool) -> int:
        """What a `<parameter=` or `<tool_call>` at tag `k` inside a value is: the tag to go on
        from when it begins a quotation (the `</parameter>` or `</tool_call>` that ends it) or
        is text (`k`), or -1 when it ends the value because it starts the next parameter or
        call."""
        if self.kinds[k] == _PARAM:
            end = self.next_value_end[k]
            if skip_quotes and end >= 0 and self.next_value_end[end] >= 0:
                return end
            return -1 if self.anchored[k] or after_weak else k
        end = self.match[k]
        if end < 0:
            return k  # a lone `<tool_call>` line is text: the value's own close may follow
        if skip_quotes and self.next_value_end[end] >= 0:
            return end
        next_call = self.anchored[k] and (after_close or not skip_quotes) and self._declared(k)
        return -1 if next_call else k

    def call(self, i: int) -> tuple[str, int, int, tuple[str, dict[str, str]] | None]:
        """Read the call whose `<tool_call>` is tag `i`: (what it is, where its text ends, the tag
        to read on from, its name and raw arguments). What it is: "call", "drop" for a call to
        leave out of the content and the calls, or "text" for an `<tool_call>` that is no call."""
        text, kinds, starts, ends = self.text, self.kinds, self.starts, self.ends
        n, j = self.count, i + 1
        if j >= n or kinds[j] != _FUNC or not self.spaced[j] or not self.names[j].strip():
            return self._nameless(i)
        name, params = self.names[j].strip(), {}
        if name not in self.declared:
            self.declared[name] = frozenset(self.schemas.get(name) or ())
        declared = self.declared[name]
        fclosed, last_closed = False, True
        k = j + 1
        while k < n:
            kind = kinds[k]
            if kind == _PARAM:
                key = self.names[k].strip()
                allowed = _Allowed(declared, params.keys() | {key}) if declared else None
                stop, resume = self.value_end(k + 1, allowed)
                end = len(text) if stop is None else starts[stop]
                raw = text[ends[k] : end].removeprefix("\n").removesuffix("\n")
                params[key] = raw
                last_closed = stop is not None and kinds[stop] == _PARAM_END
                k = resume
                continue
            if kind == _FUNC_END:
                fclosed = True
            elif kind == _CLOSE:
                return "call", ends[k], k + 1, (name, params)
            elif kind == _OPEN:
                done = last_closed and (params or fclosed)
                return ("call" if done else "drop"), starts[k], k, (name, params)
            k += 1
        done = fclosed and last_closed  # the reply ended after `</function>`
        return ("call" if done else "drop"), len(text), n, (name, params) if done else None

    def _nameless(self, i: int) -> tuple[str, int, int, None]:
        """A `<tool_call>` that does not open a function. It ends at the first `</tool_call>` or
        `<tool_call>`; prose that mentions the tag, with no other tag in it, stays content."""
        for k in range(i + 1, self.count):
            if self.kinds[k] == _CLOSE:
                return "drop", self.ends[k], k + 1, None
            if self.kinds[k] == _OPEN:
                prose = "<" not in self.text[self.ends[i] : self.starts[k]]
                return ("text" if prose else "drop"), self.starts[k], k, None
        return "drop", len(self.text), self.count, None


def _call(name: str, params: dict[str, str], schemas: dict[str, dict]) -> dict:
    """One call as an OpenAI tool call."""
    declared = schemas.get(name, {})
    args = {key: _value(raw, declared.get(key)) for key, raw in params.items()}
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def parse(text: str, tools: list[dict]) -> tuple[str, list[dict]]:
    """The reply's content and its tool calls. The content is the text outside the calls, less
    the whitespace before each call and at the end; a reply without `<tool_call>` is unchanged."""
    if OPEN not in text:
        return text, []
    schemas = _param_schemas(tools)
    tags = _Tags(text, schemas)
    kept, calls, pos, i = [], [], 0, 0
    while i < tags.count:
        if tags.kinds[i] != _OPEN:
            i += 1
            continue
        what, end, resume, found = tags.call(i)
        if what == "text":
            i += 1
            continue
        kept.append(text[pos : tags.starts[i]].rstrip())
        if what == "call" and found is not None:
            calls.append(_call(found[0], found[1], schemas))
        pos, i = end, resume
    kept.append(text[pos:])
    return "".join(kept).rstrip(), calls


class ToolCallText:
    """A streamed reply's text with its tool calls held back. `push` returns the content that is
    safe to send: all but trailing whitespace and a tail that may begin `<tool_call>`, and
    nothing from the first `<tool_call>` on. `finish` returns, at the end of the reply, the
    content not yet sent and the calls; together they are what `parse` makes of the whole text."""

    def __init__(self, tools: list[dict]):
        self.tools = tools
        self.pending = ""  # text held back before any call
        self.gap = ""  # the whitespace before the first `<tool_call>`
        self.held: str | None = None  # the text from the first `<tool_call>` on, once it appears

    def push(self, piece: str) -> str:
        if self.held is not None:
            self.held += piece
            return ""
        text = self.pending + piece
        start = text.find(OPEN)
        if start >= 0:
            self.held = text[start:]
            text = text[:start]
            keep = 0
        else:
            keep = next((n for n in range(len(OPEN) - 1, 0, -1) if text.endswith(OPEN[:n])), 0)
        safe = len(text[: len(text) - keep].rstrip())
        self.pending = text[safe:]
        if self.held is not None:
            self.gap, self.pending = self.pending, ""
        return text[:safe]

    def finish(self) -> tuple[str, list[dict]]:
        pending, self.pending = self.pending, ""
        if self.held is None:
            return pending, []
        return parse(self.gap + self.held, self.tools)
