"""Tokenizer and chat rendering with the checkpoint's own files.

Rendering is `apply_chat_template(messages, tools, add_generation_prompt=True, tokenize=True)` over the
checkpoint's chat_template.jinja, i.e. the same ids an independent transformers render produces for
ordinary prompts. The template escapes control-token text (its `flint_text` macro) in the content of
system, developer, user and tool messages, so text there never becomes a control token. It renders
every other field verbatim, so `_normalise` and `render_chat` apply the same escape (`flint_text`
below, the macro's replacements) to every other client string that reaches the template: assistant
content, reasoning_content, tool calls (function names, argument keys and values) and the `tools`
definitions (names, descriptions and every string of a schema). Text with no control token and no
`<think>` tag renders as before. What remains of a control-token surface after that escape
(a `<tool_call>`, or a `<think>` rebuilt around a removed one) is encoded as ordinary text, character
for character (`_shield`), so no client string becomes a control-token id. OpenAI-style message parts are flattened to text and tool-call
arguments given as JSON strings are parsed, because the template iterates `arguments.items()`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence

# The control tokens the checkpoint's template escapes in message content (`flint_text`): each
# becomes its HTML-escaped form. Its <think> tags are removed, as is NUL.
CONTROL_TOKENS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)


def flint_text(text: str) -> str:
    """`text` as the template's `flint_text` macro renders it, so control-token text stays text."""
    text = text.replace("\x00", "").replace("<think>", "").replace("</think>", "")
    for token in CONTROL_TOKENS:
        text = text.replace(token, token.replace("<", "&lt;").replace(">", "&gt;"))
    return text


def escape_all(value):
    """`value` with `flint_text` applied to every string in it, dictionary keys included."""
    if isinstance(value, str):
        return flint_text(value)
    if isinstance(value, dict):
        return {escape_all(k): escape_all(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [escape_all(v) for v in value]
    return value


# Shield sentinels are drawn from the top of the last private-use plane, skipping any character the
# request itself holds, so a sentinel in the rendered text is always one the shield put there.
_SENTINEL_FIRST = 0x10FFFD


class ChatTemplateError(ValueError):
    """A chat request refused before any model work (the server answers 400 with this message)."""


class Tokenizer:
    def __init__(self, model_path: str):
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        added = self.tok.get_added_vocab()
        # longest first so a surface that holds another is matched whole
        self.added_surfaces = tuple(sorted(added, key=lambda x: (-len(x), x)))
        self.added_ids = frozenset(int(i) for i in added.values())
        self._surface_re = re.compile("|".join(re.escape(x) for x in self.added_surfaces))
        self._plain = None  # the tokenizer without added tokens, built on first use (False: none)

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tok(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    def piece(self, token_id: int) -> str:
        return self.tok.convert_ids_to_tokens(int(token_id)) or ""

    @staticmethod
    def _normalise(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            m = dict(m)
            c = m.get("content")
            if isinstance(c, list):
                m["content"] = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in c
                )
            elif c is None:
                m["content"] = ""
            # the template escapes system, developer, user and tool content itself, once; doing it
            # first keeps those roles as the template renders them and lets `_shield` see what
            # the template's single pass leaves (a tag rebuilt around a removed <think>)
            if isinstance(m["content"], str):
                m["content"] = flint_text(m["content"])
            if isinstance(m.get("reasoning_content"), str):
                m["reasoning_content"] = flint_text(m["reasoning_content"])
            calls = m.get("tool_calls")
            if calls:
                fixed = []
                for tc in calls:
                    tc = json.loads(json.dumps(tc))
                    fn = tc.get("function", tc)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args) if args.strip() else {}
                        except ValueError:
                            fn["arguments"] = {"arguments": args}
                    elif args is None:
                        fn["arguments"] = {}
                    fn["arguments"] = escape_all(fn["arguments"])
                    if isinstance(fn.get("name"), str):
                        fn["name"] = flint_text(fn["name"])
                    fixed.append(tc)
                m["tool_calls"] = fixed
            out.append(m)
        return out

    # ------------------------------------------------------------------ control-token text
    @staticmethod
    def _chars(obj, seen: set) -> None:
        """Add every character of every string in `obj` (keys included) to `seen`."""
        if isinstance(obj, str):
            seen.update(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                Tokenizer._chars(k, seen)
                Tokenizer._chars(v, seen)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                Tokenizer._chars(v, seen)

    def _shield(self, messages, tools):
        """`messages` and `tools` with every control-token surface in their strings (keys
        included) replaced by a private-use sentinel character, and the map from sentinel to
        surface. The fast tokenizer matches added tokens in any text it encodes, so text that is
        part of a conversation (a tool result that quotes `</think>`, a file the model wrote that
        holds `<tool_call>`) would otherwise render as a real control token; refusing it made one
        such string fail every later turn, since an agent resends its whole history. The sentinel
        goes through the template like any character, and `_encode_rendered` puts the surface
        back as ordinary text, so the model sees the client's exact characters (a surface
        copied into a later edit still matches the file) while control tokens in the render come
        only from the template. A sentinel is a character the request does not contain."""
        used: set = set()
        self._chars(messages, used)
        self._chars(tools, used)
        sentinels: dict[str, str] = {}
        nxt = [_SENTINEL_FIRST]

        def sentinel_for(surface: str) -> str:
            if surface not in sentinels:
                while chr(nxt[0]) in used:
                    nxt[0] -= 1
                sentinels[surface] = chr(nxt[0])
                nxt[0] -= 1
            return sentinels[surface]

        def walk(obj):
            if isinstance(obj, str):
                return self._surface_re.sub(lambda m: sentinel_for(m.group()), obj)
            if isinstance(obj, dict):
                return {walk(k): walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [walk(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(walk(v) for v in obj)
            return obj

        out_messages, out_tools = walk(messages), walk(tools)
        return out_messages, out_tools, {c: surface for surface, c in sentinels.items()}

    def _plain_ids(self, surface: str) -> list[int]:
        """Token ids of `surface` read as ordinary text: the tokenizer without its added tokens,
        or, if that cannot be built or still yields a control id, one character at a time."""
        plain = self._plain
        if plain is None:
            from tokenizers import Tokenizer as FastTokenizer

            backend = getattr(self.tok, "backend_tokenizer", None)
            spec = json.loads(backend.to_str()) if backend is not None else None
            if spec is not None:
                spec["added_tokens"] = []
                plain = FastTokenizer.from_str(json.dumps(spec))
            self._plain = plain or False
        if plain:
            ids = [int(t) for t in plain.encode(surface, add_special_tokens=False).ids]
            if not self.added_ids.intersection(ids):
                return ids
        ids = [i for ch in surface for i in self.encode(ch)]
        if self.added_ids.intersection(ids):
            raise ChatTemplateError("a control-token surface in the request cannot be read as text")
        return ids

    def _encode_rendered(self, text: str, surfaces: dict[str, str]) -> list[int]:
        """Token ids of the rendered `text`: the template's own text as the tokenizer reads it,
        and each sentinel as the ordinary-text ids of the surface it stands for."""
        if not surfaces or not any(c in text for c in surfaces):
            return self.encode(text)
        ids: list[int] = []
        parts = re.split("([" + re.escape("".join(surfaces)) + "])", text)
        for part in parts:
            if part in surfaces:
                ids += self._plain_ids(surfaces[part])
            elif part:
                ids += self.encode(part)
        return ids

    def _encode_rendered(self, text: str, surfaces: dict[str, str]) -> list[int]:
        """Token ids of the rendered `text`: the template's own text as the tokenizer reads it,
        and each sentinel as the ordinary-text ids of the surface it stands for."""
        parts = re.split("([" + re.escape("".join(surfaces)) + "])", text)
        ids: list[int] = []
        for part in parts:
            if part in surfaces:
                ids += self._plain_ids(surfaces[part])
            elif part:
                ids += self.encode(part)
        return ids

    def render_chat(self, messages: list[dict], tools: list | None = None) -> list[int]:
        """Token ids of the checkpoint's template applied to `messages` and `tools`, with a
        generation prompt. Control-token text in any client string (a role's content, a tool
        call, a tool definition) reaches the model as its exact characters, read as ordinary
        text, after the template's own escape (`flint_text`); a request with none renders
        exactly as `apply_chat_template` does."""
        msgs = self._normalise(messages)
        tl = escape_all(tools) if tools else None
        msgs, tl, surfaces = self._shield(msgs, tl)
        kw = {"tools": tl} if tl else {}
        if not surfaces:
            ids = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_dict=False, **kw
            )
            return [int(t) for t in ids]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
        return self._encode_rendered(text, surfaces)


class IncrementalDecoder:
    """Text deltas for streaming: never splits a UTF-8 character (holds back a trailing U+FFFD).

    Incremental (re-decoding every id per token cost O(n) per token, 7.5 ms at 64K): only
    the ids after the last clean character boundary are decoded per push, usually one. Exact because the checkpoint's
    decoder is byte-level (concatenated token bytes, lossy UTF-8, no space clean-up): once ids[:p] decode without a
    trailing U+FFFD their bytes end on a character boundary, and decode(ids) == decode(ids[:p]) + decode(ids[p:]).
    `text` is therefore always what the full re-decode gave (tests: every prefix of long real and random streams)."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ""
        self._p = 0  # ids[:_p] decode to `text`, ending on a character boundary

    def push(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        tail = self.tokenizer.decode(self.ids[self._p :])
        if tail.endswith(
            "\ufffd"
        ):  # an incomplete character (or an invalid byte) at the end: hold it back
            return ""
        self.text += tail
        self._p = len(self.ids)
        return tail

    def flush(self) -> str:
        tail = self.tokenizer.decode(self.ids[self._p :])
        self.text += tail
        self._p = len(self.ids)
        return tail


class StopMatcher:
    def __init__(self, stops: Sequence[str] | None):
        self.stops = [s for s in (stops or ()) if isinstance(s, str) and s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1

    def __bool__(self) -> bool:
        return bool(self.stops)

    def find(self, text: str) -> int | None:
        """Index where the earliest stop string starts, or None."""
        best = None
        for s in self.stops:
            i = text.find(s)
            if i >= 0 and (best is None or i < best):
                best = i
        return best

    def safe_prefix(self, text: str) -> int:
        """How much of `text` can be emitted without risking a split stop string."""
        return max(0, len(text) - self.hold) if self.stops else len(text)
