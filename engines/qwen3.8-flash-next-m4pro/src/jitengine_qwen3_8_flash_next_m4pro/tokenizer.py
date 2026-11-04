"""Tokenizer: the checkpoint's AutoTokenizer and chat template, and IncrementalDecoder, which renders
a stream.

Chat rendering is hardened because client input reaches the Jinja template and the encoder:
`chat_template_kwargs` is an allow-list of `enable_thinking` alone (other names can be
`apply_chat_template` parameters such as `chat_template`), and the request's `tools` reach it as
`tools`; the rendered text is bounded before encoding; and control-token text in messages or
tools is encoded as ordinary text, character for character (`_shield`, `_encode_rendered`),
because the fast tokenizer matches added tokens whatever `add_special_tokens` says. As a backstop,
the rendered control-token ids must match those of the same messages with placeholder content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Sequence

# The one template variable a client may set; everything else is refused by name.
ALLOWED_CHAT_TEMPLATE_KWARGS = ("enable_thinking",)
# Shield sentinels are drawn from the top of the last private-use plane, skipping any character the
# request itself holds, so a sentinel in the rendered text is always one the shield put there.
_SENTINEL_FIRST = 0x10FFFD
MAX_RENDERED_CHARS = (
    4 * 2**20
)  # bound on the rendered text (surfaces at full length), before encoding
MAX_PROMPT_TOKENS = 32768  # per-request prompt bound (server/protocol.py refuses past it)
# Placeholder renders kept for the backstop, least recently used first out; each is a digest and a
# handful of counts, so the bound is on entries, not bytes.
BLANK_CACHE_ENTRIES = 64


class ChatTemplateError(ValueError):
    """A chat request refused before any model work (the server answers 400 with this message)."""


def _with_call_arguments(messages: list[dict]) -> list[dict]:
    """`messages` with each tool call's `arguments` as the object the template iterates: OpenAI
    clients send them as a JSON string."""
    out = []
    for i, m in enumerate(messages):
        calls = m.get("tool_calls")
        if not calls:
            out.append(m)
            continue
        if not isinstance(calls, list):
            raise ChatTemplateError(f"message {i} has tool_calls that are not a list")
        fixed = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ChatTemplateError(f"message {i} has a tool call without a function name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = None
            if not isinstance(args, dict):
                raise ChatTemplateError(
                    f"message {i} has a tool call whose arguments are not a JSON object"
                )
            fixed.append({**call, "function": {**fn, "arguments": args}})
        out.append({**m, "tool_calls": fixed})
    return out


def _text_of(content) -> str | None:
    """The text the template renders for `content`: a string, or a list of text parts. None when it
    holds anything else (an image or video part renders as vision markers, not as text)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    out = []
    for part in content:
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            return None
        if "image" in part or "image_url" in part or "video" in part:
            return None
        if part.get("type") in ("image", "video"):
            return None
        out.append(part["text"])
    return "".join(out)


def _split_think(content: str) -> tuple[str, str] | None:
    """(reasoning, reply) the way the Qwen3.6 template splits an assistant turn's content on
    `</think>`, or None when it holds none."""
    content = content.strip()
    if "</think>" not in content:
        return None
    head, reply = content.split("</think>")[0], content.split("</think>")[-1]
    return head.rstrip("\n").split("<think>")[-1].lstrip("\n"), reply.lstrip("\n")


def _blank_type(key, value):
    """A `type` field that names a kind (`function`, `text`, `image`) stays, since templates branch
    on it; anything else is blanked like any client string."""
    if key == "type" and isinstance(value, str) and re.fullmatch(r"[A-Za-z_]\w{0,31}", value):
        return value
    return _blank_value(value)


def _blank_value(value):
    """`value` with every string replaced by "." (an empty or blank one by ""), keeping the shape."""
    if isinstance(value, str):
        return "." if value.strip() else ""
    if isinstance(value, list):
        return [_blank_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _blank_type(k, v) for k, v in value.items()}
    return value


def _blank_message(m: dict, counter: list[int]) -> dict:
    """`m` with every client string neutralized the way the placeholder render needs: text becomes
    "." (empty stays empty), a tool call's name and its argument keys become distinct names
    (`n0`, `k1`, ...), the structure and the role stay. The template then writes its own text next
    to placeholders no surface can be built from."""
    out = {}
    for key, value in m.items():
        if key == "role":
            out[key] = value
        elif key == "tool_calls" and isinstance(value, list):
            out[key] = [_blank_call(c, counter) for c in value]
        else:
            out[key] = _blank_value(value)
    return out


def _blank_call(call, counter: list[int]):
    if not isinstance(call, dict):
        return _blank_value(call)
    out = {k: _blank_type(k, v) for k, v in call.items() if k != "function"}
    fn = call.get("function")
    if isinstance(fn, dict):
        fixed = {k: _blank_value(v) for k, v in fn.items() if k not in ("name", "arguments")}
        counter[0] += 1
        fixed["name"] = f"n{counter[0]}"
        args = fn.get("arguments")
        if isinstance(args, dict):
            fixed["arguments"] = {}
            for v in args.values():
                counter[0] += 1
                fixed["arguments"][f"k{counter[0]}"] = _blank_value(v)
        else:
            fixed["arguments"] = _blank_value(args)
        out["function"] = fixed
    return out


class Tokenizer:
    def __init__(self, model_path: str):
        # transformers' import warns that PyTorch is missing, which a tokenizer never needs
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.vocab_size = len(self.tok)
        added = self.tok.get_added_vocab()
        # longest first so an error names the most specific marker it found
        self.added_surfaces = tuple(sorted(added, key=lambda s: (-len(s), s)))
        # an unknown-word id is not a control token: a placeholder name the vocabulary lacks maps to it
        unk = getattr(self.tok, "unk_token_id", None)
        self.added_ids = frozenset(int(i) for i in added.values()) - {unk}
        self._surface_re = re.compile("|".join(re.escape(x) for x in self.added_surfaces))
        self._splits_think: bool | None = None  # does the template split content on </think>
        # a surface cut short at the end of a client string, or at its start: the template's own
        # text beside it could complete it
        prefixes = {x[:i] for x in added for i in range(1, len(x))}
        suffixes = {x[i:] for x in added for i in range(1, len(x))}

        def longest_first(xs):
            return "|".join(re.escape(x) for x in sorted(xs, key=lambda x: (-len(x), x)))

        self._partial_re = (
            re.compile(f"(?:{longest_first(prefixes)})\\Z|\\A(?:{longest_first(suffixes)})")
            if added
            else re.compile("(?!)")
        )
        self._plain = None  # the tokenizer without added tokens, built on first use (False: none)
        self._blank_cache: OrderedDict = OrderedDict()
        self._blank_lock = threading.Lock()  # handler threads render concurrently

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tok(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------ chat rendering
    def _check_kwargs(self, chat_template_kwargs: dict) -> dict:
        kw = dict(chat_template_kwargs or {})
        for key in sorted(kw):
            if key not in ALLOWED_CHAT_TEMPLATE_KWARGS:
                raise ChatTemplateError(
                    f"chat_template_kwargs key {key!r} is not allowed; the only accepted key is "
                    f"{ALLOWED_CHAT_TEMPLATE_KWARGS[0]!r}"
                )
        if "enable_thinking" in kw and not isinstance(kw["enable_thinking"], bool):
            raise ChatTemplateError("chat_template_kwargs.enable_thinking must be a boolean")
        out = {"enable_thinking": False}
        out.update(kw)
        return out

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

    def _shield(self, messages, tools, partial: bool = False):
        """`messages` and `tools` with every control-token surface in their strings (keys
        included) replaced by a private-use sentinel character, and the map from sentinel to
        surface. The fast tokenizer matches added tokens in any text it encodes, so text that is
        part of a conversation (a tool result that quotes `</think>`, a file the model wrote that
        holds `<tool_call>`) would otherwise render as a real control token; refusing it made one
        such string fail every later turn, since an agent resends its whole history. The sentinel
        goes through the template like any character, and `_encode_rendered` puts the surface
        back as ordinary text, so the model sees the client's exact characters (a surface
        copied into a later edit still matches the file) while control tokens in the render come
        only from the template. A sentinel is a character the request does not contain.

        With `partial`, a proper prefix of a surface at the end of a string and a proper suffix of
        one at its start are shielded too, the characters still exact: template text written
        right after or before a client string (`<function=` + name + `>`) cannot then complete a
        surface out of it."""
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
                obj = self._surface_re.sub(lambda m: sentinel_for(m.group()), obj)
                if partial:
                    obj = self._partial_re.sub(lambda m: sentinel_for(m.group()), obj)
                return obj
            if isinstance(obj, dict):
                return {walk(k): walk(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [walk(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(walk(v) for v in obj)
            return obj

        out_messages, out_tools = walk(messages), walk(tools)
        return out_messages, out_tools, {c: surface for surface, c in sentinels.items()}

    def _template_splits_think(self, kw: dict) -> bool:
        """Whether the template reads `</think>` in an assistant turn's content as the end of its
        reasoning (Qwen3.6 does; Flash-Next leaves it as text). Found by asking it: the same turn
        with the tag in the content and with the parts given as reasoning and reply render alike."""
        if self._splits_think is None:

            def turn(**fields):
                msgs = [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", **fields},
                    {"role": "user", "content": "go"},
                ]
                return self._apply(msgs, kw, None)

            from jinja2 import TemplateError

            try:
                inline = turn(content="RRR</think>AAA")
                parts = turn(content="AAA", reasoning_content="RRR")
                self._splits_think = inline == parts
            except TemplateError, ValueError, TypeError:
                self._splits_think = False
        return self._splits_think

    def _split_reasoning(self, messages: list[dict], kw: dict) -> list[dict]:
        """`messages` with each assistant turn's `</think>` split done as the template would, before
        the shield turns the tag into a sentinel the template cannot split on. The reasoning and the
        reply then reach the template as `reasoning_content` and `content`, and every other
        surface in them is shielded as usual. A turn that has `reasoning_content` is left alone,
        as the template does; so is one whose content holds a non-text part."""
        if not any(m.get("role") == "assistant" for m in messages):
            return messages
        if not self._template_splits_think(kw):
            return messages
        out = []
        for m in messages:
            text = _text_of(m.get("content")) if m.get("role") == "assistant" else None
            split = None
            if text is not None and not isinstance(m.get("reasoning_content"), str):
                split = _split_think(text)
            if split is None:
                out.append(m)
            else:
                out.append({**m, "reasoning_content": split[0], "content": split[1]})
        return out

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

    def _apply(self, messages: list[dict], kw: dict, tools: list | None) -> str:
        return self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **kw
        )

    @staticmethod
    def _blank_key(blank: list[dict], kw: dict, tools: list | None) -> str | None:
        """A digest of the whole placeholder render input (every field of every message, not only
        the roles: `tool_calls`, for one, changes the control tokens the template emits, and so do
        the tools offered), or None when it is not JSON, which is then rendered every time."""
        try:
            text = json.dumps(
                [blank, kw, tools], sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
        except TypeError, ValueError:
            return None
        return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()

    def _blank_control_counts(
        self, messages: list[dict], kw: dict, tools: list | None, surfaces: dict[str, str]
    ) -> tuple[Counter, str | None]:
        """Counts of added-token ids the template emits for these messages with placeholder content
        and these tools, and the cache key to keep them under once the request is accepted (None
        when they came from the cache or cannot be cached).

        The placeholder keeps what the template looks at: a content that is empty (or only
        whitespace) stays empty, since a template may drop or shape a turn on empty content (Flash-Next
        drops an empty system turn), and any other text becomes ".". Image and video parts stay, so their markers
        count.
        """
        counter = [0]
        blank = [_blank_message(m, counter) for m in messages]
        tools = _blank_value(tools) if tools else tools
        key = self._blank_key(blank, kw, tools)
        if key is not None:
            with self._blank_lock:
                got = self._blank_cache.get(key)
                if got is not None:
                    self._blank_cache.move_to_end(key)
                    return got, None
        ids = self._encode_rendered(self._apply(blank, kw, tools), surfaces)
        return Counter(t for t in ids if t in self.added_ids), key

    def _keep_blank(self, key: str, counts: Counter) -> None:
        with self._blank_lock:
            self._blank_cache[key] = counts
            self._blank_cache.move_to_end(key)
            while len(self._blank_cache) > BLANK_CACHE_ENTRIES:
                self._blank_cache.popitem(last=False)

    def render_chat(
        self, messages: list[dict], tools: list | None = None, **chat_template_kwargs
    ) -> list[int]:
        """Token ids of the checkpoint's template applied to `messages` and `tools`, with a
        generation prompt. Tool-call arguments sent as a JSON string are parsed first: the
        template iterates them.

        A control-token surface in a message or tool is shielded first, so it encodes as the
        ordinary text it is, character for character; the backstop only refuses
        (`ChatTemplateError`). A request without such text renders exactly as the template does.
        An assistant turn's `</think>` is split as the template does, before shielding, when the
        template splits it. The backstop's placeholder render treats each message as the template
        does (an empty one stays empty), so an empty system message needs no special case.
        """
        kw = self._check_kwargs(chat_template_kwargs)
        messages = _with_call_arguments(messages)
        tools = tools or None
        messages = self._split_reasoning(messages, kw)
        shielded = self._shield(messages, tools)
        tried = self._render_shielded(*shielded, kw)
        if isinstance(tried, tuple):
            return tried[0]
        # A client string that ends with a part of a surface, or starts with the rest of one, can
        # be completed by the template's own text beside it (`<function=` + name + `>`). That is
        # caught here, as a control id the placeholder render does not have; the request is then
        # tried again with such partial surfaces shielded as well, which is exact for the client's
        # characters and cannot merge with template text. A request without them renders as the
        # template does on the first try.
        again = self._shield(messages, tools, partial=True)
        if again[:2] == shielded[:2]:
            raise tried
        done = self._render_shielded(*again, kw)
        if isinstance(done, tuple):
            return done[0]
        raise done

    def _render_shielded(self, messages, tools, surfaces, kw):
        """(ids,) for shielded messages and tools, or the `ChatTemplateError` that refuses them."""
        text = self._apply(messages, kw, tools)
        # a sentinel stands for its whole surface, so it counts at the surface's length
        size = len(text) + sum(text.count(c) * (len(s) - 1) for c, s in surfaces.items())
        if size > MAX_RENDERED_CHARS:
            raise ChatTemplateError(
                f"the chat template rendered {size} characters, above the limit of "
                f"{MAX_RENDERED_CHARS}"
            )
        ids = self._encode_rendered(text, surfaces)
        seen = Counter(t for t in ids if t in self.added_ids)
        expect, key = self._blank_control_counts(messages, kw, tools, surfaces)
        if seen != expect:
            extra = {
                int(t): int(seen[t] - expect.get(t, 0)) for t in seen if seen[t] != expect.get(t, 0)
            }
            return ChatTemplateError(
                f"message content produced control tokens the template did not emit ({extra}); "
                f"control tokens are a function of the message shape and the template alone"
            )
        # Only a request that will be served is remembered: a refused one (here, or past the prompt
        # bound in server/protocol.py) leaves nothing behind.
        if key is not None and len(ids) <= MAX_PROMPT_TOKENS:
            self._keep_blank(key, expect)
        return (ids,)


class IncrementalDecoder:
    """Feed token ids one at a time; `push` returns the newly rendered text for that step."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ""

    def push(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        full = self.tokenizer.decode(self.ids)
        if full.endswith("�"):  # incomplete multi-byte sequence: wait for the rest
            return ""
        if full.startswith(self.text):
            delta = full[len(self.text) :]
        else:
            delta = full
        self.text = full
        return delta
