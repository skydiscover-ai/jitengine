"""Host and origin checks: the API has no authentication, so a request for another host name or from
a browser page of another origin is refused before any work. A page whose domain re-resolves to
127.0.0.1 (DNS rebinding) would otherwise be same-origin with the server and read every response.

The functions are from engines/qwen3.6-35b-a3b-m4pro (same repository, Apache-2.0)."""

from __future__ import annotations

import ipaddress
import json
import re

# Host names always served, besides IP literals (which DNS cannot rebind) and the bind address. The
# API has no authentication, so any other name in Host is refused: a page whose domain re-resolves
# to 127.0.0.1 would otherwise be same-origin with the server and read every response (DNS
# rebinding, as Ollama's CVE-2024-28224). `serve(..., allowed_hosts=)` adds names.
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
WILDCARD_BINDS = ("", "0.0.0.0", "::")


_ORIGIN = re.compile(r"(https?)://([^/?#\\\s@]+)/?")
_HOST_LABEL = re.compile(r"[a-z0-9_]([a-z0-9_-]*[a-z0-9_])?")
_PORT = re.compile(r"[0-9]{1,5}")


def host_name(authority: str) -> str | None:
    """The host of a Host header value or URL authority ("name", "name:port", "[v6]:port"),
    lower-cased without its port or a trailing dot; None when it is not exactly that: userinfo
    (`@`), backslashes, spaces, a path or any other character a host cannot hold, a port that is
    not digits, a name that is not dot-separated labels or an IP literal."""
    a = authority.strip()
    if not a or not a.isascii() or not all(c.isalnum() or c in "-._:[]%" for c in a):
        return None
    if a.startswith("["):
        end = a.find("]")
        rest = a[end + 1 :] if end > 0 else "x"
        if rest and not (rest.startswith(":") and _PORT.fullmatch(rest[1:])):
            return None
        name = a[1:end] if end > 0 else ""
        if ":" not in name or _ip(name) is None:
            return None
    elif a.count(":") == 1:
        name, port = a.split(":", 1)
        if not _PORT.fullmatch(port):
            return None
    elif a.count(":") > 1:
        name = a  # an unbracketed IPv6 literal
        if _ip(name) is None:
            return None
    else:
        name = a
    name = name.lower().rstrip(".")
    if not name:
        return None
    if ":" not in name and _ip(name) is None:
        if not all(_HOST_LABEL.fullmatch(label) for label in name.split(".")):
            return None
    return name


def _ip(name: str):
    try:
        return ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return None


def served_names(bind_host: str, extra=()) -> frozenset:
    """The host names a request may name: the loopback names, the bind address unless it is a
    wildcard, and `extra`."""
    names = set(LOOPBACK_NAMES)
    b = host_name(bind_host or "")
    if b and b not in WILDCARD_BINDS:
        names.add(b)
    names.update(n for n in (host_name(str(x)) for x in extra) if n)
    return frozenset(names)


def refusal(headers, names: frozenset) -> str | None:
    """Why a request must be refused before any work, or None. `Host` must be a served name or an
    IP literal (a missing one is not a browser's: HTTP/1.1 browsers always send it). `Origin`,
    which browsers send on cross-origin requests and on every POST, must be a loopback or served
    host, or the request's own; non-browser clients (the OpenAI SDKs, curl) send none."""
    host = headers.get("Host")
    hname = None
    if host is not None:
        hname = host_name(host)
        if hname is None or (hname not in names and _ip(hname) is None):
            return f"host {host!r} is not served here"
    origin = headers.get("Origin")
    if origin is not None:
        m = _ORIGIN.fullmatch(origin.strip())
        oname = host_name(m.group(2)) or "" if m else ""  # "null" (a sandboxed page) has none
        oname = oname or None
        ip = _ip(oname) if oname else None
        if oname is None or not (
            oname in names or oname == hname or (ip is not None and ip.is_loopback)
        ):
            return f"origin {origin!r} may not use this server"
    return None


def refuse(handler, why: str) -> None:
    """Send a 403 and close the connection; no body is read."""
    raw = json.dumps(
        {
            "error": {
                "message": f"forbidden: {why}",
                "type": "invalid_request_error",
                "code": "forbidden",
            }
        }
    ).encode()
    handler.close_connection = True
    handler._set_completion_headers(403)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(raw)
    handler.wfile.flush()


def install(handler_class, names: frozenset) -> None:
    """Check every GET, POST and OPTIONS before the handler sees it."""
    for method in ("do_GET", "do_POST", "do_OPTIONS"):
        original = getattr(handler_class, method)

        def checked(self, _original=original):
            why = refusal(self.headers, names)
            if why is not None:
                return refuse(self, why)
            return _original(self)

        setattr(handler_class, method, checked)
