"""The server's port binding, off a Mac: a live listener on the port refuses the bind, also a wildcard
or dual-stack one, and only the TIME_WAIT ends of a previous server are bound over."""

from __future__ import annotations

import errno
import socket
import sys

import pytest

from jitengine_flint_m2.server import Handler, Server, _live_listener


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _bind_error(port: int):
    """The errno of Server's bind on 127.0.0.1:port, or None when it binds."""
    try:
        srv = Server(("127.0.0.1", port), Handler)
    except OSError as exc:
        return exc.errno
    srv.server_close()
    return None


def test_live_listener_probe():
    """Refused on a free port; answered by a live listener, also for the wildcard host."""
    assert not _live_listener(("127.0.0.1", _free_port()))
    with socket.socket() as ls:
        ls.bind(("127.0.0.1", 0))
        ls.listen(8)
        port = ls.getsockname()[1]
        assert _live_listener(("127.0.0.1", port))
        assert _live_listener(("0.0.0.0", port))


@pytest.mark.parametrize("wild", ["0.0.0.0", "::"])
def test_no_wildcard_shadow(wild):
    """With a LIVE listener on 0.0.0.0:P, or on a dual-stack [::]:P (IPV6_V6ONLY 0), a server on
    127.0.0.1:P refuses the bind (EADDRINUSE) instead of binding beside it with SO_REUSEADDR and
    answering the port's loopback clients in its place (BSD semantics, as on macOS); the original
    listener still takes 127.0.0.1:P afterwards."""
    six = ":" in wild
    try:
        first = socket.socket(socket.AF_INET6 if six else socket.AF_INET)
        if six:
            first.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        first.bind((wild, 0))
    except OSError:
        if six:
            pytest.skip("no dual-stack IPv6 on this host")
        raise
    with first:
        first.listen(8)
        first.settimeout(5)
        port = first.getsockname()[1]
        got = _bind_error(port)
        assert got == errno.EADDRINUSE, (
            f"127.0.0.1:{port} beside a live {wild} listener: errno {got}, want EADDRINUSE"
        )
        with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
            mine = c.getsockname()[1]
            for _ in range(8):  # the backlog also holds the probe's connection
                a, peer = first.accept()
                a.close()
                if peer[1] == mine:
                    break
            else:
                raise AssertionError(f"127.0.0.1:{port} is no longer taken by the {wild} listener")


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="BSD rule: SO_REUSEADDR on the new socket alone binds over TIME_WAIT; Linux also needs it on the old one",
)
def test_rebind_over_time_wait():
    """A restart on the same port binds although the previous server's connection end sits in
    TIME_WAIT (the server closes first): the plain bind is refused, no live listener answers, so
    the server rebinds with SO_REUSEADDR."""
    with socket.socket() as ls:
        ls.bind(("127.0.0.1", 0))
        ls.listen(1)
        port = ls.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
            a, _ = ls.accept()
            a.close()  # the server's end closes first
            assert c.recv(1) == b""
    with socket.socket() as plain, pytest.raises(OSError) as refused:
        plain.bind(("127.0.0.1", port))
    assert refused.value.errno == errno.EADDRINUSE, "nothing holds the port: no TIME_WAIT end"
    assert _bind_error(port) is None
