"""The prefix store's spill (`kvcache.store_spill`), on MLX's CPU build or Metal: entries written to
the scratch file behind the caller and read back byte for byte, the store refusing lookups and
admissions while spilled, a lost entry dropped as a miss, a failed write keeping its entry, and the
file's space reused without ever overwriting a live entry's bytes."""

# ruff: noqa: E402
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from jitengine_qwen3_8_flash_next_m4pro.kvcache import store_spill as _ss
from jitengine_qwen3_8_flash_next_m4pro.kvcache.prefix_cache import Entry, PrefixCache
from jitengine_qwen3_8_flash_next_m4pro.kvcache.store_spill import StoreSpill


def _entry(key, seed, n=5):
    r = np.random.default_rng(seed)
    kv = (
        "kv",
        mx.array(r.normal(size=(1, 2, n, 8)).astype(np.float32)).astype(mx.bfloat16),
        mx.array(r.normal(size=(1, 2, n, 8)).astype(np.float32)),
        mx.array(r.normal(size=(1, 1, n, 4)).astype(np.float32)).astype(mx.bfloat16),
    )
    delta = (
        "delta",
        mx.array(r.normal(size=(1, 3, 6)).astype(np.float32)),
        mx.array(r.normal(size=(1, 2, 4, 4)).astype(np.float32)),
    )
    ple = [
        (
            "ple",
            1,
            mx.array(r.integers(0, 1 << 40, size=(1, 2))),
            mx.array(r.normal(size=(1, 3, 8)).astype(np.float32)),
        )
    ]
    e = Entry(tuple(key), [kv, delta], ple)
    mx.eval(*e.arrays())
    return e


def _bytes(e):
    return [
        np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes() for a in e.arrays()
    ]


def _store(tmp_path, cap=1 << 20, spiller=None):
    s = PrefixCache(cap_bytes=cap, min_divergence=1)
    s.spiller = spiller or StoreSpill(str(tmp_path), nocache=False)
    return s


def _spill(s):
    """`spill` as the scheduler drives it: start the writes, let them land, then drop."""
    got = s.spill()
    if got is None:
        s.spiller.wait()
        got = s.spill()
    assert got is not None and s.spilled
    return got


def test_spill_and_restore_give_back_every_byte(requires_mlx_any, tmp_path):
    s = _store(tmp_path)
    a, b = _entry(range(10), 1), _entry(range(20, 36), 2)
    want = {a.key: _bytes(a), b.key: _bytes(b)}
    assert s.admit(a) and s.admit(b)
    freed = _spill(s)
    assert freed == a.nbytes + b.nbytes and not a.resident and not b.resident
    assert all(x is None for t in a.layers for x in t[1:])
    assert s.covered_len(list(range(12))) == 10  # keys still answer
    with pytest.raises(RuntimeError, match="spilled"):
        s.lookup(list(range(12)))
    with pytest.raises(RuntimeError, match="spilled"):
        s.admit(_entry(range(40, 44), 3))
    s.restore()
    assert not s.spilled and not a.resident and s.spiller.bytes_read == 0  # read back on demand
    got = s.lookup(list(range(12)))
    assert got is a and a.resident and not b.resident and s.spiller.bytes_read == a.nbytes
    assert _bytes(a) == want[a.key]
    assert s.lookup(list(range(20, 40))) is b and _bytes(b) == want[b.key]
    assert [x.dtype for x in a.arrays()] == [
        mx.bfloat16,
        mx.float32,
        mx.bfloat16,
        mx.float32,
        mx.float32,
        mx.int64,
        mx.float32,
    ]
    # a second spill writes nothing new: the entries are immutable and already on file
    w0 = s.spiller.bytes_written
    assert s.spill() == a.nbytes + b.nbytes  # nothing to write: it drops at once
    assert s.spiller.bytes_written == w0
    s.restore()
    assert s.lookup(list(range(12))) is a and s.lookup(list(range(20, 40))) is b
    assert {e.key: _bytes(e) for e in (a, b)} == want
    st = s.stats()
    assert st["spills"] == 1 and st["spill_read_backs"] == 4 and st["spilled"] is False


def test_writes_run_behind_the_caller_and_nothing_drops_until_they_land(
    requires_mlx_any, tmp_path, monkeypatch
):
    s = _store(tmp_path)
    a = _entry(range(10), 1)
    s.admit(a)
    gate = __import__("threading").Event()
    real = _ss.os.pwrite

    def slow(fd, mv, off):
        gate.wait(10)
        return real(fd, mv, off)

    monkeypatch.setattr(_ss.os, "pwrite", slow)
    assert (
        s.spill() is None and a.resident and not s.spilled
    )  # handed to the writer, not waited for
    assert s.spill() is None and a.resident  # still in flight: nothing dropped
    assert s.stats()["spill_writes_pending"] == 1
    # an admission arriving before the writes land waits for them (bounded), then is served
    import threading

    threading.Timer(0.2, gate.set).start()
    s.restore()
    assert s.stats()["spill_writes_pending"] == 0 and a.spill_at is not None and a.resident
    assert s.lookup(list(range(12))) is a
    s.spiller.wait()
    monkeypatch.setattr(_ss.os, "pwrite", real)
    assert s.spill() == a.nbytes and not a.resident
    s.restore()
    want = _entry(range(10), 1)
    assert s.lookup(list(range(12))) is a and _bytes(a) == _bytes(want)


def test_an_entry_that_cannot_be_read_back_is_a_miss(requires_mlx_any, tmp_path):
    s = _store(tmp_path)
    a = _entry(range(10), 1)
    s.admit(a)
    _spill(s)
    a.spill_at = None  # the file lost it
    s.restore()
    assert s.lookup(list(range(12))) is None and len(s) == 0 and s.bytes == 0
    assert s.spiller.restore_failures == 1


def test_a_failed_write_keeps_its_entry_resident(requires_mlx_any, tmp_path, monkeypatch):
    s = _store(tmp_path)
    a, b = _entry(range(10), 1), _entry(range(20, 30), 2)
    want = _bytes(a)
    s.admit(a)
    calls = {"n": 0}
    real = _ss.os.pwrite

    def fail_first_entry(fd, mv, off):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real(fd, mv, off)

    monkeypatch.setattr(_ss.os, "pwrite", fail_first_entry)
    s.admit(b)
    freed = _spill(s)
    assert a.resident and a.spill_failed and a.spill_at is None and not b.resident
    assert freed == b.nbytes and s.spill_kept == a.nbytes and s.spiller.write_failures == 1
    s.restore()
    assert s.lookup(list(range(12))) is a and _bytes(a) == want
    # never retried: the next spill keeps it again rather than writing it on every turn (b was not
    # read back, so nothing new drops)
    assert (
        _spill(s) == 0 and a.resident and s.spill_kept == a.nbytes and s.spiller.write_failures == 1
    )


def test_space_is_reused_and_no_live_entry_ever_loses_its_bytes(requires_mlx_any, tmp_path):
    """Entries come and go through the cap across many spills; the ones that stay spilled (never
    looked up) keep their bytes, and evicted entries' space is reused, so the file stays bounded."""
    probe = _entry(range(10), 0, n=40)
    s = _store(tmp_path, cap=2 * probe.nbytes + 64)  # room for two entries
    want, keys = {}, []
    for i in range(12):
        n = 20 + 13 * (i % 3)  # sizes vary, so a gap does not always fit
        key = tuple(range(1000 * i, 1000 * i + 10))
        e = _entry(key, 100 + i, n=n)
        want[key] = _bytes(e)
        s.restore()
        s.admit(e)
        keys.append(key)
        _spill(s)  # older entries stay spilled across this round: they are never read back here
        for k in s.keys():  # every live entry's bytes, read from the file without a read-back
            regs = s._entries[k].spill_at
            assert regs is not None
            on_file = [s.spiller._read(o, n, sh, dt) for o, n, sh, dt in regs]
            assert [
                np.asarray(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()
                for a in on_file
            ] == want[k], (i, len(k))
    s.restore()
    live = s.keys()
    assert 1 <= len(live) <= 2 and all(k in want for k in live)
    for k in live:
        assert _bytes(s.lookup(list(k) + [1])) == want[k]
    assert s.spiller.file_bytes <= 3 * s.cap_bytes + 64 * _ss.ALIGN


def test_stores_sharing_one_file_never_answer_for_each_other(requires_mlx_any, tmp_path):
    """The warm-up's reset builds a new store on the same spill file: an entry with the same key in
    the new store is its own write, never the old store's bytes."""
    sp = StoreSpill(str(tmp_path), nocache=False)
    s1 = _store(tmp_path, spiller=sp)
    old = _entry(range(10), 1)
    s1.admit(old)
    _spill(s1)
    s2 = _store(tmp_path, spiller=sp)
    new = _entry(range(10), 2)  # same key, different bytes
    want = _bytes(new)
    s2.admit(new)
    _spill(s2)
    assert new.spill_at is not None and not new.resident
    s2.restore()
    assert _bytes(s2.lookup(list(range(12)))) == want


def test_every_transfer_is_whole_pages_and_close_removes_its_directory(requires_mlx_any, tmp_path):
    """The device cross-check counts `page_bytes_*`: each array moves as whole 16 KiB pages (its tail
    padded), on write and on read-back, and the payload comes back exact."""
    import os

    d = tmp_path / "jitengine-spill"
    sp = StoreSpill(str(d), nocache=False)
    s = _store(tmp_path, spiller=sp)
    a = _entry(range(10), 1, n=700)  # arrays past one page, none a whole number of pages
    want = _bytes(a)
    pages = sum(-(-len(b) // _ss.ALIGN) * _ss.ALIGN for b in want)
    assert any(len(b) > _ss.ALIGN and len(b) % _ss.ALIGN for b in want)
    s.admit(a)
    _spill(s)
    assert sp.bytes_written == sum(len(b) for b in want) and sp.page_bytes_written == pages
    assert os.path.getsize(f"/dev/fd/{sp.fd}") % _ss.ALIGN == 0
    s.restore()
    assert _bytes(s.lookup(list(range(12)))) == want
    assert sp.page_bytes_read == pages and sp.bytes_read == sp.bytes_written
    io = sp.io_stats()
    assert io["spill_page_bytes"] == 2 * pages and io["spill_dir"] == str(d)
    assert d.is_dir()
    sp.close()
    assert not d.exists()
    keep = tmp_path / "theirs"  # a directory the spill did not create stays
    keep.mkdir()
    StoreSpill(str(keep), nocache=False).close()
    assert keep.is_dir()
