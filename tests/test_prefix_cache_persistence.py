# SPDX-License-Identifier: Apache-2.0
"""
Reproduction tests for prefix-cache disk-persistence corruption bugs.

Failure modes documented here (see analysis 2026-05-03):

A. Stale ``index.json`` + freshly-overwritten ``entry_i.*`` files —
   if a server is killed mid-shutdown after rewriting some entry files
   but before ``index.json`` is rewritten, the next start loads using
   the stale ``num_tokens`` field. ``arr.fromfile(f, num_tokens_old)``
   silently truncates the new tokens.bin, producing an entry whose
   ``tokens_key`` length disagrees with ``cache.offset``. Subsequent
   fetches return that mismatched cache to the scheduler, which
   appends new tokens at the wrong position → garbage attention →
   token-id-0 collapse (``!!!!!`` in user output).

B. Orphan files from a previous save are not removed when the next
   save writes fewer entries. They sit on disk indefinitely; the next
   crash that interrupts ``save_to_disk`` mid-rewrite turns them into
   the inconsistency described in (A).

C. ``mx.save_safetensors`` is called directly on the target path
   (no ``.tmp`` + rename), so a SIGKILL during a single-entry write
   leaves a half-written safetensors. ``mx.load`` will usually raise
   on it (caught and dropped silently), but combined with (A) it can
   amplify the inconsistency.

D. ``mx.load`` is lazy — it parses the header and returns array
   handles without materializing data. A safetensors with a valid
   header but truncated body passes ``load_from_disk`` silently and
   is registered as a usable cache entry. The corruption only
   surfaces at the first attention call, often inside a worker thread
   where the RuntimeError can be swallowed.

These tests use real ``mlx_lm`` ``KVCache`` objects with very small
tensors (1×4×N×8 fp16) so they run fast (<1s each).
"""

from __future__ import annotations

import array
import json
import os

import pytest

mx = pytest.importorskip("mlx.core")
KVCache = pytest.importorskip("mlx_lm.models.cache").KVCache
save_prompt_cache = pytest.importorskip("mlx_lm.models.cache").save_prompt_cache

from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_kvcache(num_tokens: int, *, n_layers: int = 2, fill: float = 1.0) -> list:
    """Build a populated ``mlx_lm`` KVCache list with ``num_tokens`` positions.

    Tiny shape (1, 4, num_tokens, 8) fp16 keeps per-entry I/O <1 KB so
    these tests stay fast.
    """
    layers = []
    for layer_idx in range(n_layers):
        c = KVCache()
        keys = mx.full((1, 4, num_tokens, 8), fill + layer_idx, dtype=mx.float16)
        values = mx.full((1, 4, num_tokens, 8), -(fill + layer_idx), dtype=mx.float16)
        c.update_and_fetch(keys, values)
        layers.append(c)
    return layers


def fresh_cache() -> MemoryAwarePrefixCache:
    """Build a small in-memory cache for testing."""
    return MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100),
    )


def write_entry_files(
    cache_dir: str, entry_idx: int, tokens: list[int], kv_layers: list
) -> None:
    """Write a single (safetensors + tokens.bin) pair, mimicking save_to_disk."""
    save_prompt_cache(
        os.path.join(cache_dir, f"entry_{entry_idx}.safetensors"),
        kv_layers,
        metadata={"num_tokens": str(len(tokens))},
    )
    arr = array.array("i", tokens)
    with open(os.path.join(cache_dir, f"entry_{entry_idx}_tokens.bin"), "wb") as f:
        arr.tofile(f)


# --------------------------------------------------------------------------
# Sanity check — clean roundtrip works
# --------------------------------------------------------------------------


def test_clean_roundtrip_save_then_load(tmp_path):
    """Sanity: clean save → load → entry preserved."""
    cache = fresh_cache()
    tokens = list(range(11))
    cache.store(tokens, make_kvcache(num_tokens=11))
    assert cache.save_to_disk(str(tmp_path)) is True

    cache2 = fresh_cache()
    loaded = cache2.load_from_disk(str(tmp_path))
    assert loaded == 1

    entry = next(iter(cache2._entries.values()))
    assert entry.tokens == tuple(tokens)
    # Must hold for any well-formed entry: KV state is exactly as long
    # as the token sequence it claims to represent.
    assert entry.cache[0].offset == len(entry.tokens)


# --------------------------------------------------------------------------
# BUG A — stale index.json + overwritten entry files = poisoned tokens_key
# --------------------------------------------------------------------------


def test_stale_index_with_overwritten_entry_loads_without_error(tmp_path):
    """BUG A — load must reject (or normalize) entries whose tokens.bin
    size disagrees with the index.json claim. Such entries are the
    fingerprint of a previous interrupted save_to_disk.
    """
    # --- session 1: clean save with one entry of 11 tokens
    cache_v1 = fresh_cache()
    tokens_v1 = list(range(11))
    cache_v1.store(tokens_v1, make_kvcache(num_tokens=11))
    cache_v1.save_to_disk(str(tmp_path))

    # Sanity: index claims num_tokens=11
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["entries"][0]["num_tokens"] == 11

    # --- session 2: simulate kill DURING save_to_disk —
    # entry_0 files get rewritten with a longer 20-token payload,
    # but the process dies before index.json is rewritten.
    tokens_v2 = list(range(100, 120))  # 20 fresh tokens
    write_entry_files(str(tmp_path), 0, tokens_v2, make_kvcache(num_tokens=20))

    # index.json untouched — still says num_tokens=11
    index_after = json.loads((tmp_path / "index.json").read_text())
    assert index_after["entries"][0]["num_tokens"] == 11

    # --- session 3: load — the size-mismatch check must reject entry_0,
    # since tokens.bin is now 80 bytes (20 ints) but index.json claims 11.
    cache_v3 = fresh_cache()
    loaded = cache_v3.load_from_disk(str(tmp_path))
    assert loaded == 0, (
        "loader accepted an entry whose tokens.bin size disagrees with "
        "index.json's num_tokens — that's the BUG A poisoning vector"
    )
    assert len(cache_v3._entries) == 0

    # If any entry did slip through, its invariant must hold.
    for entry in cache_v3._entries.values():
        assert len(entry.tokens) == entry.cache[0].offset


def test_poisoned_entry_returns_misaligned_cache_via_fetch(tmp_path):
    """BUG A user-visible effect — a poisoned entry must NOT reach fetch().

    If the loader rejects it (correct), fetch returns None / no match.
    If a future regression lets one through, the returned cache.offset
    must at least equal the matched-prefix length so the scheduler
    appends tokens at the right position.
    """
    # Set up the same poisoned state as the previous test.
    cache_v1 = fresh_cache()
    cache_v1.store(list(range(11)), make_kvcache(num_tokens=11))
    cache_v1.save_to_disk(str(tmp_path))

    # Overwrite entry_0 with 20-token content; leave index.json stale.
    tokens_v2 = list(range(100, 120))
    write_entry_files(str(tmp_path), 0, tokens_v2, make_kvcache(num_tokens=20))

    cache_v3 = fresh_cache()
    cache_v3.load_from_disk(str(tmp_path))

    # User sends a prompt that begins with the (would-be-truncated) cached prefix.
    prompt = list(range(100, 111)) + [777, 888, 999]
    kv, remaining = cache_v3.fetch(prompt)

    if kv is None:
        # Correct path: poisoned entry was rejected at load_from_disk;
        # fetch sees an empty cache and returns a clean miss.
        assert remaining == prompt
        return

    # If for some reason an entry slipped through, the returned cache
    # offset must match the matched-prefix length.
    matched_len = len(prompt) - len(remaining)
    returned_offset = kv[0].offset
    assert returned_offset == matched_len, (
        f"fetch returned cache with offset={returned_offset} for "
        f"matched_len={matched_len} prefix. Scheduler would write next "
        f"token at the wrong position."
    )


# --------------------------------------------------------------------------
# BUG B — orphan files from a previous save are not cleaned up
# --------------------------------------------------------------------------


def test_save_to_disk_removes_orphans_from_previous_save(tmp_path):
    """BUG B — directory-rename swap must leave no orphan entry files."""
    # --- session 1: 5 entries
    cache_v1 = fresh_cache()
    for i in range(5):
        cache_v1.store(
            list(range(i * 100, i * 100 + 11)), make_kvcache(num_tokens=11, fill=i + 1)
        )
    cache_v1.save_to_disk(str(tmp_path))

    # --- session 2: only 2 entries (fresh cache, simulates eviction)
    cache_v2 = fresh_cache()
    for i in range(2):
        cache_v2.store(
            list(range(500 + i * 100, 500 + i * 100 + 11)),
            make_kvcache(num_tokens=11, fill=i + 10),
        )
    cache_v2.save_to_disk(str(tmp_path))

    # New index.json reflects only 2 entries
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["num_entries"] == 2

    # CORRECT BEHAVIOR (xfail): orphan entry_2..entry_4 from session 1
    # must be removed so a future crash mid-save can't resurrect them.
    for i in range(2, 5):
        sf = tmp_path / f"entry_{i}.safetensors"
        tk = tmp_path / f"entry_{i}_tokens.bin"
        assert not sf.exists(), (
            f"orphan {sf.name} from previous save was not cleaned up — "
            f"this is the precondition that turns a half-written next save "
            f"into BUG A."
        )
        assert not tk.exists(), f"orphan {tk.name} not cleaned up"


# --------------------------------------------------------------------------
# Characterization tests — document current behavior (no xfail)
# --------------------------------------------------------------------------


def test_severely_truncated_safetensors_is_silently_skipped(tmp_path):
    """Document current behavior: a header-corrupt .safetensors is dropped silently.

    This is *acceptable* on its own (the entry just won't be used), but
    no structured signal is propagated upward — operators have no way
    to notice gradual cache decay. Once the diagnostic-counter fix
    (P3 in the analysis) lands, this test should also assert that
    ``loaded`` returns a (loaded, skipped) tuple or that a structured
    warning was emitted.
    """
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(tmp_path))

    # Truncate aggressively (16 bytes — far short of safetensors header)
    sf = tmp_path / "entry_0.safetensors"
    sf.write_bytes(sf.read_bytes()[:16])

    # Load: must not raise, just skip
    cache2 = fresh_cache()
    loaded = cache2.load_from_disk(str(tmp_path))
    assert loaded == 0
    assert len(cache2._entries) == 0


def test_body_truncated_safetensors_should_fail_eagerly_at_load(tmp_path):
    """BUG D — load_from_disk must reject a body-truncated safetensors
    even though ``mx.load`` will lazily mmap it without complaint.
    """
    import struct

    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(tmp_path))

    sf = tmp_path / "entry_0.safetensors"
    full = sf.read_bytes()

    # Compute the maximum data offset declared by the header so we can
    # truncate strictly inside the body region — guards against future
    # changes to padding/alignment in save_prompt_cache.
    header_len = struct.unpack("<Q", full[:8])[0]
    header = json.loads(full[8 : 8 + header_len])
    max_end = max(
        meta["data_offsets"][1]
        for name, meta in header.items()
        if name != "__metadata__"
    )
    declared_total = 8 + header_len + max_end
    cut_to = declared_total - 100
    assert cut_to > 8 + header_len, (
        "test setup: cut would land in the header region, not the body — "
        "use a larger entry"
    )
    sf.write_bytes(full[:cut_to])

    cache2 = fresh_cache()
    loaded = cache2.load_from_disk(str(tmp_path))
    assert loaded == 0, (
        "Body-truncated safetensors was loaded as a usable cache entry. "
        "It will blow up later at attention time with a RuntimeError, "
        "likely inside a worker thread."
    )


# --------------------------------------------------------------------------
# Crash-recovery for interrupted save_to_disk swap
# --------------------------------------------------------------------------


def test_load_recovers_from_swap_interrupted_after_first_rename(tmp_path):
    """If the process died after ``cache_dir → .old`` but before
    ``.new → cache_dir``, load_from_disk must promote ``.new`` because
    it holds the freshly-committed snapshot.
    """
    cache_dir = tmp_path / "snap"
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"

    # Snapshot 1 → ends up at .old (simulates the first rename of the swap)
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))
    cache_dir.rename(old_dir)

    # Snapshot 2 built in a side dir, then placed at .new (simulates the
    # staging dir of an interrupted save — done writing, swap not yet
    # finished). Using a side dir avoids triggering the next save's
    # pre-clean of .old.
    side_dir = tmp_path / "side"
    c2 = fresh_cache()
    c2.store(list(range(50, 65)), make_kvcache(num_tokens=15, fill=2.0))
    c2.save_to_disk(str(side_dir))
    side_dir.rename(new_dir)

    assert not cache_dir.exists()
    assert new_dir.exists()
    assert old_dir.exists()

    # Load: should promote .new to cache_dir, drop .old
    c3 = fresh_cache()
    loaded = c3.load_from_disk(str(cache_dir))
    assert loaded == 1
    assert cache_dir.exists()
    assert not new_dir.exists()
    assert not old_dir.exists()
    entry = next(iter(c3._entries.values()))
    assert entry.tokens == tuple(range(50, 65))


def test_load_recovers_from_swap_interrupted_with_only_old(tmp_path):
    """If only ``.old`` survives (e.g. ``.new`` was never finalized),
    load_from_disk must restore ``.old`` to ``cache_dir``.
    """
    cache_dir = tmp_path / "snap"
    c1 = fresh_cache()
    c1.store(list(range(7)), make_kvcache(num_tokens=7))
    c1.save_to_disk(str(cache_dir))

    # Simulate crash mid-swap with no .new survivor
    cache_dir.rename(tmp_path / "snap.old")
    assert not cache_dir.exists()

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    assert loaded == 1
    assert cache_dir.exists()
    entry = next(iter(c2._entries.values()))
    assert entry.tokens == tuple(range(7))


def test_load_cleans_orphan_staging_dirs(tmp_path):
    """If ``cache_dir`` exists alongside leftover ``.new`` / ``.old``
    staging dirs, load_from_disk must wipe the orphans so the next
    save starts from a clean slate.
    """
    cache_dir = tmp_path / "snap"
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))

    # Sprinkle leftover staging dirs
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"
    new_dir.mkdir()
    (new_dir / "leftover.txt").write_text("orphan")
    old_dir.mkdir()
    (old_dir / "leftover.txt").write_text("orphan")

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    assert loaded == 1
    assert not new_dir.exists()
    assert not old_dir.exists()


def test_partial_new_index_json_is_not_promoted(tmp_path):
    """If .new/index.json exists but is corrupt JSON (e.g. crash mid
    json.dump), recovery must NOT promote .new — fall back to .old or
    leave cache_dir absent rather than handing the partial snapshot
    to subsequent json.load.
    """
    cache_dir = tmp_path / "snap"
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"

    # Build a valid snapshot at .old (the previous committed state)
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))
    cache_dir.rename(old_dir)

    # Hand-craft a .new with a *partial* index.json (simulates crash
    # in the middle of json.dump).
    new_dir.mkdir()
    (new_dir / "index.json").write_text('{"versi')

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    # Should fall through to .old, recovering the previous snapshot
    assert loaded == 1
    entry = next(iter(c2._entries.values()))
    assert entry.tokens == tuple(range(11))


def test_save_handles_trailing_slash_in_cache_dir(tmp_path):
    """A user-supplied cache_dir with a trailing separator must still
    swap atomically. Without the rstrip in save_to_disk, ``cache_dir +
    '.new'`` would become a *child* of cache_dir rather than a sibling,
    silently breaking the swap.
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(cache_dir) + "/")

    # The committed snapshot lives at cache_dir, NOT cache_dir/.new
    assert cache_dir.exists()
    assert (cache_dir / "index.json").exists()
    assert not (cache_dir / ".new").exists()
    assert not (tmp_path / "snap/.new").exists()

    # Round-trips with trailing slash on load too.
    c2 = fresh_cache()
    assert c2.load_from_disk(str(cache_dir) + "/") == 1


def test_load_into_non_empty_cache_skips_duplicates(tmp_path):
    """If load_from_disk is called on a cache that already contains some
    keys (e.g. populated by warmup before lifespan calls load), entries
    whose tokens_key matches an in-memory entry must be skipped — not
    re-inserted. Otherwise bisect.insort produces duplicate keys in
    _sorted_keys and _current_memory double-counts.
    """
    cache_dir = tmp_path / "snap"
    # Persist two entries to disk: one duplicates a future in-memory
    # entry; one is fresh.
    persisted = fresh_cache()
    persisted.store(list(range(11)), make_kvcache(num_tokens=11))
    persisted.store(list(range(50, 61)), make_kvcache(num_tokens=11, fill=2.0))
    persisted.save_to_disk(str(cache_dir))

    # Simulated warmup state: the [0..10] entry is already in memory.
    runtime = fresh_cache()
    runtime.store(list(range(11)), make_kvcache(num_tokens=11))
    warmup_mem = runtime._current_memory
    warmup_keys = list(runtime._sorted_keys)

    loaded = runtime.load_from_disk(str(cache_dir))
    assert loaded == 1, "exactly one fresh entry should have been loaded"
    # The duplicate did not double-insert
    assert runtime._sorted_keys.count(tuple(range(11))) == 1
    assert tuple(range(50, 61)) in runtime._sorted_keys
    # Memory grew by exactly the new entry's footprint
    new_entry_mem = runtime._current_memory - warmup_mem
    assert new_entry_mem > 0
    # Pre-existing entry untouched in keys list ordering wrt itself
    assert warmup_keys[0] in runtime._sorted_keys


def test_recovery_rejects_new_with_index_but_no_entry_files(tmp_path):
    """If ``.new/index.json`` references entries but the entry files are
    missing on disk (manual deletion, fs corruption, partial restore),
    recovery must NOT promote ``.new``. Doing so would discard ``.old``
    in favor of an empty snapshot — net data loss.
    """
    cache_dir = tmp_path / "snap"
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"

    # Build a real, complete snapshot in .old
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))
    cache_dir.rename(old_dir)

    # Hand-craft a .new with valid-looking index.json but NO entry files
    new_dir.mkdir()
    (new_dir / "index.json").write_text(
        json.dumps(
            {
                "version": 2,
                "num_entries": 1,
                "total_memory_bytes": 12345,
                "entries": [{"index": 0, "num_tokens": 11, "memory_bytes": 12345}],
            }
        )
    )

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    # Recovery should fall through to .old, not silently lose the snapshot
    assert loaded == 1, "recovery promoted an empty .new and lost .old"
    entry = next(iter(c2._entries.values()))
    assert entry.tokens == tuple(range(11))


def test_load_dedup_check_runs_before_safetensors_load(tmp_path, monkeypatch):
    """Performance + memory: a tokens_key that's already in the in-memory
    cache must skip ``load_prompt_cache`` entirely. Otherwise every
    duplicate entry mmaps its safetensors only to discard it — wastes
    file descriptors, memory, and time.
    """
    cache_dir = tmp_path / "snap"
    persisted = fresh_cache()
    persisted.store(list(range(11)), make_kvcache(num_tokens=11))
    persisted.save_to_disk(str(cache_dir))

    # Spy on load_prompt_cache to count calls
    import mlx_lm.models.cache as mlx_cache_mod

    real_load = mlx_cache_mod.load_prompt_cache
    call_count = {"n": 0}

    def spy(path):
        call_count["n"] += 1
        return real_load(path)

    monkeypatch.setattr(mlx_cache_mod, "load_prompt_cache", spy)
    monkeypatch.setattr("vllm_mlx.memory_cache.load_prompt_cache", spy, raising=False)

    runtime = fresh_cache()
    runtime.store(list(range(11)), make_kvcache(num_tokens=11))
    loaded = runtime.load_from_disk(str(cache_dir))

    assert loaded == 0, "the only persisted entry was a duplicate"
    assert call_count["n"] == 0, (
        f"load_prompt_cache should not be called for duplicates "
        f"(was called {call_count['n']} time(s))"
    )


def test_save_routes_writes_through_staging_dir(tmp_path, monkeypatch):
    """Atomicity invariant: save_safetensors must be called with a path
    inside a sibling ``<cache_dir>.new`` staging directory, not directly
    inside ``cache_dir``. The directory-rename swap is what makes the
    snapshot all-or-nothing.
    """
    seen_paths: list[str] = []

    real_save = mx.save_safetensors

    def spy(path, *args, **kwargs):
        seen_paths.append(path)
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr("mlx.core.save_safetensors", spy)

    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(cache_dir))

    assert seen_paths, "save_safetensors was never called"
    expected_staging = str(cache_dir) + ".new"
    for p in seen_paths:
        assert p.startswith(expected_staging + os.sep), (
            f"save_safetensors called with {p!r}, expected to be inside "
            f"{expected_staging!r}. Direct writes into the committed "
            f"cache_dir defeat the atomic-snapshot guarantee."
        )

    # After save returns, the staging dir is gone (renamed into place).
    assert not (tmp_path / "snap.new").exists()
    assert (cache_dir / "index.json").exists()
    assert (cache_dir / "entry_0.safetensors").exists()
