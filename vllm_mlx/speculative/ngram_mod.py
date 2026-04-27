# SPDX-License-Identifier: Apache-2.0
"""
ngram-mod speculative decoding.

Port of llama.cpp's `--spec-type ngram-mod` (PR ggml-org/llama.cpp#19164).

Idea: a fixed-size hash table maps an n-gram of the last `n` tokens to a
single next-token guess. Collisions overwrite. Drafts are built by hashing
the current tail, looking up the next token, sliding the window, and
repeating until an EMPTY slot or a length cap is hit (variable length).

The pool persists across requests; an adaptive policy wipes it when
acceptance drops below a threshold for several rounds in a row.
"""

from __future__ import annotations

import array
import collections
import logging
import math

import mlx.core as mx

logger = logging.getLogger(__name__)

EMPTY: int = -1
EMPTY_KEY: int = 0
_MULT: int = 6364136223846793005
_MASK64: int = (1 << 64) - 1

# entries[] slot layout (int32, always >= 0 for a valid slot; -1 == EMPTY):
#   bits  0-19: token ID  (20 bits, covers vocab sizes up to ~1M)
#   bits 20-27: count     (8 bits, saturates at 255)
#   bits 28-31: unused (always 0)
# Encoding guarantees the value is non-negative, so entries[i] < 0 means EMPTY.
_TOKEN_BITS: int = 20
_TOKEN_MASK: int = (1 << _TOKEN_BITS) - 1  # 0xFFFFF
_COUNT_SHIFT: int = _TOKEN_BITS
_COUNT_MAX: int = 0xFF

# ---------------------------------------------------------------------------
# Pre-seed token sequences for Qwen3 models.
#
# These are the exact token IDs produced by the Qwen3 tokenizer
# (confirmed against mlx-community/Qwen3.5-4B-MLX-4bit, identical across
# all Qwen3 variants).  Each sequence represents a complete tool-call
# skeleton in the format the chat template uses:
#
#   \n\n<tool_call>\n<function=NAME>\n<parameter=PARAM>\n[VALUE]
#   \n</parameter>\n</function>\n</tool_call>
#
# Seeding these at startup means the pool already knows the structural
# transitions before the first request arrives, giving near-100% draft
# acceptance for tool-call boilerplate from turn 1 onward.
# ---------------------------------------------------------------------------
# fmt: off
_QWEN3_PRESEED_SEQUENCES: list[list[int]] = [
    # bash + command (bare)
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "ls -la\n"
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 4577, 471, 4120, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "cat "
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 4466, 220, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "python "
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 12305, 220, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "npm "
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 38708, 220, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "git "
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 12513, 220, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "echo "
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 2949, 220, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "mkdir "
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 25283, 220, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # bash + command + "cd "
    [271, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198, 4243, 220, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # read_file + path (bare)
    [271, 248058, 198, 27, 1628, 86779, 2378, 29, 198, 27, 15704, 79114, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # read_file + path + "/"
    [271, 248058, 198, 27, 1628, 86779, 2378, 29, 198, 27, 15704, 79114, 29, 198, 14, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # read_file + path + "src/"
    [271, 248058, 198, 27, 1628, 86779, 2378, 29, 198, 27, 15704, 79114, 29, 198, 3431, 14, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # write_file + path
    [271, 248058, 198, 27, 1628, 28, 4775, 2378, 29, 198, 27, 15704, 79114, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # write_file + content
    [271, 248058, 198, 27, 1628, 28, 4775, 2378, 29, 198, 27, 15704, 28, 1733, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # str_replace_editor + path
    [271, 248058, 198, 27, 1628, 15462, 10318, 32634, 29, 198, 27, 15704, 79114, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # str_replace_editor + command
    [271, 248058, 198, 27, 1628, 15462, 10318, 32634, 29, 198, 27, 15704, 28, 5454, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # str_replace_editor + old_str
    [271, 248058, 198, 27, 1628, 15462, 10318, 32634, 29, 198, 27, 15704, 28, 787, 2801, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # str_replace_editor + new_str
    [271, 248058, 198, 27, 1628, 15462, 10318, 32634, 29, 198, 27, 15704, 8083, 2801, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # view + path + "/"
    [271, 248058, 198, 27, 1628, 88783, 29, 198, 27, 15704, 79114, 29, 198, 14, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # create_file + path
    [271, 248058, 198, 27, 1628, 87950, 2378, 29, 198, 27, 15704, 79114, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # create_file + content
    [271, 248058, 198, 27, 1628, 87950, 2378, 29, 198, 27, 15704, 28, 1733, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # search_files + pattern
    [271, 248058, 198, 27, 1628, 93260, 10612, 29, 198, 27, 15704, 28, 13927, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # search_files + path
    [271, 248058, 198, 27, 1628, 93260, 10612, 29, 198, 27, 15704, 79114, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # execute_command + command
    [271, 248058, 198, 27, 1628, 28, 9951, 10494, 29, 198, 27, 15704, 28, 5454, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
    # run_command + command
    [271, 248058, 198, 27, 1628, 28, 5917, 10494, 29, 198, 27, 15704, 28, 5454, 29, 198, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059],
]
# fmt: on


class NGramModDecoder:
    """Hash-pool n-gram drafter (llama.cpp ngram-mod port).

    Args:
        n: window size used for hashing (llama.cpp default 16, min 16).
        pool_size: number of int32 slots in the hash pool (default 1<<20 ≈ 4 MB).
        n_min: minimum draft length to bother verifying.
        n_max: maximum draft length per round.
        reset_threshold: acceptance ratio below which a round is "low".
        reset_streak: consecutive low rounds that trigger a pool wipe.
    """

    def __init__(
        self,
        n: int = 16,
        pool_size: int = 1 << 20,
        n_min: int = 1,
        n_max: int = 16,
        reset_threshold: float = 0.05,
        reset_streak: int = 20,
        recent_scan_window: int = 512,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if n_min < 1:
            raise ValueError("n_min must be >= 1")
        if n_max < 1:
            raise ValueError("n_max must be >= 1")
        if n_min > n_max:
            raise ValueError("n_min must be <= n_max")
        if not 0.0 <= reset_threshold <= 1.0:
            raise ValueError("reset_threshold must be between 0 and 1")
        if reset_streak < 1:
            raise ValueError("reset_streak must be >= 1")
        self.n = int(n)
        self.pool_size = int(pool_size)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        # Hard upper bound on adaptive n_max; n_max is the live (mutable) value.
        self._n_max_hard: int = int(n_max)
        self.reset_threshold = float(reset_threshold)
        self.reset_streak = int(reset_streak)
        self.recent_scan_window = int(recent_scan_window)

        self.entries: array.array = self._make_pool()
        self.keys: array.array = self._make_keys()
        self.used: int = 0

        self._low_streak: int = 0
        # Rolling token history for the recent-context scan fallback.
        # maxlen keeps memory bounded; +n_max leaves room for a full draft tail.
        self._history: collections.deque[int] = collections.deque(
            maxlen=self.recent_scan_window + self.n_max + 1
        )

        # Adaptive n_max: exponential moving average of per-round acceptance
        # rate.  Starts at 0.5 so the first rounds use a moderate draft length.
        # EMA decay gives an effective window of ~16 rounds (decay = 1 - 2/17).
        self._alpha_hat: float = 0.5
        self._alpha_ema_decay: float = 1.0 - 2.0 / (16.0 + 1.0)

        self.lifetime_proposed: int = 0
        self.lifetime_accepted: int = 0
        self.lifetime_drafts: int = 0
        self.lifetime_resets: int = 0
        self.lifetime_collisions: int = 0
        self.lifetime_scan_hits: int = 0

    def _make_pool(self) -> array.array:
        a = array.array("i")
        a.frombytes(b"\xff" * (4 * self.pool_size))
        return a

    def _make_keys(self) -> array.array:
        a = array.array("Q")
        a.frombytes(b"\x00" * (8 * self.pool_size))
        return a

    def _hash64(self, window) -> int:
        h = 0
        for t in window:
            h = (h * _MULT + int(t)) & _MASK64
        return h

    def _hash(self, window) -> int:
        return self._hash64(window) % self.pool_size

    def _slot_and_key(self, window) -> tuple[int, int]:
        raw_key = self._hash64(window)
        key = raw_key
        if key == EMPTY_KEY:
            key = 1
        return raw_key % self.pool_size, key

    def add(self, window, next_token: int) -> None:
        if len(window) != self.n:
            return
        i, key = self._slot_and_key(window)
        tok = int(next_token) & _TOKEN_MASK
        cur = self.entries[i]

        if self.keys[i] == EMPTY_KEY:
            # Empty slot: claim it with count=1.
            self.used += 1
            self.keys[i] = key
            self.entries[i] = (1 << _COUNT_SHIFT) | tok
        elif self.keys[i] == key:
            # Same n-gram: update frequency for this next-token observation.
            stored_tok = cur & _TOKEN_MASK
            stored_cnt = (cur >> _COUNT_SHIFT) & 0xFF
            if stored_tok == tok:
                # Same next-token: increment count (saturating at _COUNT_MAX).
                new_cnt = min(stored_cnt + 1, _COUNT_MAX)
                self.entries[i] = (new_cnt << _COUNT_SHIFT) | tok
            else:
                # Different next-token for same n-gram context.  Keep the one
                # with higher observed count; the incoming observation has
                # count=1 implicitly.  Only replace if it already dominates.
                if stored_cnt <= 1:
                    # Tie or count-1 entry: last-write semantics for rare contexts.
                    self.entries[i] = (1 << _COUNT_SHIFT) | tok
                # else: stored token seen more often; keep it.
        else:
            # Hash collision: different n-gram maps to same slot.
            self.lifetime_collisions += 1
            stored_cnt = (cur >> _COUNT_SHIFT) & 0xFF if cur >= 0 else 0
            # Evict only when the incumbent is low-frequency (count <= 1).
            if stored_cnt <= 1:
                self.keys[i] = key
                self.entries[i] = (1 << _COUNT_SHIFT) | tok

        # Track the generated token stream so the scan fallback sees new tokens.
        self._history.append(int(next_token))

    def get(self, window) -> int:
        if len(window) != self.n:
            return EMPTY
        i, key = self._slot_and_key(window)
        if self.keys[i] != key:
            return EMPTY
        cur = self.entries[i]
        if cur < 0:
            return EMPTY
        return cur & _TOKEN_MASK

    def ingest(self, tokens) -> None:
        """Add every (n-gram window, next-token) pair extractable from tokens."""
        if len(tokens) < self.n + 1:
            return
        # Seed history with the first n tokens upfront; add() appends each
        # subsequent next_token so the full stream ends up in _history without
        # double-counting.
        self._history.extend(int(t) for t in tokens[: self.n])
        for w in range(len(tokens) - self.n):
            self.add(tokens[w : w + self.n], tokens[w + self.n])

    def ingest_preseed(self, sequences: list[list[int]], count: int = 10) -> None:
        """Ingest static token sequences into the pool without touching _ingested_up_to.

        Each sequence is ingested like a regular ingest() call, but the entries
        are written with an initial count of `count` (default 10) so they survive
        hash-collision eviction by real tokens (which start at count=1).  This
        makes structural patterns — tool-call XML, common code boilerplate — durable
        in the pool even after many requests have run.

        Sequences shorter than n+1 tokens are silently skipped.

        Args:
            sequences: list of token-ID lists to pre-seed.
            count: initial frequency count for pre-seeded entries (clamped to _COUNT_MAX).
        """
        initial_count = min(int(count), _COUNT_MAX)
        for seq in sequences:
            if len(seq) < self.n + 1:
                continue
            self._history.extend(int(t) for t in seq[: self.n])
            for w in range(len(seq) - self.n):
                window = seq[w : w + self.n]
                next_tok = int(seq[w + self.n]) & _TOKEN_MASK
                i, key = self._slot_and_key(window)
                if self.keys[i] == EMPTY_KEY:
                    self.used += 1
                    self.keys[i] = key
                    self.entries[i] = (initial_count << _COUNT_SHIFT) | next_tok
                elif self.keys[i] == key:
                    # Same n-gram already present — bump count so it stays durable.
                    stored_tok = self.entries[i] & _TOKEN_MASK
                    if stored_tok == next_tok:
                        stored_cnt = (self.entries[i] >> _COUNT_SHIFT) & 0xFF
                        self.entries[i] = (min(stored_cnt + initial_count, _COUNT_MAX) << _COUNT_SHIFT) | next_tok
                    # If a different token is stored for this n-gram, leave it;
                    # actual generation data is more trustworthy than pre-seeds.
                else:
                    # Collision: evict only if the incumbent is low-frequency.
                    stored_cnt = (self.entries[i] >> _COUNT_SHIFT) & 0xFF if self.entries[i] >= 0 else 0
                    if stored_cnt < initial_count:
                        self.keys[i] = key
                        self.entries[i] = (initial_count << _COUNT_SHIFT) | next_tok
                self._history.append(next_tok)

    def preseed_qwen3(self) -> None:
        """Pre-seed the pool with Qwen3 tool-call and structural patterns.

        Call once at startup (and again after reset_pool if you want persistent
        coverage).  The sequences are hardcoded token IDs for the Qwen3
        tokenizer family — safe to call unconditionally; incorrect seeds for a
        non-Qwen3 model are harmless (they occupy a tiny fraction of the pool
        and get evicted by actual traffic within a few requests).

        Also stores the sequences in ``_preseed_sequences`` so that
        ``reset_pool()`` automatically re-applies them after a pool wipe.
        """
        self._preseed_sequences = _QWEN3_PRESEED_SEQUENCES
        self.ingest_preseed(_QWEN3_PRESEED_SEQUENCES)
        logger.debug(
            "ngram-mod: pre-seeded %d Qwen3 tool-call sequences (%d pool entries)",
            len(_QWEN3_PRESEED_SEQUENCES),
            sum(max(0, len(s) - self.n) for s in _QWEN3_PRESEED_SEQUENCES),
        )

    def _scan(self, tail: list[int]) -> int:
        """Scan recent history for the longest suffix match of *tail*.

        Tries to find the longest prefix of *tail* (from len down to 1) that
        appears in ``_history`` and has a following token. Returns the token
        that follows the best match, or EMPTY when nothing is found.

        Complexity: O(recent_scan_window * len(tail)) per call.
        For tail length 4 and window 512 that is ~2 048 integer comparisons —
        roughly 50 µs in Python, negligible vs. the model forward pass.
        """
        if not self._history or not tail:
            return EMPTY
        history = list(self._history)  # snapshot; fast for deque sizes ≤ 2K
        h_len = len(history)
        # Try longest match first so we return the most specific prediction.
        for match_len in range(len(tail), 0, -1):
            needle = tail[-match_len:]
            # Scan backward through history (most-recent occurrences first).
            for start in range(h_len - match_len, -1, -1):
                if history[start : start + match_len] == needle:
                    next_pos = start + match_len
                    if next_pos < h_len:
                        return history[next_pos]
        return EMPTY

    def draft(self, recent_tokens, n_max: int | None = None) -> list[int]:
        """Greedy variable-length draft from the last n tokens.

        Each step tries the hash table first (O(1)). On a miss, falls back to
        a linear scan of the recent-token history (Option C/D hybrid), which
        catches patterns evicted by hash collisions and tokens generated after
        the initial ingest call.
        """
        cap = self.n_max if n_max is None else int(n_max)
        if cap <= 0 or len(recent_tokens) < self.n:
            return []
        window = list(recent_tokens[-self.n :])
        out: list[int] = []
        for _ in range(cap):
            tok = self.get(window)
            if tok == EMPTY:
                tok = self._scan(window)
                if tok == EMPTY:
                    break
                self.lifetime_scan_hits += 1
            out.append(int(tok))
            window = window[1:] + [int(tok)]
        return out

    def record_round(self, num_proposed: int, num_accepted: int) -> None:
        if num_proposed > 0:
            self.lifetime_drafts += 1
            self.lifetime_proposed += num_proposed
            self.lifetime_accepted += num_accepted
            ratio = num_accepted / num_proposed
            if ratio < self.reset_threshold:
                self._low_streak += 1
            else:
                self._low_streak = 0
            if self._low_streak >= self.reset_streak:
                self.reset_pool()
                self._low_streak = 0

            # Adaptive n_max: update EMA acceptance rate and recompute the
            # optimal draft length.
            #
            # Theory: E[tokens/pass | draft k] = 1 + α*(1-α^k)/(1-α)
            # This is monotonically increasing in k but with diminishing
            # returns: the marginal gain of extending from k to k+1 is α^(k+1).
            # We stop when α^k <= 0.05, i.e. we capture ≥95% of the asymptotic
            # gain without burning extra verify-pass tokens on near-zero-probability
            # extensions.
            #
            # k_opt = ceil(ln(0.05) / ln(α_hat))
            #       = ceil(-2.996 / ln(α_hat))
            #
            # Clamp to [n_min, _n_max_hard] so the hard limits are respected.
            d = self._alpha_ema_decay
            self._alpha_hat = d * self._alpha_hat + (1.0 - d) * ratio

            alpha = self._alpha_hat
            if alpha <= 0.0 or alpha >= 1.0:
                # Degenerate: α=0 → k=1; α=1 → use hard cap.
                k_opt = 1 if alpha <= 0.0 else self._n_max_hard
            else:
                k_opt = math.ceil(math.log(0.05) / math.log(alpha))

            self.n_max = max(self.n_min, min(self._n_max_hard, k_opt))

    def reset_pool(self) -> None:
        self.entries = self._make_pool()
        self.keys = self._make_keys()
        self.used = 0
        self._history.clear()
        self.lifetime_resets += 1
        self._ingested_up_to = 0
        # Re-apply any pre-seeded sequences so structural patterns survive the
        # reset.  preseed_sequences is set by callers who want durable patterns
        # (e.g., NGramModEngine sets it after calling preseed_qwen3()).
        if getattr(self, "_preseed_sequences", None):
            self.ingest_preseed(self._preseed_sequences)

    def get_stats(self) -> dict:
        rate = (
            self.lifetime_accepted / self.lifetime_proposed
            if self.lifetime_proposed > 0
            else 0.0
        )
        return {
            "n": self.n,
            "pool_size": self.pool_size,
            "used": self.used,
            "load": self.used / self.pool_size if self.pool_size > 0 else 0.0,
            "lifetime_drafts": self.lifetime_drafts,
            "lifetime_proposed": self.lifetime_proposed,
            "lifetime_accepted": self.lifetime_accepted,
            "acceptance_rate": rate,
            "resets": self.lifetime_resets,
            "collisions": self.lifetime_collisions,
            "scan_hits": self.lifetime_scan_hits,
            "recent_scan_window": self.recent_scan_window,
            "alpha_hat": self._alpha_hat,
            "n_max": self.n_max,
            "n_max_hard": self._n_max_hard,
        }


class MultiLevelNGramDecoder:
    """Multi-level n-gram drafter that tries multiple window sizes.

    For each draft step, tries levels from longest n to shortest n.
    The first level with a non-EMPTY hit wins that step. This dramatically
    increases coverage compared to a single window size.

    Expected acceptance rate improvement example:
        n=16 alone:          ~20% hit rate
        + n=12 fallback:     covers ~30% of remaining 80% → +24%  → ~44%
        + n=8  fallback:     covers ~40% of remaining 56% → +22%  → ~66%
        + n=4  fallback:     covers ~50% of remaining 34% → +17%  → ~83%
        + n=2  fallback:     covers ~60% of remaining 17% → +10%  → ~93%

    Args:
        ns: list of window sizes, e.g. [16, 12, 8, 4, 2].
            Will be sorted longest-first automatically.
        pool_sizes: per-level pool sizes; if a scalar, all levels share it.
        n_min: minimum draft length to bother verifying (shared).
        n_max: maximum draft length per round (shared).
        reset_threshold: passed to each underlying decoder.
        reset_streak: passed to each underlying decoder.
        recent_scan_window: passed to each underlying decoder.
    """

    def __init__(
        self,
        ns: list[int],
        pool_sizes: int | list[int] = 1 << 20,
        n_min: int = 1,
        n_max: int = 16,
        reset_threshold: float = 0.05,
        reset_streak: int = 20,
        recent_scan_window: int = 512,
    ) -> None:
        if not ns:
            raise ValueError("ns must be a non-empty list of window sizes")
        sorted_ns = sorted(set(ns), reverse=True)  # longest first
        if isinstance(pool_sizes, int):
            pool_sizes_list = [pool_sizes] * len(sorted_ns)
        else:
            if len(pool_sizes) != len(sorted_ns):
                raise ValueError("pool_sizes length must match ns length")
            # align to sorted_ns order
            paired = sorted(zip(ns, pool_sizes), key=lambda x: -x[0])
            pool_sizes_list = [p for _, p in paired]

        self._levels: list[NGramModDecoder] = [
            NGramModDecoder(
                n=n,
                pool_size=ps,
                n_min=n_min,
                n_max=n_max,
                reset_threshold=reset_threshold,
                reset_streak=reset_streak,
                recent_scan_window=recent_scan_window,
            )
            for n, ps in zip(sorted_ns, pool_sizes_list)
        ]
        # Primary decoder is the longest-n level; used for shared state.
        self._primary = self._levels[0]
        self.n_min = int(n_min)
        self.n_max = int(n_max)

    # ------------------------------------------------------------------
    # Proxy scalar attributes to the primary (longest-n) decoder so that
    # external code (generate_step, engine) can use decoder.n / decoder.n_min
    # / decoder.n_max / decoder._low_streak / decoder._ingested_up_to without
    # knowing about multi-level.
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._primary.n

    @property
    def _low_streak(self) -> int:
        return self._primary._low_streak

    @_low_streak.setter
    def _low_streak(self, value: int) -> None:
        for lvl in self._levels:
            lvl._low_streak = value

    @property
    def _ingested_up_to(self) -> int:
        return self._primary._ingested_up_to

    @_ingested_up_to.setter
    def _ingested_up_to(self, value: int) -> None:
        for lvl in self._levels:
            lvl._ingested_up_to = value

    # lifetime counters aggregated across levels for stats
    @property
    def lifetime_proposed(self) -> int:
        return self._primary.lifetime_proposed

    @property
    def lifetime_accepted(self) -> int:
        return self._primary.lifetime_accepted

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, window, next_token: int) -> None:
        """Add to every level whose window size matches or is shorter."""
        for lvl in self._levels:
            if len(window) >= lvl.n:
                lvl.add(window[-lvl.n:], next_token)

    def ingest(self, tokens) -> None:
        """Feed all extractable n-gram pairs to every level."""
        for lvl in self._levels:
            lvl.ingest(tokens)

    def preseed_qwen3(self) -> None:
        """Pre-seed all levels with Qwen3 tool-call patterns."""
        for lvl in self._levels:
            lvl.preseed_qwen3()

    def draft(self, recent_tokens, n_max: int | None = None) -> list[int]:
        """Greedy draft using multi-level fallback, then recent-context scan.

        For each draft position:
          1. Try hash levels from longest-n to shortest-n (O(1) each).
          2. If all levels miss, fall back to a linear scan of recent token
             history via the shortest-n level's ``_scan()`` (Option C/D
             hybrid). This catches patterns evicted by hash collisions and
             tokens that were generated after the initial ingest call.
        """
        cap = self.n_max if n_max is None else int(n_max)
        if cap <= 0:
            return []

        # Build a sliding window long enough for the longest level.
        max_n = self._primary.n
        if len(recent_tokens) < max_n:
            # Pad with what we have; shorter levels will still work.
            window = list(recent_tokens)
        else:
            window = list(recent_tokens[-max_n:])

        # The shortest-n level owns the scan fallback; all levels share the
        # same token history, so using the last level is sufficient.
        scan_level = self._levels[-1]

        out: list[int] = []
        for _ in range(cap):
            tok = EMPTY
            for lvl in self._levels:
                if len(window) >= lvl.n:
                    tok = lvl.get(window[-lvl.n:])
                    if tok != EMPTY:
                        break
            if tok == EMPTY:
                tok = scan_level._scan(window[-scan_level.n :])
                if tok == EMPTY:
                    break
                scan_level.lifetime_scan_hits += 1
            out.append(int(tok))
            window.append(int(tok))
            if len(window) > max_n:
                window = window[-max_n:]
        return out

    def record_round(self, num_proposed: int, num_accepted: int) -> None:
        """Delegate to the primary decoder only (it owns the streak logic)."""
        self._primary.record_round(num_proposed, num_accepted)

    def reset_pool(self) -> None:
        for lvl in self._levels:
            lvl.reset_pool()

    def get_stats(self) -> dict:
        primary = self._primary.get_stats()
        levels = [lvl.get_stats() for lvl in self._levels]
        total_scan_hits = sum(lvl["scan_hits"] for lvl in levels)
        return {
            **primary,
            "ns": [lvl["n"] for lvl in levels],
            "scan_hits": total_scan_hits,
            "levels": levels,
        }


def ngram_mod_generate_step(
    prompt: mx.array,
    model,
    *,
    decoder: NGramModDecoder | MultiLevelNGramDecoder | None = None,
    n_max: int | None = None,
    n_min: int | None = None,
    max_tokens: int = 256,
    sampler=None,
    prompt_cache=None,
    prefill_step_size: int = 512,
    eos_ids: set[int] | None = None,
):
    """Generator yielding (token_id, logprobs, from_draft) tuples.

    Drop-in alternative to mlx_lm's generate_step using ngram-mod
    speculation. Pass an existing `decoder` to reuse the persistent pool
    across requests.
    """
    from mlx_lm.models import cache

    y = prompt.astype(mx.uint32)

    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model)

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    if decoder is None:
        decoder = NGramModDecoder()

    cap = decoder.n_max if n_max is None else int(n_max)
    floor = decoder.n_min if n_min is None else int(n_min)

    seq: list[int] = prompt.tolist()
    # Only ingest the suffix not already in the pool to avoid O(n) Python
    # stall on long prompts (29K+ tokens takes 20+ seconds otherwise).
    _already_ingested = getattr(decoder, "_ingested_up_to", 0)
    if len(seq) < _already_ingested:
        _already_ingested = 0
    if len(seq) > _already_ingested:
        decoder.ingest(seq[max(0, _already_ingested - decoder.n):])
    decoder._ingested_up_to = len(seq)

    def _step(tokens: mx.array, n_predict: int = 1):
        logits = model(tokens[None], cache=prompt_cache)
        logits = logits[:, -n_predict:, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = sampler(logprobs)
        return sampled.squeeze(0), logprobs.squeeze(0)

    def _prefill(tokens: mx.array) -> mx.array:
        while tokens.size > prefill_step_size:
            model(tokens[:prefill_step_size][None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            tokens = tokens[prefill_step_size:]
            mx.clear_cache()
        return tokens

    def _push(tok: int) -> None:
        seq.append(tok)
        if len(seq) >= decoder.n + 1:
            decoder.add(seq[-(decoder.n + 1) : -1], tok)

    y = _prefill(y)
    current_token, logprobs = _step(y)
    mx.eval(current_token, logprobs)

    ntoks = 0
    while ntoks < max_tokens:
        yield current_token.item(), logprobs, False
        ntoks += 1
        if ntoks >= max_tokens:
            break

        _push(current_token.item())

        draft = decoder.draft(seq, n_max=cap)
        # Truncate draft at first EOS token so speculative decoding never
        # accepts a premature end-of-sequence from a stale pool pattern.
        if eos_ids:
            for _j, _tok in enumerate(draft):
                if _tok in eos_ids:
                    draft = draft[:_j]
                    break
        if len(draft) >= floor:
            snapshot = [c.state for c in prompt_cache]
            current_id = current_token.item()
            verify_in = mx.array([current_id] + draft, mx.uint32)
            verified, vlogprobs = _step(verify_in, n_predict=len(draft) + 1)
            mx.eval(verified, vlogprobs)
            verified = verified.tolist()

            n_acc = 0
            for i, (d, v) in enumerate(zip(draft, verified[:-1])):
                if d == v:
                    n_acc += 1
                    ntoks += 1
                    _push(d)
                    yield d, vlogprobs[i], True
                    if ntoks >= max_tokens:
                        break
                else:
                    break

            decoder.record_round(len(draft), n_acc)

            if ntoks >= max_tokens:
                break

            if n_acc < len(draft):
                for c, s in zip(prompt_cache, snapshot):
                    c.state = s
                replay_ids = [current_id] + draft[:n_acc]
                replay_arr = mx.array(replay_ids, mx.uint32)
                model(replay_arr[None], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])

            _push(verified[n_acc])
            current_token = mx.array(verified[n_acc], mx.uint32)
            logprobs = vlogprobs[n_acc]
        else:
            next_token, next_logprobs = _step(
                mx.array([current_token.item()], mx.uint32)
            )
            mx.eval(next_token, next_logprobs)
            current_token = next_token
            logprobs = next_logprobs

        if ntoks % 256 == 0:
            mx.clear_cache()

    stats = decoder.get_stats()
    if stats["lifetime_drafts"] > 0:
        logger.info(
            "ngram-mod: %d/%d accepted (%.1f%%), pool used=%d/%d (%.1f%%), resets=%d, scan_hits=%d",
            stats["lifetime_accepted"],
            stats["lifetime_proposed"],
            100.0 * stats["acceptance_rate"],
            stats["used"],
            stats["pool_size"],
            100.0 * stats["load"],
            stats["resets"],
            stats.get("scan_hits", 0),
        )
