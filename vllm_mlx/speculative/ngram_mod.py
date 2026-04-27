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
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

EMPTY: int = -1
EMPTY_KEY: int = 0
_MULT: int = 6364136223846793005
_MASK64: int = (1 << 64) - 1


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
        n_min: int = 2,
        n_max: int = 16,
        reset_threshold: float = 0.5,
        reset_streak: int = 3,
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
        self.reset_threshold = float(reset_threshold)
        self.reset_streak = int(reset_streak)

        self.entries: array.array = self._make_pool()
        self.keys: array.array = self._make_keys()
        self.used: int = 0

        self._low_streak: int = 0

        self.lifetime_proposed: int = 0
        self.lifetime_accepted: int = 0
        self.lifetime_drafts: int = 0
        self.lifetime_resets: int = 0
        self.lifetime_collisions: int = 0

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
        if self.keys[i] == EMPTY_KEY:
            self.used += 1
        elif self.keys[i] != key:
            self.lifetime_collisions += 1
        self.keys[i] = key
        self.entries[i] = int(next_token) & 0x7FFFFFFF

    def get(self, window) -> int:
        if len(window) != self.n:
            return EMPTY
        i, key = self._slot_and_key(window)
        if self.keys[i] != key:
            return EMPTY
        return self.entries[i]

    def ingest(self, tokens) -> None:
        """Add every (n-gram window, next-token) pair extractable from tokens."""
        if len(tokens) < self.n + 1:
            return
        for w in range(len(tokens) - self.n):
            self.add(tokens[w : w + self.n], tokens[w + self.n])

    def draft(self, recent_tokens, n_max: int | None = None) -> list[int]:
        """Greedy variable-length draft from the last n tokens."""
        cap = self.n_max if n_max is None else int(n_max)
        if cap <= 0 or len(recent_tokens) < self.n:
            return []
        window = list(recent_tokens[-self.n :])
        out: list[int] = []
        for _ in range(cap):
            tok = self.get(window)
            if tok == EMPTY:
                break
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

    def reset_pool(self) -> None:
        self.entries = self._make_pool()
        self.keys = self._make_keys()
        self.used = 0
        self.lifetime_resets += 1

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
        }


def ngram_mod_generate_step(
    prompt: mx.array,
    model,
    *,
    decoder: NGramModDecoder | None = None,
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
    decoder.ingest(seq)

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
            "ngram-mod: %d/%d accepted (%.1f%%), pool used=%d/%d (%.1f%%), resets=%d",
            stats["lifetime_accepted"],
            stats["lifetime_proposed"],
            100.0 * stats["acceptance_rate"],
            stats["used"],
            stats["pool_size"],
            100.0 * stats["load"],
            stats["resets"],
        )
