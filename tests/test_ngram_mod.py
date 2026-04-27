# SPDX-License-Identifier: Apache-2.0
"""Tests for NGramModDecoder."""

import pytest

from vllm_mlx.speculative.ngram_mod import EMPTY, NGramModDecoder


class TestInit:
    def test_default_init(self):
        d = NGramModDecoder()
        assert d.n == 16
        assert d.pool_size == 1 << 20
        assert d.n_min == 2
        assert d.n_max == 16
        assert d.used == 0
        assert d.lifetime_proposed == 0
        assert d.lifetime_accepted == 0

    def test_custom_init(self):
        d = NGramModDecoder(n=4, pool_size=128, n_min=1, n_max=8)
        assert d.n == 4
        assert d.pool_size == 128
        assert d.n_min == 1
        assert d.n_max == 8

    def test_empty_pool(self):
        d = NGramModDecoder(n=2, pool_size=64)
        assert all(d.entries[i] == EMPTY for i in range(64))

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            NGramModDecoder(n=0)

    def test_invalid_pool(self):
        with pytest.raises(ValueError):
            NGramModDecoder(pool_size=0)

    def test_invalid_draft_bounds(self):
        with pytest.raises(ValueError):
            NGramModDecoder(n_min=0)
        with pytest.raises(ValueError):
            NGramModDecoder(n_max=0)
        with pytest.raises(ValueError):
            NGramModDecoder(n_min=8, n_max=4)


class TestHash:
    def test_hash_deterministic(self):
        d = NGramModDecoder(n=3, pool_size=1024)
        assert d._hash([1, 2, 3]) == d._hash([1, 2, 3])

    def test_hash_in_range(self):
        d = NGramModDecoder(n=3, pool_size=64)
        for window in [[1, 2, 3], [99, 100, 101], [0, 0, 0]]:
            i = d._hash(window)
            assert 0 <= i < 64

    def test_hash_differs(self):
        d = NGramModDecoder(n=3, pool_size=1 << 20)
        assert d._hash([1, 2, 3]) != d._hash([3, 2, 1])


class TestAddGet:
    def test_add_then_get(self):
        d = NGramModDecoder(n=3, pool_size=1024)
        d.add([10, 20, 30], 42)
        assert d.get([10, 20, 30]) == 42
        assert d.used == 1

    def test_overwrite_keeps_used(self):
        d = NGramModDecoder(n=3, pool_size=1024)
        d.add([1, 2, 3], 100)
        d.add([1, 2, 3], 200)
        assert d.get([1, 2, 3]) == 200
        assert d.used == 1

    def test_wrong_window_size_noop(self):
        d = NGramModDecoder(n=3, pool_size=1024)
        d.add([1, 2], 5)
        assert d.used == 0
        assert d.get([1, 2]) == EMPTY

    def test_get_missing_returns_empty(self):
        d = NGramModDecoder(n=3, pool_size=1024)
        assert d.get([7, 8, 9]) == EMPTY

    def test_collision_does_not_return_wrong_token(self):
        d = NGramModDecoder(n=2, pool_size=1)
        d.add([1, 2], 3)
        assert d.get([9, 9]) == EMPTY
        assert d.lifetime_collisions == 0
        d.add([9, 9], 10)
        assert d.get([1, 2]) == EMPTY
        assert d.get([9, 9]) == 10
        assert d.lifetime_collisions == 1


class TestIngest:
    def test_ingest_short_sequence_noop(self):
        d = NGramModDecoder(n=4, pool_size=1024)
        d.ingest([1, 2, 3])
        assert d.used == 0

    def test_ingest_full_sequence(self):
        d = NGramModDecoder(n=2, pool_size=1024)
        d.ingest([1, 2, 3, 4, 5])
        assert d.get([1, 2]) == 3
        assert d.get([2, 3]) == 4
        assert d.get([3, 4]) == 5

    def test_ingest_repeating_pattern(self):
        d = NGramModDecoder(n=3, pool_size=4096)
        d.ingest([1, 2, 3, 4, 1, 2, 3, 4])
        assert d.get([1, 2, 3]) == 4


class TestDraft:
    def test_no_draft_when_history_short(self):
        d = NGramModDecoder(n=4, pool_size=1024)
        assert d.draft([1, 2]) == []

    def test_draft_variable_length_until_empty(self):
        d = NGramModDecoder(n=2, pool_size=4096, n_max=10)
        d.add([1, 2], 3)
        d.add([2, 3], 4)
        d.add([3, 4], 5)
        # No mapping for [4,5] → stops there
        assert d.draft([0, 1, 2]) == [3, 4, 5]

    def test_draft_caps_at_n_max(self):
        d = NGramModDecoder(n=2, pool_size=4096, n_max=2)
        d.add([1, 2], 3)
        d.add([2, 3], 4)
        d.add([3, 4], 5)
        assert d.draft([0, 1, 2]) == [3, 4]

    def test_draft_explicit_n_max_override(self):
        d = NGramModDecoder(n=2, pool_size=4096, n_max=10)
        d.add([1, 2], 3)
        d.add([2, 3], 4)
        assert d.draft([1, 2], n_max=1) == [3]

    def test_draft_zero_when_first_lookup_empty(self):
        d = NGramModDecoder(n=2, pool_size=1024)
        assert d.draft([99, 100]) == []


class TestAdaptiveReset:
    def test_low_streak_triggers_reset(self):
        d = NGramModDecoder(
            n=2, pool_size=64, n_min=1, reset_threshold=0.5, reset_streak=3
        )
        d.add([1, 2], 3)
        assert d.used == 1
        for _ in range(3):
            d.record_round(num_proposed=4, num_accepted=1)  # 25% < 50%
        assert d.used == 0
        assert d.lifetime_resets == 1

    def test_high_acceptance_resets_streak(self):
        d = NGramModDecoder(
            n=2, pool_size=64, n_min=1, reset_threshold=0.5, reset_streak=3
        )
        d.add([1, 2], 3)
        d.record_round(4, 1)
        d.record_round(4, 1)
        d.record_round(4, 4)  # high → resets streak
        d.record_round(4, 1)
        d.record_round(4, 1)
        assert d.used == 1  # not wiped yet
        assert d.lifetime_resets == 0

    def test_zero_proposed_does_not_count(self):
        d = NGramModDecoder(reset_streak=2)
        d.record_round(0, 0)
        d.record_round(0, 0)
        assert d.lifetime_resets == 0
        assert d._low_streak == 0

    def test_lifetime_counters(self):
        d = NGramModDecoder(reset_threshold=0.0)
        d.record_round(8, 6)
        d.record_round(4, 2)
        assert d.lifetime_drafts == 2
        assert d.lifetime_proposed == 12
        assert d.lifetime_accepted == 8
        s = d.get_stats()
        assert s["acceptance_rate"] == pytest.approx(8 / 12)


class TestPersistence:
    def test_pool_persists_across_rounds(self):
        d = NGramModDecoder(n=2, pool_size=1024)
        d.add([1, 2], 3)
        d.record_round(4, 4)
        assert d.get([1, 2]) == 3

    def test_reset_pool_clears_entries_only(self):
        d = NGramModDecoder(n=2, pool_size=64)
        d.add([1, 2], 3)
        d.add([2, 3], 4)
        d.lifetime_proposed = 100
        d.lifetime_accepted = 50
        d.reset_pool()
        assert d.used == 0
        assert d.get([1, 2]) == EMPTY
        # lifetime stats untouched
        assert d.lifetime_proposed == 100
        assert d.lifetime_accepted == 50
