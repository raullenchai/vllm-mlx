# SPDX-License-Identifier: Apache-2.0

import vllm_mlx.request_metrics as request_metrics
from vllm_mlx.request_metrics import RequestRecorder


def test_finish_uses_engine_ttft_for_single_chunk_generation_tps(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(request_metrics.time, "time", lambda: now[0])

    recorder = RequestRecorder()
    req_id = recorder.start("/v1/chat/completions")

    now[0] = 110.0
    recorder.finish(
        req_id,
        prompt_tokens=1000,
        generated_tokens=100,
        non_streaming=False,
        engine_ttft=2.0,
    )

    entry = recorder.last()
    assert entry is not None
    assert entry["ttft"] == 2.0
    assert entry["generation_tps"] == 12.5
