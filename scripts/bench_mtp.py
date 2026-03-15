#!/usr/bin/env python3
"""
Benchmark MTP (Multi-Token Prediction) for SimpleEngine.

Compares decode throughput with and without MTP enabled.

Usage:
    python3.12 scripts/bench_mtp.py <model_path> [--num-tokens 200] [--warmup 1]
"""

import argparse
import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("bench_mtp")


async def run_benchmark(
    model_path: str,
    num_tokens: int = 200,
    warmup: int = 1,
    num_runs: int = 3,
):
    from vllm_mlx.engine.simple import SimpleEngine

    prompt = (
        "Write a detailed explanation of how neural networks learn "
        "through backpropagation, including the chain rule, gradient descent, "
        "and weight updates. Cover both forward and backward passes."
    )

    results = {}

    for mode_name, enable_mtp, optimistic in [
        ("baseline (no MTP)", False, False),
        ("MTP verified", True, False),
        ("MTP optimistic", True, True),
    ]:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode_name}")
        print(f"{'='*60}")

        engine = SimpleEngine(
            model_name=model_path,
            enable_mtp=enable_mtp,
            mtp_optimistic=optimistic,
        )
        await engine.start()

        mtp_status = "N/A"
        if enable_mtp and hasattr(engine.model, "_mtp_validated"):
            mtp_status = "validated" if engine.model._mtp_validated else "FAILED"
            print(f"  MTP validated: {mtp_status}")
            if not engine.model._mtp_validated:
                print("  SKIPPING — MTP not available")
                await engine.stop()
                continue

        # Warmup
        for i in range(warmup):
            print(f"  Warmup {i+1}/{warmup}...")
            token_count = 0
            async for chunk in engine.stream_generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
            ):
                token_count += 1
                if chunk.finished:
                    break

        # Benchmark runs
        run_results = []
        for run in range(num_runs):
            # Use slightly different prompts to avoid cache effects
            run_prompt = f"{prompt} (attempt {run+1})"
            t0 = time.perf_counter()
            token_count = 0
            first_token_time = None

            async for chunk in engine.stream_generate(
                prompt=run_prompt,
                max_tokens=num_tokens,
                temperature=0.7,
            ):
                token_count += 1
                if token_count == 1:
                    first_token_time = time.perf_counter() - t0

                if chunk.finished:
                    break

            elapsed = time.perf_counter() - t0
            decode_time = elapsed - (first_token_time or 0)
            decode_tokens = token_count - 1 if token_count > 0 else 0
            tps = decode_tokens / decode_time if decode_time > 0 else 0

            run_results.append({
                "tokens": token_count,
                "elapsed": elapsed,
                "ttft": first_token_time,
                "decode_time": decode_time,
                "decode_tokens": decode_tokens,
                "tps": tps,
            })
            print(
                f"  Run {run+1}: {token_count} tokens, "
                f"TTFT={first_token_time:.3f}s, "
                f"decode={tps:.1f} tok/s"
            )

        await engine.stop()

        # Average results
        avg_tps = sum(r["tps"] for r in run_results) / len(run_results)
        avg_ttft = sum(r["ttft"] for r in run_results) / len(run_results)
        results[mode_name] = {
            "avg_tps": avg_tps,
            "avg_ttft": avg_ttft,
            "runs": run_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<25} {'Decode tok/s':>15} {'TTFT':>10} {'Speedup':>10}")
    print("-" * 60)

    baseline_tps = results.get("baseline (no MTP)", {}).get("avg_tps", 0)
    for mode, data in results.items():
        speedup = data["avg_tps"] / baseline_tps if baseline_tps > 0 else 0
        print(
            f"{mode:<25} {data['avg_tps']:>13.1f} "
            f"{data['avg_ttft']:>9.3f}s "
            f"{speedup:>9.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark MTP for SimpleEngine")
    parser.add_argument("model", type=str, help="Model path")
    parser.add_argument(
        "--num-tokens", type=int, default=200, help="Max tokens per run (default: 200)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup runs (default: 1)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=3, help="Benchmark runs (default: 3)"
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            args.model,
            num_tokens=args.num_tokens,
            warmup=args.warmup,
            num_runs=args.num_runs,
        )
    )


if __name__ == "__main__":
    main()
