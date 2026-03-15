#!/usr/bin/env python3
"""
Generate synthetic MTP weights for speed benchmarking.

Creates random quantized MTP weights matching the model's architecture.
The predictions will be random, but the code path and timing are valid.

Usage:
    python3.12 scripts/gen_synthetic_mtp_weights.py <model_path>
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to MLX model directory")
    args = parser.parse_args()

    model_dir = Path(args.model_path)
    config_path = model_dir / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    head_dim = config["head_dim"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    num_experts = config["num_experts"]
    moe_intermediate_size = config["moe_intermediate_size"]
    shared_expert_intermediate_size = config["shared_expert_intermediate_size"]
    vocab_size = config["vocab_size"]
    rms_norm_eps = config["rms_norm_eps"]

    quant_config = config.get("quantization", {})
    bits = quant_config.get("bits", 6)
    group_size = quant_config.get("group_size", 64)

    print(f"Model: {model_dir.name}")
    print(f"hidden_size={hidden_size}, num_experts={num_experts}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")

    # MTP weight shapes
    weights = {}

    # Norms (kept in FP)
    for name in [
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
    ]:
        weights[name] = mx.ones((hidden_size,))
        print(f"  FP: {name} {weights[name].shape}")

    # Q/K norm (head_dim)
    for name in [
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
    ]:
        weights[name] = mx.ones((head_dim,))
        print(f"  FP: {name} {weights[name].shape}")

    # FC: 2*hidden_size -> hidden_size
    fc_weight = mx.random.normal((hidden_size, hidden_size * 2)) * 0.01
    q_w, q_s, q_b = mx.quantize(fc_weight, group_size=group_size, bits=bits)
    mx.eval(q_w, q_s, q_b)
    weights["mtp.fc.weight"] = q_w
    weights["mtp.fc.scales"] = q_s
    weights["mtp.fc.biases"] = q_b
    print(f"  Quantized: mtp.fc.weight {q_w.shape}")

    # Attention projections
    attn_shapes = {
        "q_proj": (num_attention_heads * head_dim * 2, hidden_size),
        "k_proj": (num_key_value_heads * head_dim, hidden_size),
        "v_proj": (num_key_value_heads * head_dim, hidden_size),
        "o_proj": (hidden_size, num_attention_heads * head_dim),
    }
    for proj_name, (out_dim, in_dim) in attn_shapes.items():
        key = f"mtp.layers.0.self_attn.{proj_name}.weight"
        w = mx.random.normal((out_dim, in_dim)) * 0.01
        q_w, q_s, q_b = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(q_w, q_s, q_b)
        weights[key] = q_w
        weights[key.replace(".weight", ".scales")] = q_s
        weights[key.replace(".weight", ".biases")] = q_b
        print(f"  Quantized: {key} {q_w.shape}")

    # MoE gate (kept as 8-bit per quant_predicate)
    gate_w = mx.random.normal((num_experts, hidden_size)) * 0.01
    q_w, q_s, q_b = mx.quantize(gate_w, group_size=64, bits=8)
    mx.eval(q_w, q_s, q_b)
    weights["mtp.layers.0.mlp.gate.weight"] = q_w
    weights["mtp.layers.0.mlp.gate.scales"] = q_s
    weights["mtp.layers.0.mlp.gate.biases"] = q_b
    print(f"  Quantized 8-bit: mtp.layers.0.mlp.gate.weight {q_w.shape}")

    # Shared expert gate (8-bit)
    se_gate_w = mx.random.normal((1, hidden_size)) * 0.01
    weights["mtp.layers.0.mlp.shared_expert_gate.weight"] = se_gate_w
    print(f"  FP: mtp.layers.0.mlp.shared_expert_gate.weight {se_gate_w.shape}")

    # Shared expert MLP
    for proj in ["gate_proj", "up_proj"]:
        key = f"mtp.layers.0.mlp.shared_expert.{proj}.weight"
        w = mx.random.normal((shared_expert_intermediate_size, hidden_size)) * 0.01
        q_w, q_s, q_b = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(q_w, q_s, q_b)
        weights[key] = q_w
        weights[key.replace(".weight", ".scales")] = q_s
        weights[key.replace(".weight", ".biases")] = q_b
        print(f"  Quantized: {key} {q_w.shape}")

    key = "mtp.layers.0.mlp.shared_expert.down_proj.weight"
    w = mx.random.normal((hidden_size, shared_expert_intermediate_size)) * 0.01
    q_w, q_s, q_b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q_w, q_s, q_b)
    weights[key] = q_w
    weights[key.replace(".weight", ".scales")] = q_s
    weights[key.replace(".weight", ".biases")] = q_b
    print(f"  Quantized: {key} {q_w.shape}")

    # Expert MLP (stacked: [num_experts, intermediate_size, hidden_size])
    for proj in ["gate_proj", "up_proj"]:
        key = f"mtp.layers.0.mlp.switch_mlp.{proj}.weight"
        w = mx.random.normal((num_experts, moe_intermediate_size, hidden_size)) * 0.01
        q_w, q_s, q_b = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(q_w, q_s, q_b)
        weights[key] = q_w
        weights[key.replace(".weight", ".scales")] = q_s
        weights[key.replace(".weight", ".biases")] = q_b
        print(f"  Quantized: {key} {q_w.shape}")

    key = "mtp.layers.0.mlp.switch_mlp.down_proj.weight"
    w = mx.random.normal((num_experts, hidden_size, moe_intermediate_size)) * 0.01
    q_w, q_s, q_b = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q_w, q_s, q_b)
    weights[key] = q_w
    weights[key.replace(".weight", ".scales")] = q_s
    weights[key.replace(".weight", ".biases")] = q_b
    print(f"  Quantized: {key} {q_w.shape}")

    # Save
    output_file = model_dir / "model-mtp.safetensors"
    print(f"\nSaving {len(weights)} weight tensors to {output_file}")
    mx.save_safetensors(str(output_file), weights)

    total_bytes = sum(v.nbytes for v in weights.values())
    print(f"MTP weights size: {total_bytes / 1e6:.1f} MB")

    # Update index
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for key in weights:
            index["weight_map"][key] = "model-mtp.safetensors"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Updated index with {len(weights)} MTP entries")

    # Update config
    config["num_nextn_predict_layers"] = 1
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Updated config: num_nextn_predict_layers=1")

    print("\nDone! Synthetic MTP weights added for benchmarking.")
    print("Note: These are random weights — MTP predictions will be random.")
    print("Use --mtp-optimistic for maximum speed benchmark.")


if __name__ == "__main__":
    main()
