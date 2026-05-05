import json
import sys
import types


def _install_fake_mlx_lm(monkeypatch):
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = lambda *args, **kwargs: ("normal-model", "normal-tokenizer")
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)


def test_jangtq_model_uses_jang_tools_loader(tmp_path, monkeypatch):
    _install_fake_mlx_lm(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type": "deepseek_v4"}')
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"weight_format": "mxtq"})
    )

    calls = []
    jang_tools = types.ModuleType("jang_tools")
    load_jangtq = types.ModuleType("jang_tools.load_jangtq")

    def fake_load_jangtq_model(model_path):
        calls.append(model_path)
        return "jang-model", "jang-tokenizer"

    load_jangtq.load_jangtq_model = fake_load_jangtq_model
    monkeypatch.setitem(sys.modules, "jang_tools", jang_tools)
    monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq", load_jangtq)

    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    assert load_model_with_fallback(str(tmp_path)) == ("jang-model", "jang-tokenizer")
    assert calls == [tmp_path]


def test_deepseek_v4_jangtq_loader_uses_tokenizer_patch(tmp_path, monkeypatch):
    _install_fake_mlx_lm(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type": "deepseek_v4"}')
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"weight_format": "mxtq"})
    )

    events = []
    jang_tools = types.ModuleType("jang_tools")
    load_jangtq = types.ModuleType("jang_tools.load_jangtq")

    class FakePatch:
        def __init__(self, model_path):
            events.append(("init", model_path))

        def __enter__(self):
            events.append(("enter", None))

        def __exit__(self, *exc):
            events.append(("exit", None))

    def fake_load_jangtq_model(model_path):
        events.append(("load", model_path))
        return "jang-model", "jang-tokenizer"

    load_jangtq.load_jangtq_model = fake_load_jangtq_model
    monkeypatch.setitem(sys.modules, "jang_tools", jang_tools)
    monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq", load_jangtq)

    from vllm_mlx.utils import tokenizer

    monkeypatch.setattr(tokenizer, "_patch_deepseek_v4_jangtq_tokenizer", FakePatch)

    assert tokenizer.load_model_with_fallback(str(tmp_path)) == (
        "jang-model",
        "jang-tokenizer",
    )
    assert events == [
        ("init", tmp_path),
        ("enter", None),
        ("load", tmp_path),
        ("exit", None),
    ]


def test_jang_model_uses_standard_jang_loader(tmp_path, monkeypatch):
    _install_fake_mlx_lm(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5_moe"}')
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"format": "jang", "format_version": "2.0"})
    )

    calls = []
    jang_tools = types.ModuleType("jang_tools")
    loader = types.ModuleType("jang_tools.loader")

    def fake_load_jang_model(model_path):
        calls.append(model_path)
        return "jang-v2-model", "jang-v2-tokenizer"

    loader.load_jang_model = fake_load_jang_model
    monkeypatch.setitem(sys.modules, "jang_tools", jang_tools)
    monkeypatch.setitem(sys.modules, "jang_tools.loader", loader)

    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    assert load_model_with_fallback(str(tmp_path)) == (
        "jang-v2-model",
        "jang-v2-tokenizer",
    )
    assert calls == [tmp_path]


def test_non_jangtq_vendored_model_keeps_existing_fallback(tmp_path, monkeypatch):
    _install_fake_mlx_lm(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type": "deepseek_v4"}')

    from vllm_mlx.utils import tokenizer

    monkeypatch.setattr(
        tokenizer,
        "_load_with_tokenizer_fallback",
        lambda model_name: ("vendored-model", model_name),
    )

    assert tokenizer.load_model_with_fallback(str(tmp_path)) == (
        "vendored-model",
        str(tmp_path),
    )
