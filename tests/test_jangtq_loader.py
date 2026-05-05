import contextlib
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


def test_jang_loader_applies_tokenizer_chat_template(tmp_path, monkeypatch):
    _install_fake_mlx_lm(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5_moe"}')
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"format": "jang", "format_version": "2.0"})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{{ messages[0].content }}",
                "bos_token": "<s>",
                "eos_token": "</s>",
            }
        )
    )

    jang_tools = types.ModuleType("jang_tools")
    loader = types.ModuleType("jang_tools.loader")
    tokenizer = types.SimpleNamespace(chat_template=None, bos_token=None, eos_token=None)

    loader.load_jang_model = lambda model_path: ("jang-v2-model", tokenizer)
    monkeypatch.setitem(sys.modules, "jang_tools", jang_tools)
    monkeypatch.setitem(sys.modules, "jang_tools.loader", loader)

    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    assert load_model_with_fallback(str(tmp_path)) == ("jang-v2-model", tokenizer)
    assert tokenizer.chat_template == "{{ messages[0].content }}"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"


def test_deepseek_v4_jang_loader_uses_dsv4_chat_encoder(tmp_path, monkeypatch):
    _install_fake_mlx_lm(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type": "deepseek_v4"}')
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"weight_format": "mxtq"})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "hf-template"})
    )
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode='chat', reasoning_effort=None):\n"
        "    return f'dsv4:{thinking_mode}:{messages[-1][\"content\"]}'\n"
    )

    jang_tools = types.ModuleType("jang_tools")
    load_jangtq = types.ModuleType("jang_tools.load_jangtq")
    tokenizer = types.SimpleNamespace(
        chat_template=None,
        encode=lambda text, **kwargs: [ord(c) for c in text],
    )
    load_jangtq.load_jangtq_model = lambda model_path: ("jang-model", tokenizer)
    monkeypatch.setitem(sys.modules, "jang_tools", jang_tools)
    monkeypatch.setitem(sys.modules, "jang_tools.load_jangtq", load_jangtq)

    from vllm_mlx.utils import tokenizer as tokenizer_module

    monkeypatch.setattr(
        tokenizer_module,
        "_patch_deepseek_v4_jangtq_tokenizer",
        lambda model_path: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        tokenizer_module,
        "_patch_deepseek_v4_jangtq_rope_offset",
        lambda: None,
    )

    _, loaded_tokenizer = tokenizer_module.load_model_with_fallback(str(tmp_path))

    assert loaded_tokenizer.apply_chat_template(
        [{"role": "user", "content": "ok"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ) == "dsv4:chat:ok"
    assert loaded_tokenizer.apply_chat_template(
        [{"role": "user", "content": "ok"}],
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="high",
    ) == [ord(c) for c in "dsv4:thinking:ok"]


def test_deepseek_v4_rope_offset_patch_converts_scalar_offset(monkeypatch):
    class FakeOffset:
        def item(self):
            return 7

    class FakeRoPE:
        def __call__(self, x, offset=0, inverse=False, positions=None):
            return offset

    dsv4 = types.ModuleType("jang_tools.dsv4")
    mlx_model = types.ModuleType("jang_tools.dsv4.mlx_model")
    mlx_model.DeepseekV4RoPE = FakeRoPE
    dsv4.mlx_model = mlx_model
    monkeypatch.setitem(sys.modules, "jang_tools.dsv4", dsv4)
    monkeypatch.setitem(sys.modules, "jang_tools.dsv4.mlx_model", mlx_model)

    from vllm_mlx.utils.tokenizer import _patch_deepseek_v4_jangtq_rope_offset

    _patch_deepseek_v4_jangtq_rope_offset()

    rope = FakeRoPE()
    assert rope("x", offset=FakeOffset()) == 7
    assert rope("x", offset=3) == 3


def test_direct_generate_path_uses_mlx_lm_generate(monkeypatch):
    from vllm_mlx.engine.batched import BatchedEngine

    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")

    calls = []

    def fake_generate(model, tokenizer, **kwargs):
        calls.append((model, tokenizer, kwargs))
        return "4"

    mlx_lm.generate = fake_generate
    sample_utils.make_sampler = lambda temp, top_p: ("sampler", temp, top_p)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model = "model"

    class FakeTokenizer:
        _rapid_mlx_direct_generate = True

        def encode(self, text):
            return list(text)

    engine._tokenizer = FakeTokenizer()

    output = engine._run_direct_generate(
        prompt="prompt",
        max_tokens=8,
        temperature=0.6,
        top_p=0.95,
        stop=None,
    )

    assert output.text == "4"
    assert output.prompt_tokens == 6
    assert output.completion_tokens == 1
    assert calls == [
        (
            "model",
            engine._tokenizer,
            {
                "prompt": "prompt",
                "max_tokens": 8,
                "verbose": False,
                "sampler": ("sampler", 0.6, 0.95),
            },
        )
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
