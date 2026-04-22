import importlib
import sys
import types
from dataclasses import dataclass, field

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.registry import model_registry


def _make_fake_litert_lm_module():
    @dataclass
    class Responses:
        texts: list[str] = field(default_factory=list)
        scores: list[float] = field(default_factory=list)
        token_lengths: list[int] = field(default_factory=list)
        token_scores: list[list[float]] = field(default_factory=list)
        greedy_token_ids: list[list[int]] = field(default_factory=list)

    @dataclass
    class GenerateConfig:
        max_output_tokens: int | None = None

    class DecodeOptions(GenerateConfig):
        pass

    class Backend:
        CPU = "cpu"
        GPU = "gpu"
        NPU = "npu"

    class FakeSession:
        def __init__(self):
            self.prefill_texts = []
            self.prefill_token_ids = []
            self.scoring_targets = []
            self.decode_configs = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

        def run_prefill(self, contents):
            self.prefill_texts.append(list(contents))

        def run_prefill_token_ids(self, token_ids):
            self.prefill_token_ids.append(list(token_ids))

        def run_token_scoring(self, target_token_ids, store_token_lengths=False):
            del store_token_lengths
            normalized_targets = [list(tokens) for tokens in target_token_ids]
            self.scoring_targets.append(normalized_targets)
            return Responses(
                scores=[float(sum(tokens)) for tokens in normalized_targets],
                greedy_token_ids=[list(tokens) for tokens in normalized_targets],
            )

        def run_decode(self, config=None):
            self.decode_configs.append(config)
            return Responses(texts=["hello STOP trailing"])

    class FakeEngine:
        bos_token_id = 101
        eos_token_ids = [[102]]

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.sessions = []

        def create_session(self, **kwargs):
            del kwargs
            session = FakeSession()
            self.sessions.append(session)
            return session

        def tokenize(self, text):
            mapping = {
                "ctx": [11],
                " cont1": [21],
                " cont2": [22],
                "abc": [1, 2, 3],
                "abcd": [1, 2, 3, 4],
            }
            return list(mapping.get(text, []))

        def detokenize(self, token_ids):
            reverse = {
                (11,): "ctx",
                (21,): " cont1",
                (22,): " cont2",
                (1, 2, 3): "abc",
                (1, 2, 3, 4): "abcd",
            }
            return reverse.get(tuple(token_ids), "")

    return types.SimpleNamespace(
        Backend=Backend,
        DecodeOptions=DecodeOptions,
        Engine=FakeEngine,
        GenerateConfig=GenerateConfig,
        Responses=Responses,
    )


def _load_backend(monkeypatch):
    fake_module = _make_fake_litert_lm_module()
    monkeypatch.setitem(sys.modules, "litert_lm", fake_module)
    sys.modules.pop("lm_eval.models.litert_lm", None)
    model_registry._objs.pop("litert-lm", None)
    model_registry._objs.pop("litert_lm", None)
    litert_lm_model = importlib.import_module("lm_eval.models.litert_lm")
    return litert_lm_model, fake_module


def test_litert_lm_loglikelihood_groups_shared_context(monkeypatch):
    litert_lm_model, _ = _load_backend(monkeypatch)
    model = litert_lm_model.LiteRTLM(pretrained="/tmp/fake.litertlm")

    results = model._loglikelihood_tokens(
        [
            (("ctx", " cont1"), [101, 11], [21]),
            (("ctx", " cont2"), [101, 11], [22]),
        ]
    )

    assert results == [(21.0, True), (22.0, True)]
    assert len(model.engine.sessions) == 1
    session = model.engine.sessions[0]
    assert session.prefill_token_ids == [[101, 11]]
    assert session.scoring_targets == [[[21], [22]]]


def test_litert_lm_loglikelihood_bundle_mode_not_supported(monkeypatch):
    litert_lm_model, _ = _load_backend(monkeypatch)
    model = litert_lm_model.LiteRTLM(
        pretrained="/tmp/fake.litertlm", prompt_mode="bundle"
    )

    with pytest.raises(NotImplementedError, match="prompt_mode=raw"):
        model._loglikelihood_tokens([(("ctx", " cont1"), [101, 11], [21])])


def test_litert_lm_generate_until_trims_stop_string(monkeypatch):
    litert_lm_model, _ = _load_backend(monkeypatch)
    model = litert_lm_model.LiteRTLM(pretrained="/tmp/fake.litertlm")

    request = Instance(
        request_type="generate_until",
        doc={},
        arguments=("ctx", {"until": "STOP", "max_gen_toks": 5}),
        idx=0,
    )
    results = model.generate_until([request], disable_tqdm=True)

    assert results == ["hello "]
    session = model.engine.sessions[0]
    assert session.prefill_texts == [["ctx"]]
    assert session.decode_configs[0].max_output_tokens == 5


def test_litert_lm_loglikelihood_rolling_uses_token_scoring(monkeypatch):
    litert_lm_model, _ = _load_backend(monkeypatch)
    model = litert_lm_model.LiteRTLM(pretrained="/tmp/fake.litertlm", max_length=4)

    request = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        arguments=("abcd",),
        idx=0,
    )
    results = model.loglikelihood_rolling([request], disable_tqdm=True)

    assert results == [10.0]
    assert len(model.engine.sessions) == 2
    assert model.engine.sessions[0].prefill_token_ids == [[101]]
    assert model.engine.sessions[0].scoring_targets == [[[1, 2, 3]]]
    assert model.engine.sessions[1].prefill_token_ids == [[1, 2, 3]]
    assert model.engine.sessions[1].scoring_targets == [[[4]]]
