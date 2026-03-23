import importlib
import sys
import types
from dataclasses import dataclass, field

from lm_eval.api.instance import Instance
from lm_eval.api.registry import model_registry


def _make_fake_litert_lm_module():
    @dataclass
    class Responses:
        texts: list[str] = field(default_factory=list)
        scores: list[float] = field(default_factory=list)
        token_lengths: list[int] = field(default_factory=list)
        token_scores: list[list[float]] = field(default_factory=list)
        token_ids: list[list[int]] = field(default_factory=list)
        greedy_token_ids: list[list[int]] = field(default_factory=list)
        finish_reasons: list[str] = field(default_factory=list)

    @dataclass
    class SessionOptions:
        apply_prompt_template_in_session: bool | None = None
        max_output_tokens: int | None = None
        num_output_candidates: int | None = None

    @dataclass
    class DecodeOptions:
        max_output_tokens: int | None = None

    class Backend:
        CPU = "cpu"
        GPU = "gpu"
        NPU = "npu"

    class FakeTokenizer:
        bos_token_id = 101
        eos_token_ids = [[102]]

        def encode(self, text):
            mapping = {
                "ctx": [11],
                " cont1": [21],
                " cont2": [22],
                "abc": [1, 2, 3],
                "abcd": [1, 2, 3, 4],
            }
            return list(mapping.get(text, []))

        def decode(self, token_ids):
            reverse = {
                (11,): "ctx",
                (21,): " cont1",
                (22,): " cont2",
                (1, 2, 3): "abc",
                (1, 2, 3, 4): "abcd",
            }
            return reverse.get(tuple(token_ids), "")

    class FakeSession:
        def __init__(self):
            self.prefills = []
            self.scoring_targets = []
            self.decode_options = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb

        def run_prefill_token_ids(self, token_ids):
            self.prefills.append(list(token_ids))

        def run_token_scoring(self, target_token_ids, store_token_lengths=False):
            del store_token_lengths
            self.scoring_targets.append([list(tokens) for tokens in target_token_ids])
            return Responses(
                scores=[float(sum(tokens)) for tokens in target_token_ids],
                greedy_token_ids=[list(tokens) for tokens in target_token_ids],
            )

        def run_decode(self, options=None):
            self.decode_options.append(options)
            return Responses(texts=["hello STOP trailing"])

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.tokenizer = FakeTokenizer()
            self.sessions = []

        def create_session(self, options=None):
            del options
            session = FakeSession()
            self.sessions.append(session)
            return session

        def get_tokenizer(self):
            return self.tokenizer

    return types.SimpleNamespace(
        Backend=Backend,
        DecodeOptions=DecodeOptions,
        Engine=FakeEngine,
        Responses=Responses,
        SessionOptions=SessionOptions,
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
    assert model.engine.sessions[0].prefills == [[101, 11]]


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
    assert session.prefills == [[11]]
    assert session.decode_options[0].max_output_tokens == 5


def test_litert_lm_loglikelihood_rolling_reserves_one_token(monkeypatch):
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
    assert model.engine.sessions[0].prefills == [[101]]
    assert model.engine.sessions[0].scoring_targets == [[[1, 2, 3]]]
    assert model.engine.sessions[1].prefills == [[1, 2, 3]]
    assert model.engine.sessions[1].scoring_targets == [[[4]]]
