import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models import utils as model_utils


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


@register_model("litert-lm", "litert_lm")
class LiteRTLM(TemplateLM):
    """LiteRT-LM backend for evaluating `.litertlm` bundles."""

    _DEFAULT_MAX_LENGTH = 4096
    _DEFAULT_MAX_GEN_TOKS = 256

    def __init__(
        self,
        pretrained: str,
        backend: str = "cpu",
        max_length: int | None = None,
        batch_size: int = 1,
        max_batch_size: int = 64,
        max_gen_toks: int = _DEFAULT_MAX_GEN_TOKS,
        cache_dir: str = "",
        prompt_mode: str = "raw",
        input_prompt_as_hint: str = "",
        num_cpu_threads: int = 0,
        prefill_chunk_size: int = -1,
        **_: Any,
    ) -> None:
        super().__init__()

        try:
            import litert_lm
        except ImportError as exc:
            raise ImportError(
                "litert_lm is required for the litert-lm backend. "
                "Build and install the LiteRT-LM wheel first."
            ) from exc

        self._litert_lm = litert_lm
        self.pretrained = pretrained
        self.max_length = max_length or self._DEFAULT_MAX_LENGTH
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.max_gen_toks = max_gen_toks
        self.prompt_mode = prompt_mode
        self.num_cpu_threads = int(num_cpu_threads)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self._device = backend
        self._backend = self._parse_backend(backend)
        self.backend = "causal"

        if prompt_mode not in {"raw", "bundle"}:
            raise ValueError("prompt_mode must be 'raw' or 'bundle'")

        if batch_size != 1:
            eval_logger.warning(
                "LiteRT-LM generate_until currently runs one request at a time. "
                "Ignoring batch_size=%s.",
                batch_size,
            )
        if max_batch_size != 64:
            eval_logger.warning(
                "LiteRT-LM loglikelihood batching is context-grouped internally; "
                "max_batch_size=%s is currently ignored.",
                max_batch_size,
            )

        self.engine = litert_lm.Engine(
            pretrained,
            backend=self._backend,
            max_num_tokens=self.max_length,
            cache_dir=cache_dir,
            input_prompt_as_hint=input_prompt_as_hint,
            num_cpu_threads=self.num_cpu_threads,
            prefill_chunk_size=self.prefill_chunk_size,
        )
        self.tokenizer = None
        self._bos_token_id = self.engine.bos_token_id
        self._stop_token_ids = self.engine.eos_token_ids or []
        self._eot_token_id = self._select_eot_token_id()

    @staticmethod
    def _parse_backend(backend: str):
        import litert_lm

        mapping = {
            "cpu": litert_lm.Backend.CPU,
            "gpu": litert_lm.Backend.GPU,
        }
        if hasattr(litert_lm.Backend, "NPU"):
            mapping["npu"] = litert_lm.Backend.NPU
        try:
            return mapping[backend.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported LiteRT-LM backend: {backend}") from exc

    def _select_eot_token_id(self) -> int:
        for token_ids in self._stop_token_ids:
            if len(token_ids) == 1:
                return token_ids[0]
        if self._bos_token_id is not None:
            return self._bos_token_id
        raise ValueError(
            "LiteRT-LM backend requires a single-token stop token or BOS token."
        )

    def _make_session(self):
        return self.engine.create_session(
            apply_prompt_template=(self.prompt_mode == "bundle")
        )

    def _make_generate_config(self, max_output_tokens: int):
        config_cls = getattr(self._litert_lm, "GenerateConfig", None)
        if config_cls is None:
            config_cls = getattr(self._litert_lm, "DecodeOptions")
        return config_cls(max_output_tokens=max_output_tokens)

    @staticmethod
    def _is_greedy_match(scoring: Any, index: int, continuation_enc: list[int]) -> bool:
        greedy_token_ids = getattr(scoring, "greedy_token_ids", None)
        if not greedy_token_ids or index >= len(greedy_token_ids):
            return False
        return list(greedy_token_ids[index]) == list(continuation_enc)

    def _tokenize(self, string: str) -> list[int]:
        return list(self.engine.tokenize(string))

    def _detokenize(self, tokens: list[int]) -> str:
        return self.engine.detokenize(tokens)

    def _rolling_max_seq_len(self) -> int:
        # LiteRT-LM scores continuations after prefill, so the effective
        # context window for rolling perplexity is `context + continuation`.
        # Reserve one token for context to avoid hitting the runtime's exact
        # `max_num_tokens` guard on the first window.
        return max(1, self.max_length - 1)

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def prefix_token_id(self) -> int:
        return self._bos_token_id or self.eot_token_id

    @property
    def tokenizer_name(self) -> str:
        return f"litert-lm::{Path(self.pretrained).name}"

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **_: Any
    ) -> list[int]:
        token_ids = self._tokenize(string)
        if self._bos_token_id is None:
            return token_ids
        if add_special_tokens is False and token_ids[:1] == [self._bos_token_id]:
            return token_ids[1:]
        if add_special_tokens is True and token_ids[:1] != [self._bos_token_id]:
            return [self._bos_token_id, *token_ids]
        return token_ids

    def tok_decode(
        self, tokens: int | list[int], skip_special_tokens: bool = True
    ) -> str:
        del skip_special_tokens
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._detokenize(tokens)

    def loglikelihood(
        self,
        requests: list["Instance"],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        return super().loglikelihood(requests, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **_: Any,
    ) -> list[tuple[float, bool]]:
        if self.prompt_mode != "raw":
            raise NotImplementedError(
                "LiteRT-LM token-scoring path currently requires prompt_mode=raw."
            )

        results: list[tuple[float, bool] | None] = [None] * len(requests)
        grouped_requests: dict[
            tuple[int, ...],
            list[tuple[int, tuple[str, str], list[int], list[int]]],
        ] = {}

        for index, request in enumerate(requests):
            cache_key, context_enc, continuation_enc = request
            if not continuation_enc:
                result = (0.0, True)
                results[index] = result
                self.cache_hook.add_partial("loglikelihood", cache_key, result)
                continue
            grouped_requests.setdefault(tuple(context_enc), []).append(
                (index, cache_key, context_enc, continuation_enc)
            )

        for context_key, group in tqdm(
            grouped_requests.items(),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
            total=len(grouped_requests),
        ):
            del context_key
            context_enc = group[0][2]
            continuation_encs = [continuation_enc for _, _, _, continuation_enc in group]

            with self._make_session() as session:
                session.run_prefill_token_ids(context_enc)
                scoring = session.run_token_scoring(continuation_encs)

            for score_index, (score, (index, cache_key, _, continuation_enc)) in enumerate(
                zip(scoring.scores, group, strict=True)
            ):
                result = (
                    float(score),
                    self._is_greedy_match(scoring, score_index, continuation_enc),
                )
                results[index] = result
                self.cache_hook.add_partial("loglikelihood", cache_key, result)

        return [result for result in results if result is not None]

    def loglikelihood_rolling(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[float]:
        if self.prompt_mode != "raw":
            raise NotImplementedError(
                "LiteRT-LM token-scoring path currently requires prompt_mode=raw."
            )

        loglikelihoods = []

        for request in tqdm(
            requests,
            disable=disable_tqdm,
            desc="Running loglikelihood_rolling requests",
        ):
            string = request.args[0]
            rolling_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string, add_special_tokens=False),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self._rolling_max_seq_len(),
                        context_len=1,
                    ),
                )
            )

            total_score = 0.0
            for context_tokens, continuation_tokens in rolling_windows:
                with self._make_session() as session:
                    session.run_prefill_token_ids(context_tokens)
                    scoring = session.run_token_scoring([continuation_tokens])
                total_score += float(scoring.scores[0])

            loglikelihoods.append(total_score)
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total_score)

        return loglikelihoods

    def generate_until(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[str]:
        if not requests:
            return []

        generations = []

        for request in tqdm(
            requests, disable=disable_tqdm, desc="Running generate_until requests"
        ):
            context, raw_gen_kwargs = request.args
            gen_kwargs = model_utils.normalize_gen_kwargs(
                raw_gen_kwargs, default_max_gen_toks=self.max_gen_toks
            )
            until = gen_kwargs.get("until", [])
            if gen_kwargs.get("do_sample") or gen_kwargs.get("temperature", 0.0) > 0.0:
                eval_logger.warning(
                    "LiteRT-LM backend currently uses greedy generation for "
                    "lm-eval; sampling kwargs are ignored."
                )
            decode_config = self._make_generate_config(int(gen_kwargs["max_gen_toks"]))

            with self._make_session() as session:
                session.run_prefill([context])
                responses = session.run_decode(decode_config)

            text = responses.texts[0] if responses.texts else ""
            for stop_string in until:
                if stop_string:
                    text = text.split(stop_string)[0]

            generations.append(text)
            self.cache_hook.add_partial(
                "generate_until", (context, raw_gen_kwargs), text
            )

        return generations
