import logging
from collections import defaultdict
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
        )
        self.tokenizer = self.engine.get_tokenizer()
        self._bos_token_id = self.tokenizer.bos_token_id
        self._stop_token_ids = self.tokenizer.eos_token_ids
        self._eot_token_id = self._select_eot_token_id()
        self._session_options = litert_lm.SessionOptions(
            apply_prompt_template_in_session=(prompt_mode == "bundle")
        )

    @staticmethod
    def _parse_backend(backend: str):
        import litert_lm

        mapping = {
            "cpu": litert_lm.Backend.CPU,
            "gpu": litert_lm.Backend.GPU,
            "npu": litert_lm.Backend.NPU,
        }
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
        return self.engine.create_session(self._session_options)

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
        token_ids = list(self.tokenizer.encode(string))
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
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **_: Any,
    ) -> list[tuple[float, bool]]:
        grouped_requests: dict[tuple[int, ...], list[tuple[int, tuple[str, str], list[int]]]]
        grouped_requests = defaultdict(list)
        results: list[tuple[float, bool] | None] = [None] * len(requests)

        for idx, (request_args, context_enc, continuation_enc) in enumerate(requests):
            grouped_requests[tuple(context_enc)].append(
                (idx, request_args, continuation_enc)
            )

        for context_tokens, grouped in tqdm(
            grouped_requests.items(),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        ):
            continuations = [continuation for _, _, continuation in grouped]
            if any(len(continuation) == 0 for continuation in continuations):
                for idx, request_args, continuation in grouped:
                    if continuation:
                        continue
                    result = (0.0, True)
                    results[idx] = result
                    self.cache_hook.add_partial("loglikelihood", request_args, result)

            non_empty = [
                (idx, request_args, continuation)
                for idx, request_args, continuation in grouped
                if continuation
            ]
            if not non_empty:
                continue

            with self._make_session() as session:
                session.run_prefill_token_ids(list(context_tokens))
                scoring = session.run_token_scoring(continuations)
                greedy_token_ids = scoring.greedy_token_ids or []
                for offset, (idx, request_args, continuation) in enumerate(non_empty):
                    greedy = greedy_token_ids[offset] if greedy_token_ids else []
                    result = (
                        float(scoring.scores[offset]),
                        list(continuation) == list(greedy),
                    )
                    results[idx] = result
                    self.cache_hook.add_partial("loglikelihood", request_args, result)

        return [result if result is not None else (0.0, False) for result in results]

    def loglikelihood_rolling(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[float]:
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

            with self._make_session() as session:
                session.run_prefill_token_ids(self.tok_encode(context))
                responses = session.run_decode(
                    self._litert_lm.DecodeOptions(
                        max_output_tokens=gen_kwargs["max_gen_toks"]
                    )
                )

            text = responses.texts[0] if responses.texts else ""
            for stop_string in until:
                if stop_string:
                    text = text.split(stop_string)[0]

            generations.append(text)
            self.cache_hook.add_partial(
                "generate_until", (context, raw_gen_kwargs), text
            )

        return generations
