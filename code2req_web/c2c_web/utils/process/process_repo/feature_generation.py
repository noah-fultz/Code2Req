import logging
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass

import tiktoken
from django.conf import settings

from c2c_web.utils.prompts import get_tag_features_sys_prompt, get_tag_features_user_prompt


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturePromptBudget:
    input_token_limit: int
    context_window: int
    output_reserve: int
    safety_margin: int

    @property
    def max_input_tokens(self):
        available_context = self.context_window - self.output_reserve - self.safety_margin
        allowance = min(self.input_token_limit, available_context)
        if allowance <= 0:
            raise ValueError("Repository feature token budget leaves no room for input")
        return allowance


class FeaturePromptTooLarge(ValueError):
    def __init__(self, model, stage, estimated_tokens, max_input_tokens):
        self.model = model
        self.stage = stage
        self.estimated_tokens = estimated_tokens
        self.max_input_tokens = max_input_tokens
        super().__init__(
            f"Repository feature prompt for {stage} exceeds the configured input budget " +
            f"({estimated_tokens} estimated tokens > {max_input_tokens}; model={model})"
        )


class FeatureReductionError(ValueError):
    pass


class FeatureTokenizerConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class LocalTokenizerEncoding:
    tokenizer: object
    profile: str

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids


def get_feature_prompt_budget():
    return FeaturePromptBudget(
        input_token_limit=settings.REPOSITORY_FEATURE_INPUT_TOKEN_LIMIT,
        context_window=settings.REPOSITORY_FEATURE_CONTEXT_TOKENS,
        output_reserve=settings.REPOSITORY_FEATURE_OUTPUT_TOKENS,
        safety_margin=settings.REPOSITORY_FEATURE_SAFETY_MARGIN_TOKENS,
    )


@lru_cache(maxsize=4)
def _load_local_tokenizer(tokenizer_path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(tokenizer_path)


def get_feature_tokenizer(model):
    tokenizer_path = settings.REPOSITORY_FEATURE_TOKENIZER_JSON
    tokenizer_model = settings.REPOSITORY_FEATURE_TOKENIZER_MODEL
    strict = settings.REPOSITORY_FEATURE_REQUIRE_MODEL_TOKENIZER

    if tokenizer_path:
        path = Path(tokenizer_path)
        if not path.is_file():
            raise FeatureTokenizerConfigurationError(
                f"Configured repository feature tokenizer file does not exist: {path}"
            )
        if not tokenizer_model:
            raise FeatureTokenizerConfigurationError(
                "REPOSITORY_FEATURE_TOKENIZER_MODEL must identify the model tag "
                "that matches REPOSITORY_FEATURE_TOKENIZER_JSON"
            )
        if model != tokenizer_model:
            raise FeatureTokenizerConfigurationError(
                f"Tokenizer profile {tokenizer_model!r} does not match selected model {model!r}"
            )
        return LocalTokenizerEncoding(_load_local_tokenizer(str(path)), tokenizer_model)

    if strict:
        raise FeatureTokenizerConfigurationError(
            "Strict repository feature token counting requires "
            "REPOSITORY_FEATURE_TOKENIZER_JSON and REPOSITORY_FEATURE_TOKENIZER_MODEL"
        )
    return tiktoken.encoding_for_model("gpt-4")


def render_feature_messages(payload, feature_type):
    payload_text = str(payload)
    return [
        {"role": "system", "content": get_tag_features_sys_prompt(feature_type)},
        {
            "role": "user",
            "content": get_tag_features_user_prompt(payload_text),
        },
    ]


def estimate_feature_message_tokens(messages, encoding=None):
    """Count message content and reserve a fixed allowance for chat framing."""
    if encoding is None:
        encoding = tiktoken.encoding_for_model("gpt-4")

    content_tokens = sum(len(encoding.encode(message["content"])) for message in messages)
    message_framing = 64 + 4 * len(messages) + 3
    return content_tokens + message_framing


def check_feature_prompt_budget(messages, model, stage, budget=None, encoding=None):
    budget = budget or get_feature_prompt_budget()
    encoding = encoding or get_feature_tokenizer(model)
    estimated_tokens = estimate_feature_message_tokens(messages, encoding)
    max_input_tokens = budget.max_input_tokens
    estimator = getattr(encoding, "profile", "tiktoken:gpt-4-estimate")
    logger.info(
        "Repository feature prompt stage=%s model=%s estimated_input_tokens=%d "
        "input_limit=%d estimator=%s+estimated-chat-framing",
        stage,
        model,
        estimated_tokens,
        max_input_tokens,
        estimator,
    )
    if estimated_tokens > max_input_tokens:
        raise FeaturePromptTooLarge(model, stage, estimated_tokens, max_input_tokens)
    return estimated_tokens


def log_feature_response_telemetry(response, model, stage):
    prompt_eval_count = getattr(response, "prompt_eval_count", None)
    total_duration = getattr(response, "total_duration", None)
    done_reason = getattr(response, "done_reason", None)
    logger.info(
        "Repository feature response stage=%s model=%s prompt_eval_count=%s "
        "total_duration_ns=%s done_reason=%s",
        stage,
        model,
        prompt_eval_count,
        total_duration,
        done_reason,
    )


def get_feature_generation_options(budget=None):
    budget = budget or get_feature_prompt_budget()
    return {
        "temperature": 0,
        "num_ctx": budget.context_window,
        "num_predict": budget.output_reserve,
    }


def pack_feature_records(records, feature_type, model, budget=None, encoding=None):
    """Pack summary records by recounting each complete rendered prompt."""
    budget = budget or get_feature_prompt_budget()
    ordered_records = sorted((str(key), str(value)) for key, value in records.items())
    if not ordered_records:
        return []

    batches = []
    current = {}
    for key, value in ordered_records:
        candidate = dict(current)
        candidate[key] = value
        try:
            check_feature_prompt_budget(
                render_feature_messages(candidate, feature_type),
                model,
                "feature-reduction-pack",
                budget=budget,
                encoding=encoding,
            )
        except FeaturePromptTooLarge:
            if not current:
                raise
            batches.append(current)
            current = {key: value}
            check_feature_prompt_budget(
                render_feature_messages(current, feature_type),
                model,
                "feature-reduction-pack",
                budget=budget,
                encoding=encoding,
            )
        else:
            current = candidate

    if current:
        batches.append(current)
    return batches


def summarize_bounded(
    records,
    feature_type,
    model,
    stage,
    summarize_once,
    budget=None,
    encoding=None,
    initial_batches=None,
    max_rounds=8,
):
    """Map prompt-sized inputs, then reduce their Feature summaries to one result.

    `summarize_once` receives `(payload, feature_type, model, stage)` and returns
    either a Feature or `(Feature, score)`. Source batches may be supplied through
    `initial_batches` to retain syntax-aware chunking and source lineage.
    """
    budget = budget or get_feature_prompt_budget()
    batches = (
        list(initial_batches)
        if initial_batches is not None
        else pack_feature_records(records, feature_type, model, budget, encoding)
    )
    if not batches:
        raise FeatureReductionError(f"No records supplied for {stage} feature summary")

    scores = []

    def generate_batch(payload, mode, round_number):
        check_feature_prompt_budget(
            render_feature_messages(payload, mode),
            model,
            f"{stage}-round-{round_number}",
            budget=budget,
            encoding=encoding,
        )
        result = summarize_once(payload, mode, model, stage)
        if isinstance(result, tuple):
            feature, score = result
            if score is not None:
                scores.append(float(score))
        else:
            feature = result
        if not hasattr(feature, "name") or not hasattr(feature, "description"):
            raise FeatureReductionError(f"Invalid Feature result from {stage} summarization")
        return feature

    summaries = [
        generate_batch(payload, feature_type, 0)
        for payload in batches
    ]
    if len(summaries) == 1:
        return summaries[0], sum(scores) / len(scores) if scores else 0.0

    for round_number in range(1, max_rounds + 1):
        summary_records = {
            f"summary_{index:08d}": f"{feature.name}: {feature.description}"
            for index, feature in enumerate(summaries)
        }
        reduction_batches = pack_feature_records(
            summary_records,
            "Features",
            model,
            budget=budget,
            encoding=encoding,
        )
        if len(reduction_batches) >= len(summary_records):
            raise FeatureReductionError(
                f"Feature reduction for {stage} cannot make progress within the token budget"
            )

        reduced = [
            generate_batch(payload, "Features", round_number)
            for payload in reduction_batches
        ]
        if len(reduced) >= len(summaries):
            raise FeatureReductionError(
                f"Feature reduction for {stage} did not reduce the summary count"
            )
        if len(reduced) == 1:
            return reduced[0], sum(scores) / len(scores) if scores else 0.0
        summaries = reduced

    raise FeatureReductionError(
        f"Feature reduction for {stage} exceeded the {max_rounds}-round limit"
    )