import tempfile
from pathlib import Path
from types import SimpleNamespace
from django.test import SimpleTestCase, override_settings
from unittest.mock import patch

from c2c_web.utils.process.process_repo import cluster_code, join_features
from c2c_web.utils.process.process_repo.feature_generation import (
    FeaturePromptBudget,
    FeaturePromptTooLarge,
    FeatureTokenizerConfigurationError,
    check_feature_prompt_budget,
    estimate_feature_message_tokens,
    get_feature_generation_options,
    get_feature_prompt_budget,
    get_feature_tokenizer,
    render_feature_messages,
)


class CharacterEncoding:
    def encode(self, text):
        return list(text)


class StubTokenizer:
    def encode(self, text, add_special_tokens):
        return SimpleNamespace(ids=list(text))


class FeaturePromptBudgetTests(SimpleTestCase):
    def test_estimate_counts_system_user_and_message_framing(self):
        messages = [
            {"role": "system", "content": "ab"},
            {"role": "user", "content": "cde"},
        ]

        self.assertEqual(estimate_feature_message_tokens(messages, CharacterEncoding()), 80)

    def test_request_at_budget_is_accepted(self):
        messages = [{"role": "user", "content": "abc"}]
        budget = FeaturePromptBudget(
            input_token_limit=74,
            context_window=100,
            output_reserve=5,
            safety_margin=2,
        )

        count = check_feature_prompt_budget(
            messages, "qwen-test", "unit-test", budget, CharacterEncoding()
        )

        self.assertEqual(count, 74)

    def test_request_over_budget_fails_before_model_call(self):
        messages = [{"role": "user", "content": "abcd"}]
        budget = FeaturePromptBudget(
            input_token_limit=74,
            context_window=100,
            output_reserve=5,
            safety_margin=2,
        )

        with self.assertRaises(FeaturePromptTooLarge) as error:
            check_feature_prompt_budget(
                messages, "qwen-test", "unit-test", budget, CharacterEncoding()
            )

        self.assertEqual(error.exception.estimated_tokens, 75)
        self.assertEqual(error.exception.max_input_tokens, 74)

    def test_context_limit_includes_output_reserve_and_margin(self):
        budget = FeaturePromptBudget(
            input_token_limit=100,
            context_window=20,
            output_reserve=5,
            safety_margin=2,
        )

        self.assertEqual(budget.max_input_tokens, 13)

    def test_invalid_budget_is_rejected(self):
        budget = FeaturePromptBudget(
            input_token_limit=100,
            context_window=5,
            output_reserve=4,
            safety_margin=2,
        )

        with self.assertRaises(ValueError):
            _ = budget.max_input_tokens

    @override_settings(
        REPOSITORY_FEATURE_INPUT_TOKEN_LIMIT=100,
        REPOSITORY_FEATURE_CONTEXT_TOKENS=80,
        REPOSITORY_FEATURE_OUTPUT_TOKENS=20,
        REPOSITORY_FEATURE_SAFETY_MARGIN_TOKENS=10,
    )
    def test_settings_build_budget_and_generation_options(self):
        budget = get_feature_prompt_budget()

        self.assertEqual(budget.max_input_tokens, 50)
        self.assertEqual(
            get_feature_generation_options(budget),
            {"temperature": 0, "num_ctx": 80, "num_predict": 20},
        )

    def test_renderer_preserves_feature_prompt_and_payload(self):
        payload = {"file.py#work": "return value"}

        messages = render_feature_messages(payload, "Code")

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("expert at summarizing", messages[0]["content"])
        self.assertIn(str(payload), messages[1]["content"])

    @override_settings(
        REPOSITORY_FEATURE_TOKENIZER_JSON="",
        REPOSITORY_FEATURE_TOKENIZER_MODEL="",
        REPOSITORY_FEATURE_REQUIRE_MODEL_TOKENIZER=True,
    )
    def test_strict_mode_requires_a_model_tokenizer(self):
        with self.assertRaises(FeatureTokenizerConfigurationError):
            get_feature_tokenizer("qwen2.5-coder:32b")

    @override_settings(
        REPOSITORY_FEATURE_TOKENIZER_JSON="/missing/tokenizer.json",
        REPOSITORY_FEATURE_TOKENIZER_MODEL="qwen2.5-coder:32b",
        REPOSITORY_FEATURE_REQUIRE_MODEL_TOKENIZER=True,
    )
    def test_configured_missing_tokenizer_file_fails_clearly(self):
        with self.assertRaises(FeatureTokenizerConfigurationError):
            get_feature_tokenizer("qwen2.5-coder:32b")

    def test_local_tokenizer_must_match_selected_model_and_is_used_for_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer_path = Path(temp_dir) / "tokenizer.json"
            tokenizer_path.write_text("{}", encoding="utf-8")
            with override_settings(
                REPOSITORY_FEATURE_TOKENIZER_JSON=str(tokenizer_path),
                REPOSITORY_FEATURE_TOKENIZER_MODEL="qwen2.5-coder:32b",
                REPOSITORY_FEATURE_REQUIRE_MODEL_TOKENIZER=True,
            ):
                with self.assertRaises(FeatureTokenizerConfigurationError):
                    get_feature_tokenizer("qwen2.5-coder:7b")

                with patch(
                    "c2c_web.utils.process.process_repo.feature_generation._load_local_tokenizer",
                    return_value=StubTokenizer(),
                ):
                    encoding = get_feature_tokenizer("qwen2.5-coder:32b")

        self.assertEqual(encoding.profile, "qwen2.5-coder:32b")
        self.assertEqual(encoding.encode("code"), list("code"))

    def test_feature_description_uses_passed_budget_for_ollama_options(self):
        budget = FeaturePromptBudget(
            input_token_limit=5000,
            context_window=6000,
            output_reserve=700,
            safety_margin=100,
        )
        response = SimpleNamespace(
            message=SimpleNamespace(content='{"name":"Feature","description":"Description"}'),
            prompt_eval_count=123,
            total_duration=456,
            done_reason="stop",
        )

        with (
            patch.object(cluster_code, "safe_ollama_chat", return_value=response) as ollama_call,
            patch.object(cluster_code, "embed", return_value=[[1.0]]),
            patch.object(cluster_code, "cosine_similarity", return_value=0.9),
        ):
            cluster_code.get_description(
                {"artifact": "code"}, "Code", "qwen-test", budget=budget
            )

        self.assertEqual(
            ollama_call.call_args.kwargs["options"],
            {"temperature": 0, "num_ctx": 6000, "num_predict": 700},
        )

    @override_settings(
        REPOSITORY_FEATURE_INPUT_TOKEN_LIMIT=1,
        REPOSITORY_FEATURE_CONTEXT_TOKENS=100,
        REPOSITORY_FEATURE_OUTPUT_TOKENS=10,
        REPOSITORY_FEATURE_SAFETY_MARGIN_TOKENS=10,
    )
    def test_feature_call_sites_reject_oversized_prompt_before_ollama(self):
        for module in (cluster_code, join_features):
            with self.subTest(module=module.__name__):
                with patch.object(module, "safe_ollama_chat") as ollama_call:
                    with self.assertRaises(FeaturePromptTooLarge):
                        module.get_description({"large-artifact": "x" * 1000}, "Code", "qwen-test")

                    ollama_call.assert_not_called()