import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_dflash_router import count_prompt_tokens, select_backend  # noqa: E402


def test_short_single_request_uses_dflash() -> None:
    assert select_backend(12_000, 1, 14_000) == "dflash_n4"


def test_long_request_uses_target_only() -> None:
    assert select_backend(16_000, 1, 14_000) == "target_only"


def test_batch_request_uses_target_only() -> None:
    assert select_backend(8_000, 2, 14_000) == "target_only"


class _DictTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": [list(range(16_002))], "attention_mask": [[]]}


def test_chat_template_dict_counts_input_ids() -> None:
    tokens, batch_size = count_prompt_tokens(
        _DictTokenizer(), {"messages": [{"role": "user", "content": "x"}]}
    )
    assert tokens == 16_002
    assert batch_size == 1
