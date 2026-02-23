from pathlib import Path

import numpy as np
import transformers
import typer
from datasets import load_dataset
from transformers import AutoTokenizer

# Transformers v5 renamed 'additional_special_tokens' to 'extra_special_tokens'
_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5
_SPECIAL_TOKENS_KEY = "extra_special_tokens" if _TRANSFORMERS_V5 else "additional_special_tokens"

app = typer.Typer()

# Import from main codebase instead of duplicating
from mini_trainer.gpt_oss_utils import is_gpt_oss_tokenizer as is_gpt_oss_model


def is_gpt_oss_assistant_channel(pos, original_ids, tokenizer):
    """Check if current position is within a GPT-OSS assistant channel pattern."""
    if not tokenizer:
        return False
    # Look for pattern: <|start|>assistant<|channel|>CHANNEL_NAME<|message|>
    if pos >= len(original_ids):
        return False

    # Try to encode the expected pattern tokens
    try:
        start_token = tokenizer.encode("<|start|>", add_special_tokens=False)
        assistant_tokens = tokenizer.encode("assistant", add_special_tokens=False)
        channel_token = tokenizer.encode("<|channel|>", add_special_tokens=False)
        message_token = tokenizer.encode("<|message|>", add_special_tokens=False)
    except:
        return False

    # Look backwards from current position to see if we're in an assistant channel
    for lookback in range(min(pos + 1, 50)):  # Look back up to 50 tokens
        start_pos = pos - lookback
        if start_pos < 0:
            break

        # Check if we have the start of an assistant channel pattern
        if (
            start_pos + len(start_token) + len(assistant_tokens) + len(channel_token) <= len(original_ids)
            and original_ids[start_pos : start_pos + len(start_token)] == start_token
            and original_ids[start_pos + len(start_token) : start_pos + len(start_token) + len(assistant_tokens)]
            == assistant_tokens
            and original_ids[
                start_pos + len(start_token) + len(assistant_tokens) : start_pos
                + len(start_token)
                + len(assistant_tokens)
                + len(channel_token)
            ]
            == channel_token
        ):
            # Found assistant channel start, now look for the message token ahead
            message_start = start_pos + len(start_token) + len(assistant_tokens) + len(channel_token)
            for forward in range(
                min(len(original_ids) - message_start, 20)
            ):  # Look forward up to 20 tokens for message token
                if (
                    message_start + forward + len(message_token) <= len(original_ids)
                    and original_ids[message_start + forward : message_start + forward + len(message_token)]
                    == message_token
                ):
                    # Found complete assistant channel pattern
                    message_end = message_start + forward + len(message_token)
                    # Check if current position is between message token and next special sequence
                    if pos >= message_end:
                        return True
                    break
            break

    return False


def make_input_ids_from_messages(sample: dict, tokenizer):
    sample["pretrain"] = False
    sample["error"] = False
    sample["input_ids"] = None
    sample["len"] = None
    try:
        roles = [s["role"] for s in sample["messages"]]

        if "pretrain" in roles:
            assert len(roles) == 1
            content = sample["messages"][0]["content"]
            sample["input_ids"] = tokenizer.encode(content + tokenizer.eos_token, add_special_tokens=False)
            sample["pretrain"] = True
        else:
            sample["input_ids"] = tokenizer.apply_chat_template(sample["messages"], tokenize=True)
            messages = sample["messages"]
            for m in messages:
                # Check if assistant has content, thinking, or reasoning_content
                if m["role"] == "assistant":
                    has_content = bool(m.get("content"))
                    has_thinking = bool(m.get("thinking"))
                    has_reasoning = bool(m.get("reasoning_content"))
                    if not (has_content or has_thinking or has_reasoning):
                        sample["error"] = True
        sample["len"] = len(sample["input_ids"])
        return sample
    except Exception:
        sample["error"] = True
        return sample


def make_labels_from_input_ids(sample: dict, assistant_tk_ids: list, user_tk_ids: list, tokenizer=None):
    """
    Create training labels by unmasking only the assistant's reply tokens and masking all other tokens (user messages and special delimiters) with -100. For pretraining samples, labels equal the input_ids.
    """
    if sample["pretrain"]:
        sample["labels"] = sample["input_ids"]
        return sample

    original_ids = sample["input_ids"]
    labels = []
    unmasking = False

    # Check if this is a GPT-OSS model for special handling
    is_gpt_oss = is_gpt_oss_model(tokenizer)

    i = 0
    while i < len(original_ids):
        # Check if the next tokens match the assistant delimiter sequence
        if original_ids[i : i + len(assistant_tk_ids)] == assistant_tk_ids:
            unmasking = True
            labels.extend([-100 for _ in assistant_tk_ids])
            i += len(assistant_tk_ids)
            continue
        # Check if the next tokens match the user delimiter sequence
        elif original_ids[i : i + len(user_tk_ids)] == user_tk_ids:
            unmasking = False
            # i += len(user_tk_ids)
            # continue

        # Special case: Check for GPT-OSS assistant channel patterns
        # For GPT-OSS, always unmask assistant channels
        if is_gpt_oss and is_gpt_oss_assistant_channel(i, original_ids, tokenizer):
            unmasking = True

        # else:
        token = original_ids[i]
        if unmasking:
            labels.append(token)
        else:
            labels.append(-100)
        i += 1

    sample["labels"] = labels
    return sample


def make_num_loss_tokens_from_labels(sample: dict):
    sample["num_loss_counted_tokens"] = sum([l != -100 for l in sample["labels"]])
    return sample


def infer_special_token_sequences(tokenizer):
    """
    this function tries to infer the special token sequences from the tokenizer.
    Sometimes the chat template adds some tokens to the messages in between that should be unmasked.
    """
    tk_1 = tokenizer.encode("1", add_special_tokens=False)
    assert len(tk_1) == 1
    tk_1 = tk_1[0]

    messages = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "1"},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    assert sum([t == tk_1 for t in input_ids]) == 2, "tk_1 should be only 2 times to infer the special token sequences"
    tk_1_first_idx = [i for i, t in enumerate(input_ids) if t == tk_1][0]
    tk_1_second_idx = [i for i, t in enumerate(input_ids) if t == tk_1][1]

    user_tk_ids_1 = input_ids[:tk_1_first_idx]
    assistant_tk_ids_1 = input_ids[tk_1_first_idx + 1 : tk_1_second_idx]

    messages = [
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": "1"},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    assert sum([t == tk_1 for t in input_ids]) == 2, "tk_1 should be only 2 times to infer the special token sequences"
    tk_1_first_idx = [i for i, t in enumerate(input_ids) if t == tk_1][0]
    tk_1_second_idx = [i for i, t in enumerate(input_ids) if t == tk_1][1]

    user_tk_ids_2 = input_ids[tk_1_first_idx + 1 : tk_1_second_idx]
    assistant_tk_ids_2 = input_ids[:tk_1_first_idx]

    def longest_common_subsequence(seq1, seq2):
        """Return the longest common subsequence (LCS) between two sequences using dynamic programming."""
        m, n = len(seq1), len(seq2)
        # dp[i][j] will hold the length of the LCS of seq1[:i] and seq2[:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Build the dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Reconstruct the LCS from the dp table
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                lcs.append(seq1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        lcs.reverse()
        return lcs

    assistant_tk_ids = longest_common_subsequence(assistant_tk_ids_1, assistant_tk_ids_2)
    user_tk_ids = longest_common_subsequence(user_tk_ids_1, user_tk_ids_2)
    return assistant_tk_ids, user_tk_ids


@app.command()
def process_data(
    input_jsonl: str = typer.Option(..., "--input-file", help="path to the input jsonl file"),
    output_jsonl: str = typer.Option(..., "--output-file", help="path to the output tokenizedjsonl file"),
    model_name_or_path: str = typer.Option(..., "--model-name-or-path"),
    max_sample_num_tokens: int = typer.Option(
        2147483647,
        help="max number of tokens in a sample, samples longer than this will be removed",
    ),
    string_for_printing_masks: str = typer.Option(
        "<|mAsK|>",
        "--string-for-printing-masks",
        help="when printing samples at the end, the masked tokens in the labels will be replaced with this string",
    ),
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    assistant_tk_ids, user_tk_ids = infer_special_token_sequences(tokenizer)
    tokenizer.add_special_tokens({_SPECIAL_TOKENS_KEY: [string_for_printing_masks]})
    string_for_printing_masks_tk = tokenizer.encode(string_for_printing_masks, add_special_tokens=False)[0]

    dataset = load_dataset("json", data_files=input_jsonl, split="train")

    dataset_with_input_ids = dataset.map(
        lambda x: make_input_ids_from_messages(x, tokenizer),
        num_proc=64,
    )
    dataset_with_input_ids = dataset_with_input_ids.filter(
        lambda x: not x["error"],
        num_proc=64,
    )
    print(
        "\033[38;5;196m"
        + f"Total number of filtered samples after removing samples with errors: {len(dataset) - len(dataset_with_input_ids)}"
        + "\033[0m"
    )

    dataset_with_input_ids = dataset_with_input_ids.filter(
        lambda x: x["len"] <= max_sample_num_tokens,
        num_proc=64,
    )
    print(
        "\033[38;5;196m"
        + f"Total number of filtered samples after removing samples longer than {max_sample_num_tokens} tokens: {len(dataset) - len(dataset_with_input_ids)}"
        + "\033[0m"
    )

    dataset_with_labels = dataset_with_input_ids.map(
        lambda x: make_labels_from_input_ids(x, assistant_tk_ids, user_tk_ids, tokenizer),
        num_proc=64,
    )

    dataset_with_labels = dataset_with_labels.map(
        make_num_loss_tokens_from_labels,
        num_proc=64,
    )

    # printing some samples to check the results
    random_indices = np.random.permutation(len(dataset_with_labels))[:2]
    for i in random_indices:
        sample = dataset_with_labels[int(i)]
        print("original messages:")
        print("\033[38;5;10m" + str(sample["messages"]) + "\033[0m")
        print("input_ids:")
        print("\033[38;5;51m" + tokenizer.decode(sample["input_ids"]) + "\033[0m")
        print("labels:")
        label_ids = [l if l != -100 else string_for_printing_masks_tk for l in sample["labels"]]
        print("\033[38;5;208m" + tokenizer.decode(label_ids) + "\033[0m")
        print("-" * 100)

    dataset_with_labels.to_json(Path(output_jsonl), num_proc=64)


if __name__ == "__main__":
    app()

"""
# python process_data.py --input-file /new_data/knowledge_rh/quality/training_mix/entigraph_knowledge1.0_phi4_first_24_n_5_5_percent.jsonl \
python process_data.py --input-file '/Users/aldo/Downloads/data.jsonl' \
      --output-file ./some_product_puzzle_tokenized_qwen1.5b.jsonl \
      --model-name-or-path Qwen/Qwen2.5-1.5B \
      --string-for-printing-masks "<|mAsK|>" \
      --max-sample-num-tokens 16384
"""
