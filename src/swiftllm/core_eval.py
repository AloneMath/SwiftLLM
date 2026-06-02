from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from jinja2 import Template


@dataclass
class CoreTaskMeta:
    task_type: str
    dataset_uri: str
    num_fewshot: int
    continuation_delimiter: str


class GPT2TokenizerCompat:
    def __init__(self, enc) -> None:
        self.enc = enc

    def __call__(self, text, prepend=None):
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids = [prepend] + ids
            return ids
        return [self.__call__(t, prepend=prepend) for t in text]

    def get_bos_token_id(self) -> int:
        return self.enc.eot_token


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(choice=choice, **context) for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(context=context_option, **context) for context_option in item["context_options"]]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {"left": range(min_len), "right": range(-1, -min_len - 1, -1)}[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    if not (start_idx < end_idx and tokens_without == tokens_with[:start_idx]):
        raise ValueError("LM prompt prefix invariant failed")
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    bsz, seq_len = input_ids.size()
    outputs = model(input_ids)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        outputs.view(bsz * seq_len, -1),
        target_ids.view(bsz * seq_len),
        reduction="none",
    ).view(bsz, seq_len)
    losses[:, -1] = float("nan")
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta: CoreTaskMeta):
    item = data[idx]

    fewshot_examples = []
    if task_meta.num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, task_meta.num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_meta.task_type == "multiple_choice":
        prompts = render_prompts_mc(item, task_meta.continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_meta.task_type == "schema":
        prompts = render_prompts_schema(item, task_meta.continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_meta.task_type == "language_modeling":
        prompts = render_prompts_lm(item, task_meta.continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_meta.task_type}")

    if hasattr(model, "cfg") and hasattr(model.cfg, "max_seq_len"):
        max_tokens = model.cfg.max_seq_len
        new_tokens, new_start, new_end = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                crop = len(t) - max_tokens
                nt = t[-max_tokens:]
                ns = s - crop
                ne = e - crop
                if ns < 0 or ne < 0:
                    continue
                new_tokens.append(nt)
                new_start.append(ns)
                new_end.append(ne)
            else:
                new_tokens.append(t)
                new_start.append(s)
                new_end.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start, new_end

    if not tokens:
        return False

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, preds = forward_model(model, input_ids)

    if task_meta.task_type == "language_modeling":
        si, ei = start_idxs[0], end_idxs[0]
        pred_t = preds[0, si - 1 : ei - 1]
        gold_t = input_ids[0, si:ei]
        return bool(torch.all(pred_t == gold_t).item())

    mean_losses = [losses[i, si - 1 : ei - 1].mean().item() for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == item["gold"]


def evaluate_task(model, tokenizer, data, device, task_meta: CoreTaskMeta) -> float:
    correct = 0
    for idx in range(len(data)):
        correct += int(evaluate_example(idx, model, tokenizer, data, device, task_meta))
    return correct / max(1, len(data))


@torch.no_grad()
def evaluate_core(
    model,
    tokenizer,
    work_dir: Path,
    device: torch.device,
    max_per_task: int = 100,
    bundle_path: str | Path | None = None,
) -> dict:
    if bundle_path is not None and str(bundle_path).strip():
        bundle = Path(bundle_path)
    else:
        bundle = work_dir / "eval_bundle"

    if not bundle.exists():
        raise FileNotFoundError(
            f"CORE eval bundle not found: {bundle}. "
            "Place it locally or set benchmark.core_bundle_path."
        )

    config_path = bundle / "core.yaml"
    data_base_path = bundle / "eval_data"
    meta_csv = bundle / "eval_meta_data.csv"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered = {}

    for task in tasks:
        label = task["label"]
        meta = CoreTaskMeta(
            task_type=task["icl_task_type"],
            dataset_uri=task["dataset_uri"],
            num_fewshot=task["num_fewshot"][0],
            continuation_delimiter=task.get("continuation_delimiter", " "),
        )

        data_path = data_base_path / meta.dataset_uri
        with data_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line.strip()) for line in f]

        rng = random.Random(1337)
        rng.shuffle(rows)
        if max_per_task > 0:
            rows = rows[:max_per_task]

        acc = evaluate_task(model, tokenizer, rows, device, meta)
        results[label] = acc

        rb = random_baselines[label]
        centered_result = (acc - 0.01 * rb) / (1.0 - 0.01 * rb)
        centered[label] = centered_result

    core_metric = sum(centered.values()) / max(1, len(centered))
    return {
        "results": results,
        "centered_results": centered,
        "core_metric": core_metric,
    }
