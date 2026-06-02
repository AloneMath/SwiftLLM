from __future__ import annotations

import argparse
import builtins as py_builtins
import json
import math
import multiprocessing as mp
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swiftllm.config import load_config
from swiftllm.model import SwiftLLM
from swiftllm.tokenizer import SwiftTokenizer

SAFE_IMPORTS = {
    "math",
    "typing",
    "itertools",
    "collections",
    "functools",
    "operator",
    "heapq",
    "bisect",
    "re",
    "random",
    "statistics",
    "decimal",
    "fractions",
    "datetime",
    "copy",
    "string",
    "enum",
    "sys",
}


def extract_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local eval file not found: {p}")
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Local eval file is empty: {p}")
    return rows


def read_parquet_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Local parquet file not found: {path}")
    return pq.read_table(path).to_pylist()


def sample_rows(rows: list[dict], max_samples: int) -> list[dict]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(1337)
    rows = rows[:]
    rng.shuffle(rows)
    return rows[:max_samples]


def trim_left_to_fit(ids: list[int], max_len: int) -> list[int]:
    if max_len <= 0:
        return []
    if len(ids) <= max_len:
        return ids
    return ids[-max_len:]


def fit_prompt_and_continuation(
    prompt_ids: list[int],
    continuation_ids: list[int],
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    budget = max_seq_len - len(continuation_ids) - 1
    if budget < 0:
        raise ValueError("Continuation is longer than the model context window")
    if len(prompt_ids) > budget:
        prompt_ids = trim_left_to_fit(prompt_ids, budget)
    return prompt_ids, continuation_ids


def resolve_eval_root(eval_root: str) -> Path:
    root = Path(eval_root)
    if not root.exists():
        raise FileNotFoundError(f"Eval root does not exist: {root}")
    return root


def load_model(cfg_path: str, ckpt_path: str):
    cfg = load_config(cfg_path)
    tokenizer = SwiftTokenizer.from_file(cfg.data.tokenizer_path)
    cfg.model.vocab_size = tokenizer.get_vocab_size()

    device = torch.device(cfg.train.device)
    model = SwiftLLM(cfg.model).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return cfg, tokenizer, model, device


@torch.no_grad()
def generate_chat_completion(
    model,
    tokenizer: SwiftTokenizer,
    device: torch.device,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt_ids = tokenizer.encode_prompt(messages)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
    completion_ids = y[0].tolist()[len(prompt_ids):]
    return tokenizer.decode(completion_ids, skip_special_tokens=False)


@torch.no_grad()
def generate_plain_completion(
    model,
    tokenizer: SwiftTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    bos = tokenizer.get_bos_token_id()
    prompt_ids = tokenizer.encode_ordinary(prompt)
    x = torch.tensor([[bos] + prompt_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
    completion_ids = y[0].tolist()[len(prompt_ids) + 1 :]
    return tokenizer.decode(completion_ids, skip_special_tokens=False)


@torch.no_grad()
def score_continuation_loss(
    model,
    tokenizer: SwiftTokenizer,
    device: torch.device,
    prompt: str,
    continuation: str,
) -> float:
    bos = tokenizer.get_bos_token_id()
    prompt_ids = tokenizer.encode_ordinary(prompt)
    cont_ids = tokenizer.encode_ordinary(continuation)
    if not cont_ids:
        return float("inf")

    prompt_ids, cont_ids = fit_prompt_and_continuation(prompt_ids, cont_ids, model.cfg.max_seq_len)
    full = [bos] + prompt_ids + cont_ids
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full[1:]], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        losses = model(x, y, loss_reduction="none")[0]
    start = len(prompt_ids)
    return float(losses[start : start + len(cont_ids)].mean().item())


@torch.no_grad()
def score_continuation_losses(
    model,
    tokenizer: SwiftTokenizer,
    device: torch.device,
    prompt: str,
    continuations: list[str],
) -> list[float]:
    if not continuations:
        return []

    bos = tokenizer.get_bos_token_id()
    prompt_ids_full = tokenizer.encode_ordinary(prompt)

    packed_inputs: list[list[int]] = []
    target_ranges: list[tuple[int, int]] = []
    max_len = 0

    for continuation in continuations:
        cont_ids = tokenizer.encode_ordinary(continuation)
        if not cont_ids:
            packed_inputs.append([])
            target_ranges.append((0, 0))
            continue

        prompt_ids, cont_fit = fit_prompt_and_continuation(prompt_ids_full, cont_ids, model.cfg.max_seq_len)
        full = [bos] + prompt_ids + cont_fit
        packed_inputs.append(full)
        start = len(prompt_ids)
        end = start + len(cont_fit)
        target_ranges.append((start, end))
        max_len = max(max_len, len(full))

    if max_len < 2:
        return [float("inf")] * len(continuations)

    batch = len(continuations)
    x = torch.zeros((batch, max_len - 1), dtype=torch.long, device=device)
    y = torch.full((batch, max_len - 1), -1, dtype=torch.long, device=device)

    for i, full in enumerate(packed_inputs):
        if len(full) < 2:
            continue
        ids = torch.tensor(full, dtype=torch.long, device=device)
        n = len(full) - 1
        x[i, :n] = ids[:-1]
        y[i, :n] = ids[1:]

    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
        losses = model(x, y, loss_reduction="none")

    out: list[float] = []
    for i, (start, end) in enumerate(target_ranges):
        if end <= start:
            out.append(float("inf"))
            continue
        seg = losses[i, start:end]
        out.append(float(seg.mean().item()))
    return out


def normalize_gsm8k_row(row: dict) -> tuple[str, str]:
    question = row.get("question") or row.get("input") or row.get("prompt")
    answer = row.get("answer") or row.get("target") or row.get("output")
    if question is None or answer is None:
        raise ValueError("GSM8K row must include question and answer fields")
    return str(question), str(answer)


def normalize_humaneval_row(row: dict) -> dict:
    required = ["prompt", "canonical_solution", "test", "entry_point"]
    for key in required:
        if key not in row:
            raise ValueError(f"HumanEval row missing field: {key}")
    return row


def normalize_arc_row(row: dict) -> tuple[str, list[str], list[str], str]:
    question = row.get("question")
    choices = row.get("choices")
    answer_key = row.get("answerKey")
    if question is None or choices is None or answer_key is None:
        raise ValueError("ARC row missing question/choices/answerKey")
    labels = choices["label"]
    texts = choices["text"]
    return str(question), list(texts), list(labels), str(answer_key)


def normalize_mmlu_row(row: dict) -> tuple[str, list[str], int]:
    question = row.get("question")
    choices = row.get("choices")
    answer = row.get("answer")
    if question is None or choices is None or answer is None:
        raise ValueError("MMLU row missing question/choices/answer")
    return str(question), list(choices), int(answer)


def format_mcq_prompt(question: str, choices: list[str], labels: list[str]) -> str:
    lines = ["Question:", question, "", "Choices:"]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def choose_mcq_answer(model, tokenizer: SwiftTokenizer, device: torch.device, prompt: str, labels: list[str]) -> str:
    continuations = [f" {label}" for label in labels]
    scores = score_continuation_losses(model, tokenizer, device, prompt, continuations)
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    return labels[best_idx]


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("Relative imports are disabled")
    root = name.split(".", 1)[0]
    if root not in SAFE_IMPORTS:
        raise ImportError(f"Import not allowed: {name}")
    return py_builtins.__import__(name, globals, locals, fromlist, level)


def humaneval_worker(problem: dict, completion: str, queue):
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "reversed": reversed,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "__import__": safe_import,
    }
    namespace = {"__builtins__": safe_builtins, "__name__": "__main__"}
    program = (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
    )
    try:
        exec(program, namespace, namespace)
        queue.put(True)
    except Exception:
        queue.put(False)


def humaneval_pass(problem: dict, completion: str, timeout_s: float) -> bool:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=humaneval_worker, args=(problem, completion, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    if queue.empty():
        return False
    return bool(queue.get())


@torch.no_grad()
def eval_gsm8k(
    model,
    tokenizer,
    device,
    rows: list[dict],
    max_samples: int,
    max_new_tokens: int,
) -> float:
    rows = sample_rows(rows, max_samples)
    correct = 0
    total = 0
    for row in rows:
        question, answer = normalize_gsm8k_row(row)
        gold = extract_number(answer)
        if gold is None:
            continue
        reply = generate_chat_completion(
            model,
            tokenizer,
            device,
            messages=[
                {"role": "system", "content": "You are a careful math assistant."},
                {"role": "user", "content": f"Solve the problem and output only the final numeric answer.\n{question}"},
            ],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        pred = extract_number(reply)
        correct += int(pred == gold)
        total += 1
    return correct / max(1, total)


@torch.no_grad()
def eval_arc(model, tokenizer, device, rows: list[dict], max_samples: int) -> float:
    rows = sample_rows(rows, max_samples)
    correct = 0
    total = 0
    for row in rows:
        question, choices, labels, answer_key = normalize_arc_row(row)
        prompt = format_mcq_prompt(question, choices, labels)
        pred = choose_mcq_answer(model, tokenizer, device, prompt, labels)
        correct += int(pred == answer_key)
        total += 1
    return correct / max(1, total)


@torch.no_grad()
def eval_mmlu(model, tokenizer, device, rows: list[dict], max_samples: int) -> float:
    rows = sample_rows(rows, max_samples)
    correct = 0
    total = 0
    label_map = ["A", "B", "C", "D"]
    for row in rows:
        question, choices, answer_idx = normalize_mmlu_row(row)
        labels = label_map[: len(choices)]
        prompt = format_mcq_prompt(question, choices, labels)
        pred = choose_mcq_answer(model, tokenizer, device, prompt, labels)
        correct += int(pred == label_map[answer_idx])
        total += 1
    return correct / max(1, total)


@torch.no_grad()
def eval_smoltalk_bpb(model, tokenizer, device, rows: list[dict], max_samples: int) -> float:
    rows = sample_rows(rows, max_samples)
    token_bytes = tokenizer.build_token_bytes().to(device)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        ids, labels = tokenizer.encode_messages(messages)
        if len(ids) < 2:
            continue
        if len(ids) > model.cfg.max_seq_len:
            ids = trim_left_to_fit(ids, model.cfg.max_seq_len)
            labels = trim_left_to_fit(labels, model.cfg.max_seq_len)
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            losses = model(x, y, loss_reduction="none")[0]
        assistant_mask = torch.tensor([lab != -1 for lab in labels[1:]], dtype=torch.bool, device=device)
        target_ids = torch.tensor(ids[1:], dtype=torch.long, device=device)
        total_nats += (losses * assistant_mask).sum()
        total_bytes += token_bytes[target_ids[assistant_mask]].sum()

    if int(total_bytes.item()) == 0:
        return float("inf")
    return float(total_nats.item() / (math.log(2) * int(total_bytes.item())))


@torch.no_grad()
def eval_humaneval(
    model,
    tokenizer,
    device,
    rows: list[dict],
    max_samples: int,
    timeout_s: float,
    max_new_tokens: int,
) -> float:
    rows = sample_rows(rows, max_samples)
    passes = 0
    total = 0
    for row in rows:
        problem = normalize_humaneval_row(row)
        completion = generate_plain_completion(
            model,
            tokenizer,
            device,
            prompt=problem["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        passes += int(humaneval_pass(problem, completion, timeout_s))
        total += 1
    return passes / max(1, total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate local chat model on offline datasets")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tasks", type=str, default="gsm8k,humaneval")
    p.add_argument("--eval-root", type=str, default="./data_eval", help="Local evaluation root directory")
    p.add_argument("--gsm8k-jsonl", type=str, default="", help="Optional local GSM8K jsonl override")
    p.add_argument("--humaneval-jsonl", type=str, default="", help="Optional local HumanEval jsonl override")
    p.add_argument("--gsm8k-samples", type=int, default=100)
    p.add_argument("--arc-samples", type=int, default=100)
    p.add_argument("--mmlu-samples", type=int, default=200)
    p.add_argument("--smoltalk-samples", type=int, default=200)
    p.add_argument("--humaneval-samples", type=int, default=50)
    p.add_argument("--humaneval-timeout", type=float, default=3.0)
    p.add_argument("--gsm8k-max-new-tokens", type=int, default=96)
    p.add_argument("--humaneval-max-new-tokens", type=int, default=192)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    requested = {x.strip().lower() for x in args.tasks.split(",") if x.strip()}
    if "all" in requested:
        requested = {"gsm8k", "humaneval", "arc_challenge", "arc_easy", "mmlu", "smoltalk"}
    if "arc" in requested:
        requested.remove("arc")
        requested.add("arc_challenge")

    cfg, tokenizer, model, device = load_model(args.config, args.ckpt)
    eval_root = resolve_eval_root(args.eval_root)

    result: dict[str, float] = {}

    if "gsm8k" in requested:
        rows = read_jsonl(args.gsm8k_jsonl) if args.gsm8k_jsonl else read_parquet_rows(eval_root / "gsm8k" / "test-00000-of-00001.parquet")
        acc = eval_gsm8k(
            model,
            tokenizer,
            device,
            rows,
            args.gsm8k_samples,
            args.gsm8k_max_new_tokens,
        )
        result["gsm8k_accuracy"] = acc
        print(f"gsm8k_accuracy: {acc:.4f}")

    if "arc_challenge" in requested:
        rows = read_parquet_rows(eval_root / "ai2_arc" / "ARC-Challenge" / "test-00000-of-00001.parquet")
        acc = eval_arc(model, tokenizer, device, rows, args.arc_samples)
        result["arc_challenge_accuracy"] = acc
        print(f"arc_challenge_accuracy: {acc:.4f}")

    if "arc_easy" in requested:
        rows = read_parquet_rows(eval_root / "ai2_arc" / "ARC-Easy" / "test-00000-of-00001.parquet")
        acc = eval_arc(model, tokenizer, device, rows, args.arc_samples)
        result["arc_easy_accuracy"] = acc
        print(f"arc_easy_accuracy: {acc:.4f}")

    if "mmlu" in requested:
        rows = read_parquet_rows(eval_root / "mmlu" / "all" / "test-00000-of-00001.parquet")
        acc = eval_mmlu(model, tokenizer, device, rows, args.mmlu_samples)
        result["mmlu_accuracy"] = acc
        print(f"mmlu_accuracy: {acc:.4f}")

    if "smoltalk" in requested:
        rows = read_parquet_rows(eval_root / "smol-smoltalk" / "test-00000-of-00001.parquet")
        bpb = eval_smoltalk_bpb(model, tokenizer, device, rows, args.smoltalk_samples)
        result["smoltalk_bpb"] = bpb
        print(f"smoltalk_bpb: {bpb:.5f}")

    if "humaneval" in requested:
        rows = (
            read_jsonl(args.humaneval_jsonl)
            if args.humaneval_jsonl
            else read_parquet_rows(eval_root / "openai_humaneval" / "test-00000-of-00001.parquet")
        )
        score = eval_humaneval(
            model,
            tokenizer,
            device,
            rows,
            args.humaneval_samples,
            args.humaneval_timeout,
            args.humaneval_max_new_tokens,
        )
        result["humaneval_pass1"] = score
        print(f"humaneval_pass1: {score:.4f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
