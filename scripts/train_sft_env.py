"""
SFT trainer for EnvTask imitation learning.

Baseline: train_instruct.py (production machinery: callbacks, batch-size adjustment,
  success.txt, LoRA helpers).
Differences from train_instruct.py:
  1. Dataset: loaded via load_from_disk (HF DatasetDict) instead of MyDataset.
  2. Trainer: SFTTrainer with tokenize_and_mask (assistant-only loss) instead of Trainer.
"""

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import bitsandbytes as bnb
import torch
import transformers
import datasets as hf_datasets
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import is_main_process
from trl import SFTConfig, SFTTrainer

from customized_trainer import (
    CustomEvalSaveCallback,
    WhenToEvalHandler,
    resize_if_needed,
    set_generation_config,
)
from state_manager import get_state, set_state
from utility import log_info

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SFTEnvTrainingArgs(SFTConfig):
    request_path: Optional[str] = field(default=None)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)


@dataclass
class LoraArguments:
    lora_r: int = 256
    lora_alpha: int = 512
    lora_dropout: float = 0.0
    lora_target_modules: str = "all"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


# ---------------------------------------------------------------------------
# Model helpers (mirrored from train_instruct.py)
# ---------------------------------------------------------------------------

def find_all_linear_names(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, torch.nn.Linear)):
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")
    return list(names)


def load_lora_model(training_args: SFTEnvTrainingArgs, model_path: str,
                    lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )

    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        target_modules = [m.strip() for m in lora_args.lora_target_modules.split() if m.strip()]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=PeftTaskType.CAUSAL_LM,
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    return model


def load_model(training_args: SFTEnvTrainingArgs, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    if training_args.use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        log_info("Using LIGER kernel")
        model_class = AutoLigerKernelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
    )
    return model


# ---------------------------------------------------------------------------
# Tokenisation + masking (from game-trajectories-sft/train.py)
# ---------------------------------------------------------------------------

def _sanitize_for_template(msgs: list, collapse_multi_toolcalls: bool) -> list:
    """Render-safe copy of `msgs` for apply_chat_template.

    DROP an empty `tool_calls` key instead of passing it through. Our rows come out of
    Arrow with a uniform message schema, so EVERY message carries `tool_calls` — a
    plain text turn gets `None` (or `[]`). The Llama family's template branches on the
    KEY EXISTING, not on it being non-empty:

        {%- if not (... or 'tool_calls' in message) %}   -> renders text
        {%- elif 'tool_calls' in message %}
        {%-   if not message.tool_calls|length == 1 %}   -> len(None) / len([]) == 0
        {{-     raise_exception("This model only supports single tool-calls at once!") }}

    so a text-only assistant turn is misrouted into the tool-call branch and dies —
    reproduced exactly on transformers 4.57.5 (the image's pin) + Llama-3.2-3B:
    `tool_calls=None` -> "object of type 'NoneType' has no len()" (OUR tournament
    error), `tool_calls=[]` -> "only supports single tool-calls" (the BOSS's error on
    the same task), key absent -> renders fine. Same bug, different Arrow default.

    `content: None -> ""` — a tool-call turn's content is unused by that branch, but a
    TEXT turn hits `message['content'] | trim`, which raises on None. Cheap and safe.

    `collapse_multi_toolcalls` — keep only the LAST tool_call. Our memory envs emit
    [memory_write, game_action]; game_action is last, so collapsing preserves the move
    and drops only the note.
    """
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        tool_calls = m.get("tool_calls") or []
        if not tool_calls:
            m.pop("tool_calls", None)
        elif collapse_multi_toolcalls and len(tool_calls) > 1:
            m["tool_calls"] = [tool_calls[-1]]
        else:
            m["tool_calls"] = tool_calls
        out.append(m)
    return out


def tokenize_and_mask(dataset: DatasetDict, tokenizer, max_length: int = 4096) -> DatasetDict:
    """Apply chat template and mask non-assistant tokens so loss is assistant-only."""
    def _process(example):
        msgs = example["messages"]
        tools = example.get("tools") or None
        if isinstance(tools, str):
            # Tools are stored as a JSON string, not a list of dicts: Arrow unifies
            # struct schemas across a list column's elements, which corrupts
            # heterogeneous tool `parameters.properties` (e.g. a no-arg tool's `{}`
            # becomes `{"some_other_tool_arg": None}`). JSON-encoding avoids that.
            tools = json.loads(tools)
        # Same Arrow-struct-unification issue applies to tool_calls[].function.arguments
        # across envs (e.g. game envs' {"action_id": int} vs intercode's {"command": str}):
        # generators store `arguments` JSON-encoded as a string. Decode back to a dict
        # here so the chat template's `tojson` filter renders it correctly (it would
        # double-encode an already-JSON string).
        msgs = [
            {
                **m,
                "tool_calls": [
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": (
                                json.loads(call["function"]["arguments"])
                                if isinstance(call["function"].get("arguments"), str)
                                else call["function"].get("arguments")
                            ),
                        },
                    }
                    for call in m["tool_calls"]
                ],
            }
            if m.get("tool_calls")
            else m
            for m in msgs
        ]
        # Render defensively instead of model-sniffing: the tournament picks a random
        # model per round and some families' templates reject our turn shape (len() on
        # a None content; "only supports single tool-calls at once" on a
        # [memory_write, game_action] turn). Attempt 1 is the full turn with a safe
        # content; attempt 2 collapses the turn to its game_action. The try/except IS
        # the per-model probe. A row no variant can render is dropped (filtered below)
        # rather than killing the whole run — an unrenderable row used to raise inside
        # dataset.map and fail training before step 1.
        for collapse in (False, True):
            cand = _sanitize_for_template(msgs, collapse)
            try:
                ids = tokenizer.apply_chat_template(cand, tools=tools, tokenize=True, add_generation_prompt=False)
                mask = [0] * len(ids)
                for i, msg in enumerate(cand):
                    if msg["role"] != "assistant":
                        continue
                    p = len(tokenizer.apply_chat_template(cand[:i],   tools=tools, tokenize=True, add_generation_prompt=True))
                    r = len(tokenizer.apply_chat_template(cand[:i+1], tools=tools, tokenize=True, add_generation_prompt=False))
                    for j in range(p, r):
                        mask[j] = 1
                if len(ids) > max_length:
                    ids, mask = ids[:max_length], mask[:max_length]
                return {"input_ids": ids, "assistant_masks": mask}
            except Exception:
                continue
        return {"input_ids": [], "assistant_masks": []}

    log_info("Tokenizing + masking dataset...")
    hf_datasets.disable_progress_bar()
    result = dataset.map(_process, num_proc=min(48, max(4, (os.cpu_count() or 8) - 2)))
    # Drop rows that NO template variant could render (see _process). Loud, because a
    # large drop means this model's template disagrees with our turn shape and the
    # surviving mix is not what we intended to train on.
    before = {split: len(result[split]) for split in result}
    result = result.filter(lambda r: len(r["input_ids"]) > 0)
    dropped = {s: before[s] - len(result[s]) for s in before if before[s] != len(result[s])}
    if dropped:
        log_info(f"tokenize: dropped {dropped} unrenderable row(s) (chat-template rejected them)")
    hf_datasets.enable_progress_bar()
    log_info("Tokenizing + masking done.")
    return result


# ---------------------------------------------------------------------------
# ORPO preference branch  (SWE_ORPO gen -> {prompt, chosen, rejected})
# ---------------------------------------------------------------------------

def _run_orpo(training_args, lora_args, train_request, tokenizer, raw):
    """Reference-free ORPO (odds-ratio preference) on a SWE {prompt, chosen, rejected} set.
    Reuses the SFT model-load + CustomEvalSaveCallback; ORPOTrainer tokenises the
    conversational pairs itself (no tokenize_and_mask). Entered ONLY when the dataset
    carries chosen/rejected columns, so the default SFT path stays byte-identical."""
    try:
        from trl import ORPOConfig, ORPOTrainer  # trl <= ~0.28
    except ImportError:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer  # trl >= ~0.29

    beta = float(os.environ.get("SWE_ORPO_BETA", "0.1"))
    _lr = os.environ.get("SWE_ORPO_LR")
    learning_rate = float(_lr) if _lr else training_args.learning_rate
    world = max(1, int(getattr(training_args, "world_size", 1) or 1))

    # ORPOTrainer applies the chat template ITSELF, so this path never passes through
    # tokenize_and_mask and does NOT inherit its _sanitize_for_template guard. A strict
    # template (the Llama family raises "This model only supports single tool-calls at
    # once!" on a turn whose tool_calls is empty/None or holds >1 call) would therefore
    # raise inside the trainer and take the whole task down — the exact failure that made
    # two tournament finalists ship untrained models. Today's only preference producer
    # (swe_trajectories) emits plain {role, content}, so this is a guard against a future
    # tool-calling env feeding the pair path, not a live bug. Sanitize the message columns
    # and drop any pair the tokenizer still cannot render, so a bad row costs one pair
    # instead of the run.
    def _sanitize_pref(example):
        out = {}
        for col in ("prompt", "chosen", "rejected"):
            v = example.get(col)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                out[col] = _sanitize_for_template(v, True)
        return out

    def _renderable(example) -> bool:
        base = example.get("prompt")
        if not (isinstance(base, list) and base and isinstance(base[0], dict)):
            return True  # non-conversational (plain-string) pairs: nothing to probe
        for col in ("chosen", "rejected"):
            cont = example.get(col)
            if not (isinstance(cont, list) and cont and isinstance(cont[0], dict)):
                continue
            try:
                tokenizer.apply_chat_template(list(base) + list(cont), tokenize=False)
            except Exception:
                return False
        return True

    train_ds = raw["train"]
    _n_before = len(train_ds)
    train_ds = train_ds.map(_sanitize_pref).filter(_renderable)
    n_train = len(train_ds)
    if n_train != _n_before:
        log_info(f"[orpo] dropped {_n_before - n_train} unrenderable pair(s) "
                 f"(chat-template rejected them); {n_train} left")
    if n_train == 0:
        log_info("[orpo] WARNING: every preference pair was rejected by the chat template "
                 "— nothing to train on. Check the generator's message schema.")

    orpo_args = ORPOConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_steps=training_args.warmup_steps,
        weight_decay=training_args.weight_decay,
        bf16=training_args.bf16,
        tf32=getattr(training_args, "tf32", None),
        gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if training_args.gradient_checkpointing else None
        ),
        optim=training_args.optim,
        logging_steps=training_args.logging_steps,
        save_strategy=training_args.save_strategy,
        eval_strategy="no",
        report_to=training_args.report_to,
        max_length=training_args.max_length or 8192,
        beta=beta,
        remove_unused_columns=False,
        save_only_model=True,
    )
    log_info(
        f"[orpo] beta={beta} lr={learning_rate} max_length={orpo_args.max_length} "
        f"epochs={orpo_args.num_train_epochs} pairs={n_train}"
    )

    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        resize_if_needed(train_request["model_name"], model, len(tokenizer))
    try:
        model.config.use_cache = False
    except Exception:
        pass
    set_generation_config(train_request["model_name"], model)

    if is_main_process(LOCAL_RANK):
        os.makedirs(orpo_args.output_dir, exist_ok=True)

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    max_steps = train_request.get("max_steps", -1)
    total_steps_per_epoch = max(
        1,
        n_train // (orpo_args.per_device_train_batch_size
                    * orpo_args.gradient_accumulation_steps * world),
    )
    total_steps_all_epochs = int(total_steps_per_epoch * orpo_args.num_train_epochs)
    log_info(
        f"[orpo] total_steps_per_epoch: {total_steps_per_epoch}; "
        f"total_steps_all_epochs: {total_steps_all_epochs}"
    )
    checking_step = train_request.get("checking_step", 70)
    if checking_step >= total_steps_per_epoch:
        checking_step = max(total_steps_per_epoch - 2, 1)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)

    success_file = os.path.join(orpo_args.output_dir, "success.txt")
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=train_ds,
        eval_dataset=None,
        processing_class=tokenizer,
        callbacks=[
            CustomEvalSaveCallback(
                WhenToEvalHandler(
                    train_request["end_time"],
                    train_request["save_before_remaining_time"],
                    periodic_save_steps=periodic_save_steps,
                    steps_per_epoch=total_steps_per_epoch,
                    max_steps=max_steps,
                ),
                train_request["submission_dir"],
                orpo_args.output_dir,
                train_request["model_name"],
                max_steps,
                checking_step=checking_step,
                total_steps_all_epochs=total_steps_all_epochs,
                end_time=train_request["end_time"],
                checking_mode=train_request.get("checking_mode", "none"),
            )
        ],
    )
    trainer.train()
    if is_main_process(LOCAL_RANK):
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("Training successfully done", "finish")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argument_parser = transformers.HfArgumentParser((SFTEnvTrainingArgs, LoraArguments))
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()

    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    task_id = train_request["task_id"]

    tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load pre-generated trajectories and apply assistant masking
    raw: DatasetDict = load_from_disk(train_request["dataset_path"])
    log_info(f"Dataset: train={len(raw['train'])}")

    # ORPO preference branch: the SWE_ORPO gen writes {prompt, chosen, rejected}. Detect it
    # by columns and switch to ORPOTrainer (conversational, self-tokenising). Absent those
    # columns the SFT path below is byte-identical to before.
    _cols = raw["train"].column_names if "train" in raw else []
    if "chosen" in _cols and "rejected" in _cols:
        log_info("[orpo] preference columns detected -> ORPO branch")
        _run_orpo(training_args, lora_args, train_request, tokenizer, raw)
        return

    dataset = tokenize_and_mask(raw, tokenizer, max_length=training_args.max_length or 4096)

    train_ds = dataset["train"]
    log_info(f"train_size: {len(train_ds)}")

    # Batch size adjustment (mirrors train_instruct.py)
    original_steps = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    log_info(f"original_steps: {original_steps}")

    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = max(int(max_batch_size_theory), 1)

    if (training_args.per_device_train_batch_size > max_batch_size_theory
            and train_request.get("adjust_batch_size", True)):
        log_info(
            f"Reducing batch size {training_args.per_device_train_batch_size} → {max_batch_size_theory}"
        )
        training_args.per_device_train_batch_size = max_batch_size_theory

    # Load model
    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    try:
        model.config.use_cache = False
    except Exception:
        pass

    set_generation_config(train_request["model_name"], model)

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"Output dir: {training_args.output_dir}")

    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    max_steps = train_request.get("max_steps", -1)
    training_args.save_only_model = True

    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs
    log_info(
        f"total_steps_per_epoch: {total_steps_per_epoch}; "
        f"total_steps_all_epochs: {total_steps_all_epochs}"
    )

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)

    success_file = os.path.join(training_args.output_dir, "success.txt")
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)

    checking_step = train_request.get("checking_step", 70)
    if checking_step >= total_steps_per_epoch:
        checking_step = max(total_steps_per_epoch - 2, 1)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[
            CustomEvalSaveCallback(
                WhenToEvalHandler(
                    train_request["end_time"],
                    train_request["save_before_remaining_time"],
                    periodic_save_steps=periodic_save_steps,
                    steps_per_epoch=total_steps_per_epoch,
                    max_steps=max_steps,
                ),
                train_request["submission_dir"],
                training_args.output_dir,
                train_request["model_name"],
                max_steps,
                checking_step=checking_step,
                total_steps_all_epochs=total_steps_all_epochs,
                end_time=train_request["end_time"],
                checking_mode=train_request.get("checking_mode", "none"),
            )
        ],
    )

    trainer.train()

    if is_main_process(LOCAL_RANK):
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("Training successfully done", "finish")


if __name__ == "__main__":
    main()
