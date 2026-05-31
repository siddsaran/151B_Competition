import csv
import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_ID = "sisaran/lora-math-qwen3"
DEFAULT_DATA_PATH = "data/private.jsonl"
DEFAULT_OUTPUT_PATH = "results/submission_SFT.csv"

MAX_LENGTH = 16384
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.05
DO_SAMPLE = True

THINK_CLOSE_STR = "\n</think>\n"


SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Think concisely — identify the key insight only. "
    "Do NOT write out long derivations. "
    "After thinking, your ENTIRE response must be a single line: \\boxed{X} "
    "where X is the letter of the correct option. Nothing else."
)

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Think concisely — identify the approach, do key calculations, then state the answer. "
    "Keep your reasoning under 300 words. "
    "Do NOT round your answer — give the full decimal precision. "
    "Put your final answer inside \\boxed{}. "
    "Your response MUST end with \\boxed{answer}."
)

SYSTEM_PROMPT_MULTI = (
    "You are an expert mathematician. "
    "Think concisely — identify the approach, do key calculations, then state the answers. "
    "Keep your reasoning under 400 words. "
    "Do NOT round your answers — give full decimal precision. "
    "The problem has multiple answers corresponding to each [ANS] placeholder in order. "
    "Put ALL answers inside a single \\boxed{} separated by commas, in the same order as [ANS] appears. "
    "Example: if there are 3 [ANS] placeholders, write \\boxed{ans1, ans2, ans3}. "
    "Your response MUST end with \\boxed{ans1, ans2, ...}."
)


def count_ans_placeholders(question: str) -> int:
    return question.count("[ANS]")


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{label}. {str(option).strip()}"
            for label, option in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    if count_ans_placeholders(question) > 1:
        return SYSTEM_PROMPT_MULTI, question

    return SYSTEM_PROMPT_MATH, question


def load_private_data(data_path: str) -> list[dict]:
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model():
    torch.manual_seed(42)

    print(f"Loading tokenizer from base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from HuggingFace Hub: {ADAPTER_ID}")
    llm = PeftModel.from_pretrained(base_model, ADAPTER_ID)

    print("Merging LoRA adapter.")
    llm = llm.merge_and_unload()
    llm.eval()

    return llm, tokenizer


def generate_one(model, tokenizer, item: dict) -> str:
    question = item["question"]
    options = item.get("options")

    is_mcq = bool(options)
    n_ans = count_ans_placeholders(question)

    if is_mcq:
        think_token_budget = 2000
        answer_token_budget = 1000
    elif n_ans > 1:
        think_token_budget = 3000
        answer_token_budget = 3000
    else:
        think_token_budget = 3000
        answer_token_budget = 2500

    system_prompt, user_prompt = build_prompt(question, options)

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    think_close_ids = tokenizer.encode(
        THINK_CLOSE_STR,
        add_special_tokens=False,
    )

    with torch.no_grad():
        think_output = model.generate(
            **inputs,
            max_new_tokens=think_token_budget,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    think_close_tensor = torch.tensor(
        [think_close_ids],
        dtype=torch.long,
        device=model.device,
    )

    pass2_input_ids = torch.cat(
        [think_output, think_close_tensor],
        dim=1,
    )
    pass2_attention = torch.ones_like(pass2_input_ids)

    with torch.no_grad():
        final_output = model.generate(
            input_ids=pass2_input_ids,
            attention_mask=pass2_attention,
            max_new_tokens=answer_token_budget,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    pass2_input_len = pass2_input_ids.shape[1]
    answer_tokens = final_output[0][pass2_input_len:]
    answer_text = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True,
    ).strip()

    think_input_len = inputs["input_ids"].shape[1]
    think_tokens = think_output[0][think_input_len:]
    think_text = tokenizer.decode(
        think_tokens,
        skip_special_tokens=True,
    ).strip()

    response = think_text + THINK_CLOSE_STR + answer_text

    del inputs
    del think_output
    del pass2_input_ids
    del pass2_attention
    del final_output

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response


def run_inference(
    data_path: str = DEFAULT_DATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    data = load_private_data(data_path)

    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    n_multi = sum(
        not d.get("options") and count_ans_placeholders(d["question"]) > 1
        for d in data
    )

    print(
        f"Loaded {len(data)} questions "
        f"({n_mcq} MCQ, {n_free} free-form, {n_multi} multi-answer free-form)"
    )

    model, tokenizer = load_model()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for item in tqdm(data, desc="Generating"):
        response = generate_one(model, tokenizer, item)

        rows.append(
            {
                "id": item["id"],
                "response": response,
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "response"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    run_inference()