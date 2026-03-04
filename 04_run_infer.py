# run_infer.py
# by token
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from transformers import TextIteratorStreamer
from threading import Thread
import time

BASE_MODEL = r"E:/Python/Projects/Sereia/qwen2-7b-instruct"
ADAPTER_DIR = r"E:/Python/Projects/Sereia/qwen2-lora/adapters/qwen2-lora-Yunmo-0303t"

USE_4BIT = True
MAX_CONTEXT_TOKENS = 2048

GEN_MAX_NEW_TOKENS = 256
TEMPERATURE = 0.85
TOP_P = 0.85

ID_TEMPERATURE = 0.2
ID_TOP_P = 0.9
ID_MAX_NEW_TOKENS = 32

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "你的名字是。\n"
    )
}

_ID_PAT = re.compile(
    r"(你是谁|你叫(什么|啥)|你的名字|怎么称呼你|自我介绍|介绍一下你|who\s*are\s*you|what('?s| is)\s*your\s*name)",
    re.I
)
def is_identity_query(t: str) -> bool:
    return _ID_PAT.search(t.strip()) is not None

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

print("加载基础模型...")
kwargs = {"trust_remote_code": True}
if bnb_config:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=bnb_config, **kwargs)
else:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16, **kwargs)

if os.path.isdir(ADAPTER_DIR):
    print("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, device_map="auto", local_files_only=True)
    print("adapter OK:", ADAPTER_DIR)
else:
    print("警告：未找到 adapter，使用 base。")

model.eval()

def build_prompt_with_truncation(body_history, max_tokens=MAX_CONTEXT_TOKENS):
    def render(msgs):
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(p, return_tensors="pt", truncation=False)["input_ids"]
        return p, ids.size(1)

    msgs_full = [SYSTEM_MSG] + body_history
    prompt, n = render(msgs_full)
    if n <= max_tokens:
        return prompt

    for s in range(len(body_history)):
        prompt, n = render([SYSTEM_MSG] + body_history[s:])
        if n <= max_tokens:
            return prompt

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)

print("\n======\nwaken up\n")
history = []  # 仅 user/assistant

while True:
    try:
        user_text = input("我： ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nOver")
        break
    if not user_text:
        continue
    if user_text.lower() in ("exit", "quit"):
        print("Over")
        break
    if user_text.lower() == "/reset":
        history.clear()
        print("（已清空历史）\n")
        continue

    history.append({"role": "user", "content": user_text})

    prompt = build_prompt_with_truncation(history, MAX_CONTEXT_TOKENS)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    if is_identity_query(user_text):
        temperature, top_p, max_new = ID_TEMPERATURE, ID_TOP_P, ID_MAX_NEW_TOKENS
    else:
        temperature, top_p, max_new = TEMPERATURE, TOP_P, GEN_MAX_NEW_TOKENS

        streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        streamer=streamer
    )

    print("她： ", end="", flush=True)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    resp = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        resp += new_text

    print()  # 换行

    ctrl = input("\n ").strip().lower()
    if ctrl == "in":
        history.append({"role": "assistant", "content": resp})
        print("read in\n")
    else:
        # 默认 skip：把 user 也撤掉，保证“试采样不改变下一轮”
        if history and history[-1]["role"] == "user":
            history.pop()
        print("skip\n")
