import runpod
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, login

MODEL_NAME = "SURENKUMAAR/deepseek-msu-chatbot-v1"
CACHE_DIR = "./hf_cache"

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

print("🔍 Downloading model...")
model_path = snapshot_download(
    repo_id=MODEL_NAME,
    cache_dir=CACHE_DIR,
    token=HF_TOKEN
)

print("📦 Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=HF_TOKEN
)

print("🚀 Model ready")

def ask_question(question, max_new_tokens=256):
    prompt = f"User: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

def handler(event):
    prompt = event["input"]["prompt"]
    answer = ask_question(prompt)
    return {"output": answer}

runpod.serverless.start({"handler": handler})
