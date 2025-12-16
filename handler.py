import runpod
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, login

MODEL_NAME = "SURENKUMAAR/deepseek-msu-chatbot-v1"
CACHE_DIR = "./hf_cache"

# -------------------------------------------------
# Hugging Face Authentication
# -------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("🔐 Hugging Face token loaded")
else:
    print("⚠️ No Hugging Face token found (public models only)")


def download_model_if_needed():
    print("🔍 Checking / downloading model...")
    return snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=CACHE_DIR,
        resume_download=True,
        token=HF_TOKEN
    )


def load_model(model_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    print("🚀 Using CUDA")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)

    model.eval()
    return tokenizer, model, device


print("⏳ Loading model at startup...")
model_path = download_model_if_needed()
tokenizer, model, DEVICE = load_model(model_path)



def ask_question(question, max_new_tokens=256):
    prompt = f"User: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

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
    print("EVENT:", event)

    question = event["input"].get("prompt", "")
    if not question:
        return {"error": "No prompt provided"}

    return {"message": ask_question(question)}



runpod.serverless.start({"handler": handler})
