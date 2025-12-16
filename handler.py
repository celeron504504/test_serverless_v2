import runpod
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, login

MODEL_NAME = "SURENKUMAAR/deepseek-msu-chatbot-v1"
CACHE_DIR = "./hf_cache"

# -------------------------------------------------
# CUDA CHECK
# -------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available. Make sure GPU is enabled.")

DEVICE = torch.device("cuda")
print(f"🚀 Using device: {DEVICE}")

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
    print("\n🔍 Checking cache / downloading model...\n")

    model_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=CACHE_DIR,
        resume_download=True,
        local_files_only=False,
        token=HF_TOKEN
    )

    print("\n✅ Model ready (cached or downloaded)")
    return model_path


def load_model(model_path):
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📦 Loading model to CUDA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        token=HF_TOKEN
    ).to(DEVICE)

    model.eval()

    print("🚀 Model loaded on CUDA successfully")
    return tokenizer, model


def ask_question(model, tokenizer, question, max_new_tokens=256):
    prompt = f"User: {question}\nAssistant:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    return response


# ---------------- MAIN ----------------
model_path = download_model_if_needed()
tokenizer, model = load_model(model_path)


def handler(event):
    # RunPod async payload structure
    question = event["input"].get("prompt", "")

    if not question:
        return {"error": "No prompt provided"}

    answer = ask_question(model, tokenizer, question)
    return {"message": answer}
