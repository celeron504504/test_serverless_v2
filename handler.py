import runpod
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

MODEL_NAME = "SURENKUMAAR/deepseek-msu-chatbot-v1"
CACHE_DIR = "./hf_cache"

tokenizer = None
model = None
device = None


def load_model():
    global tokenizer, model, device

    if model is not None:
        return

    print("🔄 Loading model...")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")

    model_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=CACHE_DIR,
        resume_download=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to(device)

    model.eval()
    print("✅ Model loaded")


def handler(event):
    print("EVENT:", event)

    prompt = event["input"].get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided"}

    load_model()

    inputs = tokenizer(
        f"User: {prompt}\nAssistant:",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"message": response.split("Assistant:")[-1].strip()}


runpod.serverless.start({"handler": handler})
