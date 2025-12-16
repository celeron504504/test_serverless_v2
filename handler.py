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
HF_TOKEN = os.getenv("HF_TOKEN")  # Read from environment variable

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
        token=HF_TOKEN   # <-- token used here
    )

    print("\n✅ Model ready (cached or downloaded)")
    return model_path


def load_model(model_path):
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=HF_TOKEN
    )

    print("📦 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=HF_TOKEN
    )

    print("🚀 Model loaded successfully")
    return tokenizer, model


def ask_question(model, tokenizer, question, max_new_tokens=256):
    prompt = f"User: {question}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
if __name__ == "__main__":
    model_path = download_model_if_needed()
    tokenizer, model = load_model(model_path)

    #print("\n💬 Chat started (type 'exit' to quit)")

    # while True:
    # question = input("\nYou: ")
    # if question.lower() in ["exit", "quit"]:
    # break

    # answer = ask_question(model, tokenizer, question)
    # print("Bot:", answer)

def handler(event):
    prompt = event["input"]["prompt"]
    answer = ask_question(model, tokenizer, prompt)

    return {"message": answer}









# -------------------------------------------------
#import runpod

#def handler(event):
    #return {"message": "Hello from RunPod!", "input": event}

#runpod.serverless.start({"handler": handler})
# -------------------------------------------------
