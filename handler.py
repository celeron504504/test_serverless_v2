import runpod
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

    # ---- TOKENIZATION (DICT → TENSORS) ----
    enc = tokenizer(
        f"User: {prompt}\nAssistant:",
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ---- GENERATION ----
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    # ---- DECODE ONLY NEW TOKENS ----
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    #message = response["output"]["message"]
    #message = message.split("\n")[0]
    
    return {"message": response.strip()}
    #return {"message": message}


runpod.serverless.start({"handler": handler})
