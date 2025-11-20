import runpod

def handler(event):
    return {"message": "Hello from RunPod!", "input": event}

runpod.serverless.start({"handler": handler})
