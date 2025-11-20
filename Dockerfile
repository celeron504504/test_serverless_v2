FROM runpod/serverless:latest

COPY handler.py /handler.py
COPY runpod.toml /runpod.toml

RUN pip install -r requirements.txt || true

CMD ["python", "-m", "runpod"]
