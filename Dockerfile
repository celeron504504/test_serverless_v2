FROM runpod/serverless:0.4.0

COPY handler.py /handler.py
COPY runpod.toml /runpod.toml

RUN pip install -r requirements.txt || true

CMD ["python", "-m", "runpod"]
