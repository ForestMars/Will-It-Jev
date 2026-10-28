FROM python:3.12-slim

WORKDIR /app
COPY sidecars/common /app/sidecars/common
COPY sidecars/modernbert /app/sidecars/modernbert
COPY models/modernbert-base /app/models/modernbert-base

RUN pip install --no-cache-dir -r sidecars/modernbert/requirements.txt

ENV MODERNBERT_ONNX_PATH=/app/models/modernbert-base/model_int8.onnx
ENV MODERNBERT_TOKENIZER_PATH=/app/models/modernbert-base/tokenizer.json
EXPOSE 50052
CMD ["python", "sidecars/modernbert/server.py"]
