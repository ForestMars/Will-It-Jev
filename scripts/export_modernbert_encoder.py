#!/usr/bin/env python3
"""Export a ModernBERT encoder that returns one pooled CLS vector per prompt.

The current checked-in model_int8.onnx is a fused binary route classifier. This
exporter creates the representation-only encoder.onnx used by independent
task heads. It deliberately does not export a classification layer.

Example:
    uv run --with torch --with transformers scripts/export_modernbert_encoder.py \
        --checkpoint answerdotai/ModernBERT-base \
        --output models/modernbert-base/encoder.onnx \
        --tokenizer-output models/modernbert-base/tokenizer.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


class PooledModernBERT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--output", default="models/modernbert-base/encoder.onnx")
    parser.add_argument("--tokenizer-output", default="models/modernbert-base/tokenizer.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if args.max_length < 2:
        parser.error("--max-length must be at least 2")

    model = AutoModel.from_pretrained(args.checkpoint)
    model.eval()
    wrapper = PooledModernBERT(model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)

    input_ids = torch.ones((1, min(16, args.max_length)), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(output),
            input_names=["input_ids", "attention_mask"],
            output_names=["pooled_cls"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "pooled_cls": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    tokenizer_path = Path(args.tokenizer_output)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.backend_tokenizer.save(str(tokenizer_path))
    print(f"wrote pooled encoder: {output}")
    print(f"wrote tokenizer: {tokenizer_path}")


if __name__ == "__main__":
    main()
