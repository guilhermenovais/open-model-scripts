"""Round-to-nearest W4A16 (int4 weights, group 128, symmetric) quant of
Salesforce/Llama-xLAM-2-8b-fc-r with llm-compressor, run on CPU (no calibration data).
vLLM 0.29 dropped bitsandbytes and no GPTQ/AWQ checkpoint of this model exists; bf16
(~16GB) doesn't fit the 12GB GPU. Output: models/Llama-xLAM-2-8b-fc-r-W4A16 (compressed-
tensors, served with vLLM's Marlin kernels like the RedHatAI Granite W4A16 quant)."""

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "Salesforce/Llama-xLAM-2-8b-fc-r"
OUT = "models/Llama-xLAM-2-8b-fc-r-W4A16"

model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(SRC)
oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]))
model.save_pretrained(OUT, save_compressed=True)
tokenizer.save_pretrained(OUT)
print("saved", OUT)
