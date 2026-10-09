"""Builds models/ministral-8b-hf-tokenizer, the tokenizer run_ministral_8b.sh serves with.

It is the AWQ repo's HF tokenizer files without tekken.json: transformers 5 loads
MistralCommonBackend whenever tekken.json sits next to the weights (even with
--tokenizer-mode hf), and with that tokenizer vLLM 0.29 builds a tool-call grammar that
breaks on JSON-schema $defs. tokenizer_class is set to PreTrainedTokenizerFast because the
repo's LlamaTokenizerFast class ignores tokenizer.json's byte-level pre-tokenizer and decoder
(output came back as "ĠPrecisoĠdoĠnome..."). Token ids match tekken.json: checked on a
Portuguese sentence and the [AVAILABLE_TOOLS]/[TOOL_CALLS]/[INST] control tokens.

Run with ./venv/bin/python make_ministral_hf_tokenizer.py.
"""

import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "warshanks/Ministral-8B-Instruct-2410-AWQ"
FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]
OUT = Path(__file__).resolve().parent / "models" / "ministral-8b-hf-tokenizer"

src = Path(snapshot_download(REPO, allow_patterns=FILES))
OUT.mkdir(parents=True, exist_ok=True)
for name in FILES:
    shutil.copy(src / name, OUT / name)

config_path = OUT / "tokenizer_config.json"
config = json.loads(config_path.read_text())
config["tokenizer_class"] = "PreTrainedTokenizerFast"
config.pop("legacy", None)
config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
print("wrote", OUT)
