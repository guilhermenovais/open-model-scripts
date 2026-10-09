"""Smoke test for a vLLM server's tool calling: `smoke_tool_call.py <served-model-name> [base_url]`.

Checks two things with tool_choice "auto":
  1. a question that needs the tool gets a parsed `tool_calls` entry (not raw text);
  2. a plain question gets a text answer.
Uses only the stdlib so it runs from any Python.
"""

import json
import sys
import urllib.error
import urllib.request

model = sys.argv[1]
base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_budget",
            "description": "Returns the budget (in BRL) of a Brazilian government agency for a given year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agency": {"type": "string", "description": "Agency name"},
                    "year": {"type": "integer"},
                },
                "required": ["agency", "year"],
            },
        },
    }
]


def chat(content: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["choices"][0]["message"]
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code}: {exc.read().decode()[:1000]}")


ok = True
msg = chat("Qual foi o orçamento da Agência Espacial Brasileira em 2020? Use a ferramenta.")
calls = msg.get("tool_calls") or []
if calls:
    fn = calls[0]["function"]
    print(f"[tool] OK: {fn['name']}({fn['arguments']})")
else:
    ok = False
    print(f"[tool] FAIL: no tool_calls; content={msg.get('content')!r}")

msg = chat("Quanto é 2 + 2? Responda diretamente.")
if msg.get("tool_calls"):
    print(f"[text] WARN: called a tool for a plain question: {msg['tool_calls']}")
else:
    print(f"[text] OK: {(msg.get('content') or '').strip()[:120]!r}")

sys.exit(0 if ok else 1)
