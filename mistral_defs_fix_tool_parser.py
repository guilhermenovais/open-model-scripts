"""vLLM tool-parser plugin: the `mistral` parser, with $defs hoisted in its guided schema.

For tool_choice="required" (which pydantic-ai sends, since the agent's final answer is
the `final_result` tool), vLLM 0.29's Mistral parser builds a guided JSON schema that
nests each tool's `parameters` under `anyOf`. A tool schema with `$defs` (pydantic
models, e.g. the agent's query_rows / aggregate_rows filters) keeps its `$defs` nested
there, but its `{"$ref": "#/$defs/X"}` pointers are resolved from the schema root, so
every request failed with "Grammar error: Pointer '/$defs/EqualsCondition' does not
exist". This moves all nested `$defs` to the root, where the pointers expect them.

Used by run_ministral_8b.sh via --tool-parser-plugin + --tool-call-parser mistral_defs_fix.
"""

from typing import Any

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.mistral_tool_parser import MistralToolParser


def _hoist_defs(node: Any, root_defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        nested = node.pop("$defs", None)
        if nested:
            for name, definition in nested.items():
                existing = root_defs.get(name)
                if existing is not None and existing != definition:
                    raise ValueError(f"Conflicting $defs entry {name!r} across tools")
                root_defs[name] = definition
        for value in node.values():
            _hoist_defs(value, root_defs)
    elif isinstance(node, list):
        for item in node:
            _hoist_defs(item, root_defs)
    return node


@ToolParserManager.register_module("mistral_defs_fix")
class MistralDefsFixToolParser(MistralToolParser):
    def adjust_request(self, request):
        request = super().adjust_request(request)
        structured = getattr(request, "structured_outputs", None)
        schema = getattr(structured, "json", None) if structured is not None else None
        if isinstance(schema, dict):
            root_defs: dict[str, Any] = dict(schema.pop("$defs", None) or {})
            _hoist_defs(schema, root_defs)
            # Definitions can themselves contain $defs (nested pydantic models).
            while any(isinstance(d, dict) and "$defs" in d for d in root_defs.values()):
                for definition in list(root_defs.values()):
                    _hoist_defs(definition, root_defs)
            if root_defs:
                schema["$defs"] = root_defs
        return request
