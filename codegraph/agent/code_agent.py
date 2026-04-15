"""
codegraph/agent/code_agent.py

Conversational agent that uses CodeGraphRAG as its knowledge source.

Tools the agent has
-------------------
  graph_search(query)        — semantic + graph neighbourhood retrieval
  cypher_query(cypher)       — direct Cypher for power-users / structured questions
  find_function(name)        — locate a specific function by name
  find_class(name)           — locate a specific class by name
  get_call_chain(fn_name)    — trace the call chain from a function

The agent uses Anthropic tool_use (Claude) or OpenAI function_calling.
Both follow the same interface — AgentConfig.provider picks the backend.

Usage
-----
agent = CodeAgent.from_config(app_cfg)
response = agent.chat("How does the authentication flow work?")
print(response)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from codegraph.config import AgentConfig
from codegraph.graph.neo4j_writer import Neo4jWriter
from codegraph.rag.graph_rag import CodeGraphRAG

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (provider-agnostic dict, translated per backend)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "graph_search",
        "description": (
            "Search the code knowledge graph using a natural language query. "
            "Returns information about matching functions, classes, modules, "
            "and their relationships (call chains, inheritance, imports). "
            "Use this for broad or exploratory questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query about the codebase"},
                "hop_depth": {"type": "integer", "description": "Graph traversal depth (1-3, default 2)", "default": 2},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cypher_query",
        "description": (
            "Execute a raw Cypher query against the Neo4j code graph. "
            "Use for precise structural questions like counting nodes, "
            "finding specific inheritance chains, or listing all methods of a class. "
            "Node labels: Module, Class, Function. "
            "Relationships: CONTAINS, HAS_METHOD, INHERITS, CALLS, IMPORTS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cypher": {"type": "string", "description": "Valid Cypher query string"},
            },
            "required": ["cypher"],
        },
    },
    {
        "name": "find_function",
        "description": "Find a specific function or method by name and return its details including summary, parameters, return type, and callers/callees.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function or method name (exact or partial)"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "find_class",
        "description": "Find a specific class by name and return its methods, base classes, and which modules use it.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Class name (exact or partial)"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_call_chain",
        "description": "Trace the call chain starting from a function — who it calls and who calls it. Useful for understanding data flow and dependencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "function_name": {"type": "string", "description": "Starting function name"},
                "direction": {"type": "string", "enum": ["outgoing", "incoming", "both"], "default": "both"},
                "depth": {"type": "integer", "description": "How many hops to trace (1-5)", "default": 3},
            },
            "required": ["function_name"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class ToolExecutor:
    def __init__(self, rag: CodeGraphRAG, writer: Neo4jWriter):
        self._rag    = rag
        self._writer = writer

    def execute(self, tool_name: str, tool_input: dict) -> str:
        try:
            if tool_name == "graph_search":
                return self._graph_search(**tool_input)
            elif tool_name == "cypher_query":
                return self._cypher_query(**tool_input)
            elif tool_name == "find_function":
                return self._find_function(**tool_input)
            elif tool_name == "find_class":
                return self._find_class(**tool_input)
            elif tool_name == "get_call_chain":
                return self._get_call_chain(**tool_input)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            log.warning("Tool %s failed: %s", tool_name, e, exc_info=True)
            return f"Tool error: {e}"

    def _graph_search(self, query: str, hop_depth: int = 2) -> str:
        old_depth = self._rag._hop_depth
        self._rag._hop_depth = max(1, min(hop_depth, 3))
        result = self._rag.retrieve(query)
        self._rag._hop_depth = old_depth
        return result

    def _cypher_query(self, cypher: str) -> str:
        return self._rag.retrieve_by_cypher(cypher)

    def _find_function(self, name: str) -> str:
        rows = self._writer.query("""
            MATCH (f:Function)
            WHERE f.name CONTAINS $name OR f.qualified_name CONTAINS $name
            OPTIONAL MATCH (f)-[:CALLS]->(callee:Function)
            OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
            RETURN f.qualified_name   AS qname,
                   f.name             AS name,
                   f.module_path      AS module,
                   f.return_type      AS return_type,
                   f.params_json      AS params,
                   f.summary          AS summary,
                   f.docstring        AS docstring,
                   f.is_method        AS is_method,
                   f.is_async         AS is_async,
                   collect(DISTINCT callee.name) AS calls,
                   collect(DISTINCT caller.name) AS called_by
            LIMIT 10
        """, name=name)
        if not rows:
            return f"No function named '{name}' found."
        lines = [f"## Function: {name}\n"]
        for r in rows:
            params = self._fmt_params(r.get("params") or "[]")
            lines += [
                f"**{r['name']}**({params}) → {r.get('return_type') or 'None'}",
                f"Module: `{r.get('module', '')}`",
                f"{'async ' if r.get('is_async') else ''}{'method' if r.get('is_method') else 'function'}",
                f"Summary: {r.get('summary') or r.get('docstring') or 'N/A'}",
                f"Calls:     {', '.join(r.get('calls') or []) or 'none'}",
                f"Called by: {', '.join(r.get('called_by') or []) or 'none'}",
                "",
            ]
        return "\n".join(lines)

    def _find_class(self, name: str) -> str:
        rows = self._writer.query("""
            MATCH (c:Class)
            WHERE c.name CONTAINS $name OR c.qualified_name CONTAINS $name
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Function)
            OPTIONAL MATCH (c)-[:INHERITS]->(parent:Class)
            OPTIONAL MATCH (child:Class)-[:INHERITS]->(c)
            RETURN c.qualified_name AS qname,
                   c.name           AS name,
                   c.module_path    AS module,
                   c.docstring      AS docstring,
                   collect(DISTINCT m.name)      AS methods,
                   collect(DISTINCT parent.name) AS parents,
                   collect(DISTINCT child.name)  AS children
            LIMIT 5
        """, name=name)
        if not rows:
            return f"No class named '{name}' found."
        lines = [f"## Class: {name}\n"]
        for r in rows:
            lines += [
                f"**{r['name']}**  [`{r.get('module', '')}`]",
                f"Inherits from: {', '.join(r.get('parents') or []) or 'nothing'}",
                f"Subclassed by: {', '.join(r.get('children') or []) or 'nothing'}",
                f"Methods: {', '.join(r.get('methods') or []) or 'none'}",
                f"Doc: {(r.get('docstring') or 'N/A')[:300]}",
                "",
            ]
        return "\n".join(lines)

    def _get_call_chain(self, function_name: str, direction: str = "both", depth: int = 3) -> str:
        depth = max(1, min(depth, 5))
        results: list[str] = [f"## Call chain for `{function_name}` (depth={depth})\n"]

        if direction in ("outgoing", "both"):
            rows = self._writer.query(f"""
                MATCH (start:Function)
                WHERE start.name CONTAINS $name
                CALL apoc.path.spanningTree(start, {{
                    relationshipFilter: 'CALLS>',
                    maxLevel: {depth}
                }}) YIELD path
                RETURN [n IN nodes(path) | coalesce(n.name, n.path)] AS chain
                LIMIT 20
            """, name=function_name)
            if not rows:
                # APOC fallback
                rows = self._writer.query(f"""
                    MATCH path = (start:Function)-[:CALLS*1..{depth}]->(end:Function)
                    WHERE start.name CONTAINS $name
                    RETURN [n IN nodes(path) | n.name] AS chain
                    LIMIT 20
                """, name=function_name)
            if rows:
                results.append("### Outgoing calls (what this function calls)\n")
                for r in rows:
                    chain = r.get("chain") or []
                    results.append("  " + " → ".join(str(c) for c in chain))

        if direction in ("incoming", "both"):
            rows = self._writer.query(f"""
                MATCH path = (caller:Function)-[:CALLS*1..{depth}]->(target:Function)
                WHERE target.name CONTAINS $name
                RETURN [n IN nodes(path) | n.name] AS chain
                LIMIT 20
            """, name=function_name)
            if rows:
                results.append("\n### Incoming calls (who calls this function)\n")
                for r in rows:
                    chain = r.get("chain") or []
                    results.append("  " + " → ".join(str(c) for c in chain))

        return "\n".join(results) if len(results) > 1 else f"No call chain found for '{function_name}'."

    def _fmt_params(self, params_json: str) -> str:
        try:
            params = json.loads(params_json)
            return ", ".join(
                p["name"] + (f": {p['type']}" if p.get("type") else "")
                for p in params
            )
        except Exception:
            return "..."


# ---------------------------------------------------------------------------
# Base agent interface
# ---------------------------------------------------------------------------

class BaseAgent:
    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior software engineer assistant with access to a structured
        code knowledge graph built from a GitLab repository.

        You can answer questions about:
        - What specific functions and classes do
        - How modules depend on each other
        - Call chains and data flow
        - Inheritance hierarchies
        - Where specific functionality lives

        When answering:
        1. Always use your tools to fetch accurate information — do not guess.
        2. Cite the module path and function/class name when referring to code.
        3. If a question is ambiguous, use `find_function` or `find_class` first to get oriented.
        4. For flow questions, use `get_call_chain`.
        5. For structural/counting questions, use `cypher_query`.
        6. Keep answers precise and focused. Show relevant code relationships.
    """).strip()

    def __init__(self, cfg: AgentConfig, executor: ToolExecutor):
        self._cfg      = cfg
        self._executor = executor
        self._history: list[dict] = []

    def chat(self, user_message: str) -> str:
        raise NotImplementedError

    def reset(self):
        self._history = []


import textwrap


# ---------------------------------------------------------------------------
# Anthropic (Claude) agent
# ---------------------------------------------------------------------------

class AnthropicAgent(BaseAgent):
    def __init__(self, cfg: AgentConfig, executor: ToolExecutor):
        super().__init__(cfg, executor)
        try:
            import anthropic  # type: ignore
        except ImportError:
            raise ImportError("pip install anthropic")
        import anthropic
        import os
        self._client = anthropic.Anthropic(api_key=cfg.api_key or os.getenv("ANTHROPIC_API_KEY", ""))
        # Convert tools to Anthropic format
        self._tools = [
            {
                "name":         t["name"],
                "description":  t["description"],
                "input_schema": t["parameters"],
            }
            for t in TOOLS
        ]

    def chat(self, user_message: str) -> str:
        import anthropic
        self._history.append({"role": "user", "content": user_message})

        messages = list(self._history)
        final_text = ""

        # Agentic loop: keep going until no more tool calls
        for _ in range(10):  # max 10 rounds
            response = self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                system=self.SYSTEM_PROMPT,
                tools=self._tools,
                messages=messages,
            )

            # Collect assistant turn
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                for block in assistant_content:
                    if hasattr(block, "text"):
                        final_text = block.text
                break

            if response.stop_reason == "tool_use":
                # Execute all tool calls
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        log.info("Tool call: %s(%s)", block.name, list(block.input.keys()))
                        result = self._executor.execute(block.name, block.input)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result,
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        self._history.append({"role": "assistant", "content": final_text})
        return final_text


# ---------------------------------------------------------------------------
# OpenAI agent
# ---------------------------------------------------------------------------

class OpenAIAgent(BaseAgent):
    def __init__(self, cfg: AgentConfig, executor: ToolExecutor):
        super().__init__(cfg, executor)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise ImportError("pip install openai")
        import os
        from openai import OpenAI
        self._client = OpenAI(api_key=cfg.api_key or os.getenv("OPENAI_API_KEY", ""))
        # Convert to OpenAI function format
        self._functions = [
            {"type": "function", "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["parameters"],
            }}
            for t in TOOLS
        ]

    def chat(self, user_message: str) -> str:
        from openai.types.chat import ChatCompletionMessageToolCall  # type: ignore
        self._history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + list(self._history)
        final_text = ""

        for _ in range(10):
            response = self._client.chat.completions.create(
                model=self._cfg.model or "gpt-4o",
                max_tokens=self._cfg.max_tokens,
                tools=self._functions,
                messages=messages,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump())

            if response.choices[0].finish_reason == "stop":
                final_text = msg.content or ""
                break

            if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    log.info("Tool call: %s", tc.function.name)
                    args   = json.loads(tc.function.arguments)
                    result = self._executor.execute(tc.function.name, args)
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
            else:
                final_text = msg.content or ""
                break

        self._history.append({"role": "assistant", "content": final_text})
        return final_text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class CodeAgent:
    """
    Top-level factory.  Wires together everything and exposes `.chat()`.

    from codegraph.agent import CodeAgent
    from codegraph.config import AppConfig

    cfg   = AppConfig.from_env()
    agent = CodeAgent.from_config(cfg, writer, rag)
    print(agent.chat("Explain the authentication flow"))
    """

    @staticmethod
    def from_config(
        cfg:    Any,           # AppConfig
        writer: Neo4jWriter,
        rag:    CodeGraphRAG,
    ) -> BaseAgent:
        executor = ToolExecutor(rag=rag, writer=writer)
        provider = cfg.agent.provider.lower()
        if provider == "anthropic":
            return AnthropicAgent(cfg.agent, executor)
        elif provider == "openai":
            return OpenAIAgent(cfg.agent, executor)
        else:
            raise ValueError(f"Unknown agent provider: {provider!r}. Use 'anthropic' or 'openai'.")