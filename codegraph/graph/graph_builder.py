"""
codegraph/graph/graph_builder.py

Transforms a stream of ModuleNode objects into a resolved graph of nodes
and edges, then hands them to a GraphWriter.

Compiler passes
---------------
1. Register      — add every Module / Class / Function to the symbol table
2. Link imports  — Module → Module IMPORTS edges
3. Link bases    — Class INHERITS edges; create placeholder nodes for unknown bases
4. Link calls    — Function CALLS edges; create placeholder Function nodes for
                   unresolved callees (external libs, not-yet-parsed files)
5. Summarize     — call the injected BaseSummarizer for each FunctionNode
6. Write         — flush everything to GraphWriter

The symbol table (`_registry`) is keyed by qualified name and persists across
files, so cross-file references are resolved as more modules are registered.
Placeholder nodes carry `stub=True` and are upgraded when the real definition
arrives.
"""

from __future__ import annotations

import logging
from typing import Optional

from codegraph.models import (
    CallEdge, ClassNode, EdgeKind, FunctionNode,
    InheritanceEdge, ImportEdge, ModuleNode,
)
from codegraph.summarizer.function_summarizer import BaseSummarizer, NoOpSummarizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol table entry
# ---------------------------------------------------------------------------

class _SymEntry:
    __slots__ = ("kind", "node", "stub")

    def __init__(self, kind: str, node, stub: bool = False):
        self.kind = kind   # "module" | "class" | "function"
        self.node = node
        self.stub = stub   # True = placeholder; upgrade when real def arrives


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Stateful builder.  Feed ModuleNode objects via `.add_module()`, then
    call `.flush()` to resolve cross-references and write everything.

    Usage
    -----
    builder = GraphBuilder(writer=Neo4jWriter(...), summarizer=AnthropicSummarizer(...))
    for file_entry in connector.iter_files():
        module = registry.parse(file_entry.content, file_entry.path, repo_name)
        if module:
            builder.add_module(module)
    builder.flush()
    """

    def __init__(
        self,
        writer,                          # GraphWriter instance (injected)
        summarizer: Optional[BaseSummarizer] = None,
        summarizer_workers: int = 4,
    ):
        self._writer     = writer
        self._summarizer = summarizer or NoOpSummarizer()
        self._workers    = summarizer_workers

        # qualified_name → _SymEntry
        self._registry: dict[str, _SymEntry] = {}

        # deferred edge lists (populated during add_module, resolved in flush)
        self._inheritance_edges: list[tuple[str, str]] = []   # (child_qname, base_raw)
        self._call_edges:        list[tuple[str, str, FunctionNode]] = []  # (caller, callee_raw, caller_fn)
        self._import_edges:      list[ImportEdge] = []

        # all function nodes awaiting summary
        self._pending_functions: list[FunctionNode] = []

        # all modules accumulated (for bulk write)
        self._modules: list[ModuleNode] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_module(self, module: ModuleNode):
        """Register a parsed module and queue its edges."""
        self._modules.append(module)

        # 1. Register module
        self._reg_module(module)

        # 2. Register classes + methods
        for cls in module.classes:
            self._reg_class(cls)
            for method in cls.methods:
                self._reg_function(method)
                self._pending_functions.append(method)
            # Defer inheritance resolution
            for base_raw in cls.bases:
                resolved = self._resolve_symbol(base_raw, module)
                self._inheritance_edges.append((cls.qualified_name, resolved))

        # 3. Register top-level functions
        for fn in module.functions:
            self._reg_function(fn)
            self._pending_functions.append(fn)

        # 4. Defer call edges
        all_fns = module.functions + [m for c in module.classes for m in c.methods]
        for fn in all_fns:
            for raw_call in fn.raw_calls:
                self._call_edges.append((fn.qualified_name, raw_call, fn))

        # 5. Import edges
        for alias, target_path in module.imports.items():
            self._import_edges.append(ImportEdge(
                importer_path=module.path,
                importee_path=target_path,
                names=[alias],
            ))

    def flush(self):
        """
        Resolve all deferred edges, generate summaries, and write to Neo4j.
        Call once after all modules have been added.
        """
        log.info("Summarizing %d functions...", len(self._pending_functions))
        self._summarizer.summarize_batch(self._pending_functions, workers=self._workers)

        # Flush summarizer cache
        if hasattr(self._summarizer, "flush"):
            self._summarizer.flush()

        log.info("Writing graph...")
        self._writer.begin()

        # Write all modules
        for mod in self._modules:
            self._writer.write_module(mod)

        # Resolve and write inheritance edges
        for child_qname, base_raw in self._inheritance_edges:
            parent_qname = self._ensure_class_placeholder(base_raw)
            self._writer.write_edge(EdgeKind.INHERITS, child_qname, parent_qname, {})

        # Resolve and write call edges
        for caller_qname, raw_call, caller_fn in self._call_edges:
            callee_qname = self._resolve_call(raw_call, caller_fn)
            if callee_qname:
                props = {
                    "args_repr":   str(caller_fn.params),
                    "return_type": caller_fn.return_type or "",
                }
                self._writer.write_edge(EdgeKind.CALLS, caller_qname, callee_qname, props)

        # Write import edges
        for imp in self._import_edges:
            # Ensure importee module node exists (may be external)
            if imp.importee_path not in self._registry:
                self._registry[imp.importee_path] = _SymEntry(
                    "module",
                    ModuleNode(path=imp.importee_path, repo="__external__",
                               language=__import__("codegraph.models", fromlist=["Language"]).Language.UNKNOWN),
                    stub=True,
                )
                self._writer.write_module_stub(imp.importee_path)
            self._writer.write_edge(
                EdgeKind.IMPORTS,
                imp.importer_path,
                imp.importee_path,
                {"names": ",".join(imp.names)},
            )

        self._writer.commit()
        log.info("Graph flush complete.")

    # ------------------------------------------------------------------
    # Symbol registration
    # ------------------------------------------------------------------

    def _reg_module(self, module: ModuleNode):
        key = module.path
        if key in self._registry and not self._registry[key].stub:
            return
        self._registry[key] = _SymEntry("module", module, stub=False)

    def _reg_class(self, cls: ClassNode):
        key = cls.qualified_name
        if key in self._registry and not self._registry[key].stub:
            return
        self._registry[key] = _SymEntry("class", cls, stub=False)
        # Also register by short name for best-effort resolution
        self._registry.setdefault(cls.name, _SymEntry("class", cls, stub=False))

    def _reg_function(self, fn: FunctionNode):
        key = fn.qualified_name
        if key in self._registry and not self._registry[key].stub:
            return
        self._registry[key] = _SymEntry("function", fn, stub=False)
        self._registry.setdefault(fn.name, _SymEntry("function", fn, stub=False))

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def _resolve_symbol(self, raw: str, context_module: ModuleNode) -> str:
        """Try to resolve a raw name to a qualified name via the import map."""
        # Direct hit
        if raw in self._registry:
            return raw
        # Via import alias in the context module
        imported = context_module.imports.get(raw)
        if imported and imported in self._registry:
            return imported
        # Qualified path e.g. "module.ClassName"
        if "." in raw:
            parts = raw.split(".")
            # Try trailing parts
            for i in range(len(parts)):
                candidate = "::".join(parts[i:])
                if candidate in self._registry:
                    return candidate
        # Give up: return raw as a synthetic qualified name
        return f"__unresolved__::{raw}"

    def _ensure_class_placeholder(self, qname: str) -> str:
        """Create a stub ClassNode if qname not yet in registry."""
        if qname not in self._registry:
            stub_cls = ClassNode(
                name=qname.split("::")[-1],
                qualified_name=qname,
                module_path="__stub__",
                repo="__stub__",
                start_line=0, end_line=0,
            )
            self._registry[qname] = _SymEntry("class", stub_cls, stub=True)
            self._writer.write_class_stub(qname)
        return qname

    def _resolve_call(self, raw_call: str, caller_fn: FunctionNode) -> Optional[str]:
        """
        Attempt to resolve a raw call expression to a known function qname.
        Returns None for noise (builtin calls like 'print', 'len', etc.).
        """
        _BUILTINS = {
            "print","len","range","str","int","float","list","dict","set","tuple",
            "isinstance","hasattr","getattr","setattr","super","type","enumerate",
            "zip","map","filter","sorted","reversed","open","repr","bool","abs",
            "min","max","sum","any","all","next","iter","vars","dir","id",
            "console.log","console.error","console.warn","Math.floor","Math.ceil",
        }
        if raw_call in _BUILTINS:
            return None

        # Direct qname match
        if raw_call in self._registry:
            return raw_call

        # "self.method_name" or "this.method_name"
        for prefix in ("self.", "this."):
            if raw_call.startswith(prefix):
                method_name = raw_call[len(prefix):]
                # search class context
                # caller's qname is "module::ClassName.method"
                parts = caller_fn.qualified_name.split("::")
                if len(parts) >= 2:
                    class_qname = "::".join(parts[:2])  # e.g. "src/foo.py::MyClass"
                    candidate = f"{class_qname}.{method_name}"
                    if candidate in self._registry:
                        return candidate

        # module-qualified  "module.function"
        if "." in raw_call:
            parts = raw_call.split(".")
            # Try as "module_alias::function"
            mod_alias = parts[0]
            fn_name   = parts[-1]
            imported_mod = caller_fn.module_path  # fallback
            if mod_alias in __import__("codegraph.models", fromlist=["ModuleNode"]):
                pass  # can't resolve without context here; use caller's module imports
            candidate = f"{imported_mod}::{fn_name}"
            if candidate in self._registry:
                return candidate

        # Short name match (last resort)
        if raw_call in self._registry:
            return raw_call

        # Unknown external — create a stub function node so we can at least
        # record the edge and fill it in later
        stub_qname = f"__external__::{raw_call}"
        if stub_qname not in self._registry:
            stub_fn = FunctionNode(
                name=raw_call,
                qualified_name=stub_qname,
                module_path="__external__",
                repo="__external__",
                start_line=0, end_line=0,
            )
            self._registry[stub_qname] = _SymEntry("function", stub_fn, stub=True)
            self._writer.write_function_stub(stub_qname)
        return stub_qname