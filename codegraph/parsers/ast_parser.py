"""
codegraph/parsers/ast_parser.py

Converts raw source code into structured ModuleNode objects using tree-sitter.

Design
------
- One ASTParser per supported language; a ParserRegistry dispatches by extension.
- Each parser extracts:
    * Module docstring
    * Classes  (name, bases, decorators, docstring, attributes, methods)
    * Functions / methods (name, params+types, return type, decorators,
                           docstring, source snippet, raw call list)
    * Import map  {alias → resolved_module}
- "raw_calls" are best-effort identifiers found in function bodies.
  The GraphBuilder does the resolution pass (cross-file) later.
"""

from __future__ import annotations

import logging
import re
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from codegraph.models import (
    ClassNode, FunctionNode, Language, ModuleNode, Param,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tree-sitter helpers
# ---------------------------------------------------------------------------

try:
    from tree_sitter_languages import get_language, get_parser as _get_ts_parser
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False
    log.warning("tree-sitter-languages not installed — falling back to regex parser")


def _ts_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _child_by_type(node, *types):
    for child in node.children:
        if child.type in types:
            return child
    return None


def _children_by_type(node, *types):
    return [c for c in node.children if c.type in types]


# ---------------------------------------------------------------------------
# Base parser interface
# ---------------------------------------------------------------------------

class BaseASTParser(ABC):
    language: Language = Language.UNKNOWN

    @abstractmethod
    def parse(self, source: str, module_path: str, repo: str) -> ModuleNode:
        """Parse source code and return a populated ModuleNode."""


# ---------------------------------------------------------------------------
# Python parser (tree-sitter)
# ---------------------------------------------------------------------------

class PythonASTParser(BaseASTParser):
    language = Language.PYTHON

    def parse(self, source: str, module_path: str, repo: str) -> ModuleNode:
        module = ModuleNode(path=module_path, repo=repo, language=Language.PYTHON)
        if not _TS_AVAILABLE:
            return self._regex_fallback(source, module)

        src    = source.encode("utf-8")
        parser = _get_ts_parser("python")
        tree   = parser.parse(src)
        root   = tree.root_node

        module.docstring = self._extract_module_docstring(root, src)
        module.imports   = self._extract_imports(root, src, module_path)

        for node in root.children:
            if node.type == "class_definition":
                cls = self._parse_class(node, src, module_path, repo)
                if cls:
                    module.classes.append(cls)
            elif node.type in ("function_definition", "decorated_definition"):
                fn = self._parse_function(node, src, module_path, repo, is_method=False)
                if fn:
                    module.functions.append(fn)

        return module

    # ------ class ------

    def _parse_class(self, node, src: bytes, module_path: str, repo: str) -> Optional[ClassNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = _ts_text(name_node, src)
        qname = f"{module_path}::{name}"

        cls = ClassNode(
            name=name,
            qualified_name=qname,
            module_path=module_path,
            repo=repo,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )

        # decorators
        cls.decorators = self._extract_decorators_before(node, src)

        # bases
        args_node = node.child_by_field_name("superclasses")
        if args_node:
            for child in args_node.children:
                if child.type in ("identifier", "attribute"):
                    cls.bases.append(_ts_text(child, src))

        # body
        body = node.child_by_field_name("body")
        if body:
            cls.docstring = self._extract_docstring_from_body(body, src)
            for child in body.children:
                if child.type in ("function_definition", "decorated_definition"):
                    fn = self._parse_function(child, src, module_path, repo, is_method=True)
                    if fn:
                        fn.qualified_name = f"{qname}.{fn.name}"
                        cls.methods.append(fn)
                elif child.type == "expression_statement":
                    # class-level type-annotated assignments: attr: Type = ...
                    pass
            # attribute type hints via assignment statements
            cls.attributes = self._extract_class_attributes(body, src)

        return cls

    # ------ function ------

    def _parse_function(self, node, src: bytes, module_path: str, repo: str, is_method: bool) -> Optional[FunctionNode]:
        # unwrap decorated_definition
        decorators = []
        actual = node
        if node.type == "decorated_definition":
            decorators = [_ts_text(d, src) for d in _children_by_type(node, "decorator")]
            actual = _child_by_type(node, "function_definition", "async_function_definition") or node

        is_async = actual.type == "async_function_definition" or (
            actual.children and actual.children[0].type == "async"
        )

        name_node = actual.child_by_field_name("name")
        if not name_node:
            return None
        name = _ts_text(name_node, src)

        # params
        params = self._extract_params(actual, src)

        # return type
        ret_node = actual.child_by_field_name("return_type")
        return_type = _ts_text(ret_node, src).lstrip("->").strip() if ret_node else None

        # body
        body = actual.child_by_field_name("body")
        docstring = self._extract_docstring_from_body(body, src) if body else None

        # source snippet (trim indent)
        snippet = textwrap.dedent(_ts_text(actual, src))[:1500]  # cap at 1500 chars

        # raw calls inside body
        raw_calls = self._extract_calls(actual, src)

        fn = FunctionNode(
            name=name,
            qualified_name=f"{module_path}::{name}",
            module_path=module_path,
            repo=repo,
            start_line=actual.start_point[0] + 1,
            end_line=actual.end_point[0] + 1,
            params=params,
            return_type=return_type,
            is_method=is_method,
            is_async=is_async,
            decorators=decorators,
            docstring=docstring,
            source_snippet=snippet,
            raw_calls=raw_calls,
        )
        return fn

    # ------ helpers ------

    def _extract_module_docstring(self, root, src: bytes) -> Optional[str]:
        for child in root.children:
            if child.type == "expression_statement":
                inner = child.children[0] if child.children else None
                if inner and inner.type == "string":
                    return _ts_text(inner, src).strip("\"'").strip()
        return None

    def _extract_docstring_from_body(self, body, src: bytes) -> Optional[str]:
        for child in body.children:
            if child.type == "expression_statement":
                inner = child.children[0] if child.children else None
                if inner and inner.type == "string":
                    raw = _ts_text(inner, src)
                    # strip triple quotes
                    for q in ('"""', "'''", '"', "'"):
                        if raw.startswith(q) and raw.endswith(q) and len(raw) > len(q):
                            return raw[len(q):-len(q)].strip()
                    return raw.strip()
        return None

    def _extract_params(self, fn_node, src: bytes) -> list[Param]:
        params_node = fn_node.child_by_field_name("parameters")
        if not params_node:
            return []
        result = []
        for child in params_node.children:
            if child.type == "identifier":
                result.append(Param(name=_ts_text(child, src)))
            elif child.type == "typed_parameter":
                pname = _ts_text(child.children[0], src) if child.children else "?"
                ptype = None
                type_node = child.child_by_field_name("type")
                if type_node:
                    ptype = _ts_text(type_node, src)
                result.append(Param(name=pname, type_hint=ptype))
            elif child.type == "default_parameter":
                pname_node = child.child_by_field_name("name")
                pval_node  = child.child_by_field_name("value")
                pname = _ts_text(pname_node, src) if pname_node else "?"
                pval  = _ts_text(pval_node,  src) if pval_node  else None
                result.append(Param(name=pname, default=pval))
            elif child.type == "typed_default_parameter":
                pname_node = child.child_by_field_name("name")
                ptype_node = child.child_by_field_name("type")
                pval_node  = child.child_by_field_name("value")
                result.append(Param(
                    name=_ts_text(pname_node, src) if pname_node else "?",
                    type_hint=_ts_text(ptype_node, src) if ptype_node else None,
                    default=_ts_text(pval_node, src) if pval_node else None,
                ))
        # skip 'self', 'cls'
        return [p for p in result if p.name not in ("self", "cls", "*", "**")]

    def _extract_calls(self, fn_node, src: bytes) -> list[str]:
        """Walk all call_expression nodes inside a function body."""
        calls = []

        def walk(node):
            if node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    calls.append(_ts_text(func_node, src))
            for child in node.children:
                walk(child)

        walk(fn_node)
        return list(set(calls))

    def _extract_imports(self, root, src: bytes, module_path: str) -> dict[str, str]:
        """Return {alias_or_name: full_module_path}."""
        result: dict[str, str] = {}
        module_dir = str(Path(module_path).parent)

        for node in root.children:
            if node.type == "import_statement":
                # import a, b.c as d
                for child in node.children:
                    if child.type == "dotted_name":
                        name = _ts_text(child, src)
                        result[name.split(".")[-1]] = name
                    elif child.type == "aliased_import":
                        alias_node = child.child_by_field_name("alias")
                        name_node  = child.child_by_field_name("name")
                        if alias_node and name_node:
                            result[_ts_text(alias_node, src)] = _ts_text(name_node, src)

            elif node.type == "import_from_statement":
                mod_node = node.child_by_field_name("module_name")
                mod_name = _ts_text(mod_node, src) if mod_node else ""
                # resolve relative imports
                if mod_name.startswith("."):
                    dots = len(mod_name) - len(mod_name.lstrip("."))
                    rel  = mod_name.lstrip(".")
                    parts = module_dir.split("/")
                    parts = parts[:max(0, len(parts) - dots + 1)]
                    if rel:
                        parts.append(rel.replace(".", "/"))
                    mod_name = "/".join(parts)
                else:
                    mod_name = mod_name.replace(".", "/")
                for child in node.children:
                    if child.type == "dotted_name" and child != mod_node:
                        sym = _ts_text(child, src)
                        result[sym] = f"{mod_name}/{sym}"
                    elif child.type == "aliased_import":
                        alias = child.child_by_field_name("alias")
                        name  = child.child_by_field_name("name")
                        if alias and name:
                            result[_ts_text(alias, src)] = f"{mod_name}/{_ts_text(name, src)}"
                    elif child.type == "wildcard_import":
                        result["*"] = mod_name
        return result

    def _extract_decorators_before(self, node, src: bytes) -> list[str]:
        """For decorated_definition siblings already captured above."""
        return []

    def _extract_class_attributes(self, body, src: bytes) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for child in body.children:
            if child.type in ("expression_statement",):
                inner = child.children[0] if child.children else None
                if inner and inner.type == "assignment":
                    left  = inner.child_by_field_name("left")
                    right = inner.child_by_field_name("right")
                    # look for "name: Type" patterns
                if inner and inner.type == "typed_parameter":
                    name_n = inner.children[0] if inner.children else None
                    type_n = inner.child_by_field_name("type")
                    if name_n and type_n:
                        attrs[_ts_text(name_n, src)] = _ts_text(type_n, src)
        return attrs

    def _regex_fallback(self, source: str, module: ModuleNode) -> ModuleNode:
        """Very basic regex extraction when tree-sitter is unavailable."""
        for m in re.finditer(r"^def\s+(\w+)\s*\(([^)]*)\)", source, re.M):
            fn = FunctionNode(
                name=m.group(1), qualified_name=f"{module.path}::{m.group(1)}",
                module_path=module.path, repo=module.repo,
                start_line=source[:m.start()].count("\n") + 1, end_line=0,
                source_snippet=m.group(0),
            )
            module.functions.append(fn)
        for m in re.finditer(r"^class\s+(\w+)", source, re.M):
            cls = ClassNode(
                name=m.group(1), qualified_name=f"{module.path}::{m.group(1)}",
                module_path=module.path, repo=module.repo,
                start_line=source[:m.start()].count("\n") + 1, end_line=0,
            )
            module.classes.append(cls)
        return module


# ---------------------------------------------------------------------------
# JavaScript / TypeScript parser (tree-sitter)
# ---------------------------------------------------------------------------

class JavaScriptASTParser(BaseASTParser):
    language = Language.JAVASCRIPT
    _ts_lang  = "javascript"

    def parse(self, source: str, module_path: str, repo: str) -> ModuleNode:
        module = ModuleNode(path=module_path, repo=repo, language=self.language)
        if not _TS_AVAILABLE:
            return module

        src    = source.encode("utf-8")
        parser = _get_ts_parser(self._ts_lang)
        tree   = parser.parse(src)
        root   = tree.root_node

        module.imports = self._extract_imports(root, src)

        def walk(node, parent_class: Optional[str] = None):
            if node.type == "class_declaration":
                cls = self._parse_class(node, src, module_path, repo)
                if cls:
                    module.classes.append(cls)
            elif node.type in ("function_declaration", "arrow_function",
                               "function_expression", "generator_function_declaration"):
                fn = self._parse_function(node, src, module_path, repo)
                if fn:
                    module.functions.append(fn)
            for child in node.children:
                walk(child, parent_class)

        walk(root)
        return module

    def _parse_class(self, node, src: bytes, module_path: str, repo: str) -> Optional[ClassNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name  = _ts_text(name_node, src)
        qname = f"{module_path}::{name}"
        cls   = ClassNode(
            name=name, qualified_name=qname, module_path=module_path, repo=repo,
            start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
        )
        heritage = node.child_by_field_name("heritage")
        if heritage:
            for child in heritage.children:
                if child.type == "identifier":
                    cls.bases.append(_ts_text(child, src))
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    fn = self._parse_method(child, src, module_path, repo)
                    if fn:
                        fn.qualified_name = f"{qname}.{fn.name}"
                        cls.methods.append(fn)
        return cls

    def _parse_method(self, node, src: bytes, module_path: str, repo: str) -> Optional[FunctionNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name   = _ts_text(name_node, src)
        params = self._extract_params(node, src)
        return FunctionNode(
            name=name, qualified_name=f"{module_path}::{name}",
            module_path=module_path, repo=repo,
            start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            params=params, is_method=True,
            source_snippet=textwrap.dedent(_ts_text(node, src))[:1500],
            raw_calls=self._extract_calls(node, src),
        )

    def _parse_function(self, node, src: bytes, module_path: str, repo: str) -> Optional[FunctionNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name   = _ts_text(name_node, src)
        params = self._extract_params(node, src)
        return FunctionNode(
            name=name, qualified_name=f"{module_path}::{name}",
            module_path=module_path, repo=repo,
            start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            params=params, is_method=False,
            source_snippet=textwrap.dedent(_ts_text(node, src))[:1500],
            raw_calls=self._extract_calls(node, src),
        )

    def _extract_params(self, node, src: bytes) -> list[Param]:
        params_node = node.child_by_field_name("parameters") or node.child_by_field_name("formal_parameters")
        if not params_node:
            return []
        result = []
        for child in params_node.children:
            if child.type == "identifier":
                result.append(Param(name=_ts_text(child, src)))
            elif child.type in ("assignment_pattern", "rest_element"):
                left = child.children[0] if child.children else None
                if left:
                    result.append(Param(name=_ts_text(left, src)))
        return result

    def _extract_calls(self, node, src: bytes) -> list[str]:
        calls = []
        def walk(n):
            if n.type == "call_expression":
                fn = n.child_by_field_name("function")
                if fn:
                    calls.append(_ts_text(fn, src))
            for c in n.children:
                walk(c)
        walk(node)
        return list(set(calls))

    def _extract_imports(self, root, src: bytes) -> dict[str, str]:
        result: dict[str, str] = {}
        for node in root.children:
            if node.type == "import_statement":
                src_node = node.child_by_field_name("source")
                mod = _ts_text(src_node, src).strip("'\"") if src_node else ""
                for child in node.children:
                    if child.type == "import_clause":
                        for sub in child.children:
                            if sub.type == "identifier":
                                result[_ts_text(sub, src)] = mod
                            elif sub.type == "named_imports":
                                for spec in sub.children:
                                    if spec.type == "import_specifier":
                                        alias_n = spec.child_by_field_name("alias")
                                        name_n  = spec.child_by_field_name("name")
                                        key = _ts_text(alias_n, src) if alias_n else (_ts_text(name_n, src) if name_n else "?")
                                        result[key] = mod
        return result


class TypeScriptASTParser(JavaScriptASTParser):
    language = Language.TYPESCRIPT
    _ts_lang  = "typescript"


# ---------------------------------------------------------------------------
# Generic / future parsers placeholder
# ---------------------------------------------------------------------------

class JavaASTParser(BaseASTParser):
    language = Language.JAVA

    def parse(self, source: str, module_path: str, repo: str) -> ModuleNode:
        module = ModuleNode(path=module_path, repo=repo, language=Language.JAVA)
        if not _TS_AVAILABLE:
            return module
        src    = source.encode("utf-8")
        parser = _get_ts_parser("java")
        tree   = parser.parse(src)
        root   = tree.root_node
        self._walk(root, src, module)
        return module

    def _walk(self, root, src, module: ModuleNode):
        for node in root.children:
            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if not name_node:
                    continue
                name  = _ts_text(name_node, src)
                qname = f"{module.path}::{name}"
                cls = ClassNode(
                    name=name, qualified_name=qname, module_path=module.path,
                    repo=module.repo, start_line=node.start_point[0]+1, end_line=node.end_point[0]+1,
                )
                super_node = node.child_by_field_name("superclass")
                if super_node:
                    cls.bases.append(_ts_text(super_node, src))
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        if child.type == "method_declaration":
                            fn_name_node = child.child_by_field_name("name")
                            if fn_name_node:
                                fn = FunctionNode(
                                    name=_ts_text(fn_name_node, src),
                                    qualified_name=f"{qname}.{_ts_text(fn_name_node, src)}",
                                    module_path=module.path, repo=module.repo,
                                    start_line=child.start_point[0]+1, end_line=child.end_point[0]+1,
                                    is_method=True, source_snippet=_ts_text(child, src)[:1500],
                                )
                                cls.methods.append(fn)
                module.classes.append(cls)


# ---------------------------------------------------------------------------
# Parser Registry
# ---------------------------------------------------------------------------

class ParserRegistry:
    """
    Maps file extensions to parser instances.
    Register custom parsers with `.register()`.
    """

    _EXT_DEFAULT = {
        ".py":   PythonASTParser,
        ".js":   JavaScriptASTParser,
        ".ts":   TypeScriptASTParser,
        ".java": JavaASTParser,
    }

    def __init__(self):
        self._parsers: dict[str, BaseASTParser] = {
            ext: cls() for ext, cls in self._EXT_DEFAULT.items()
        }

    def register(self, extension: str, parser: BaseASTParser):
        self._parsers[extension.lower()] = parser

    def get(self, extension: str) -> Optional[BaseASTParser]:
        return self._parsers.get(extension.lower())

    def parse(self, source: str, file_path: str, repo: str) -> Optional[ModuleNode]:
        ext    = Path(file_path).suffix.lower()
        parser = self.get(ext)
        if parser is None:
            log.debug("No parser for extension %s", ext)
            return None
        try:
            return parser.parse(source, file_path, repo)
        except Exception as e:
            log.warning("Parse error in %s: %s", file_path, e, exc_info=True)
            return None


# Convenience singleton
_default_registry: Optional[ParserRegistry] = None


def get_default_registry() -> ParserRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ParserRegistry()
    return _default_registry