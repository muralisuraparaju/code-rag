"""
codegraph/models.py

Shared data-transfer objects. Every layer speaks these types so modules
stay decoupled from each other.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Language(str, Enum):
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA       = "java"
    GO         = "go"
    UNKNOWN    = "unknown"


class NodeKind(str, Enum):
    MODULE   = "Module"
    CLASS    = "Class"
    FUNCTION = "Function"


class EdgeKind(str, Enum):
    CONTAINS     = "CONTAINS"       # Module → Class | Function
    HAS_METHOD   = "HAS_METHOD"     # Class  → Function
    INHERITS     = "INHERITS"       # Class  → Class
    CALLS        = "CALLS"          # Function → Function
    IMPORTS      = "IMPORTS"        # Module → Module
    INSTANTIATES = "INSTANTIATES"   # Function → Class  (new ClassName())
    REFERENCES   = "REFERENCES"     # Class → Class     (used as type / field)


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------

@dataclass
class Param:
    """One function/method parameter."""
    name: str
    type_hint: Optional[str] = None
    default: Optional[str] = None


@dataclass
class FunctionNode:
    name: str
    qualified_name: str          # module.ClassName.method_name
    module_path: str
    repo: str
    start_line: int
    end_line: int
    params: list[Param]          = field(default_factory=list)
    return_type: Optional[str]   = None
    is_method: bool              = False
    is_async: bool               = False
    decorators: list[str]        = field(default_factory=list)
    docstring: Optional[str]     = None
    source_snippet: str          = ""   # filled by AST parser
    summary: Optional[str]       = None # filled by summarizer
    # raw call references found in the body
    raw_calls: list[str]         = field(default_factory=list)


@dataclass
class ClassNode:
    name: str
    qualified_name: str          # module.ClassName
    module_path: str
    repo: str
    start_line: int
    end_line: int
    bases: list[str]             = field(default_factory=list)  # raw base names
    decorators: list[str]        = field(default_factory=list)
    docstring: Optional[str]     = None
    methods: list[FunctionNode]  = field(default_factory=list)
    # class-level attribute type hints  {attr_name: type_str}
    attributes: dict[str, str]   = field(default_factory=dict)


@dataclass
class ModuleNode:
    path: str                    # relative path from repo root
    repo: str
    language: Language
    docstring: Optional[str]     = None
    classes: list[ClassNode]     = field(default_factory=list)
    functions: list[FunctionNode]= field(default_factory=list)
    # raw import strings  →  {alias_or_name: full_module_path}
    imports: dict[str, str]      = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Edge models
# ---------------------------------------------------------------------------

@dataclass
class CallEdge:
    """Directed edge: caller Function → callee Function (or placeholder)."""
    caller_qname: str
    callee_qname: str            # may be unresolved at first ("requests.get")
    args_repr: list[str]         = field(default_factory=list)  # textual arg exprs
    return_type: Optional[str]   = None                         # callee's return type


@dataclass
class InheritanceEdge:
    child_qname: str
    parent_qname: str            # may be unresolved placeholder


@dataclass
class ImportEdge:
    importer_path: str           # module path
    importee_path: str           # resolved module path (best-effort)
    names: list[str]             = field(default_factory=list)  # imported symbols