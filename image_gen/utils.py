"""Utility module for class and function manipulation and dynamic loading.

This module provides utilities for extracting class structure, generating source code,
and dynamically loading custom classes with safety measures.
"""

# Standard library imports
import ast
import builtins
import inspect
import re
from types import FunctionType
from typing import Dict, Optional, Any

# Third-party imports
import dill.source
import numpy as np
import torch
import torchvision
import typing
from tqdm.autonotebook import tqdm

# Local application imports
from .diffusion import (
    BaseDiffusion,
    SubVariancePreserving,
    VarianceExploding,
    VariancePreserving,
)
from .metrics import (
    BaseMetric,
    BitsPerDimension,
    FrechetInceptionDistance,
    InceptionScore,
)
from .noise import (
    BaseNoiseSchedule,
    CosineNoiseSchedule,
    LinearNoiseSchedule,
)
from .samplers import (
    BaseSampler,
    EulerMaruyama,
    ExponentialIntegrator,
    ODEProbabilityFlow,
    PredictorCorrector,
)


def _get_function_source(func: FunctionType) -> Optional[str]:
    """Gets function source code with decorators using dill.

    Args:
        func: The function to extract source code from.

    Returns:
        The source code as a string, or None if extraction fails.
    """
    try:
        while hasattr(func, '__wrapped__'):
            func = func.__wrapped__
        return dill.source.getsource(func)
    except (OSError, TypeError, AttributeError):
        return None


def _get_class_structure(cls: type) -> Dict:
    """Extracts class structure including magic methods and properties.

    Args:
        cls: The class to analyze.

    Returns:
        Dictionary containing the class structure with attributes, methods,
        and properties.

    Raises:
        TypeError: If the input is not a class.
    """
    if not inspect.isclass(cls):
        raise TypeError("Input must be a class")

    structure = {
        'name': cls.__name__,
        'bases': [base.__name__ for base in cls.__bases__ if base is not object],
        'attributes': {},
        'methods': {},
        'properties': {}
    }

    for name, attr in cls.__dict__.items():
        # Handle properties
        if isinstance(attr, property):
            prop_info = {}
            if attr.fget:
                prop_info['getter'] = _get_function_source(attr.fget)
            if attr.fset:
                prop_info['setter'] = _get_function_source(attr.fset)
            if attr.fdel:
                prop_info['deleter'] = _get_function_source(attr.fdel)
            structure['properties'][name] = prop_info

        # Handle methods (including magic methods)
        elif isinstance(attr, (classmethod, staticmethod)):
            func = attr.__func__
            source = _get_function_source(func)
            structure['methods'][name] = {
                'source': source,
                'type': type(attr).__name__
            }

        elif isinstance(attr, FunctionType):
            source = _get_function_source(attr)
            structure['methods'][name] = {
                'source': source,
                'type': 'method'
            }

        # Handle class attributes (skip special dunder attributes)
        else:
            if not (name.startswith('__') and name.endswith('__')) and not name.startswith('_abc_'):
                structure['attributes'][name] = attr

    return structure


def _generate_class_source(structure: Dict) -> str:
    """Generates class source code from structure.

    Args:
        structure: Dictionary containing the class structure.

    Returns:
        The generated source code as a string.
    """
    lines = []

    # Class definition
    bases = ', '.join(structure['bases'])
    class_def = f"class {structure['name']}"
    if bases:
        class_def += f"({bases})"
    lines.append(class_def + ":")

    # Class attributes
    for name, value in structure['attributes'].items():
        lines.append(f"    {name} = {repr(value)}")
    if len(structure['attributes']) > 0:
        lines.append("")

    # Properties
    for prop, info in structure['properties'].items():
        if info.get('getter'):
            lines.append(info['getter'])
        if info.get('setter'):
            lines.append(info['setter'])
        if info.get('deleter'):
            lines.append(info['deleter'])

    # Methods
    for name, method in structure['methods'].items():
        if method['source']:
            source = method['source']
            lines.append('\n'.join(['' + line for line in source.split('\n')]))

    return '\n'.join(lines)


def get_class_source(cls: type) -> str:
    """Gets the source code of a class, including its methods and properties.

    Args:
        cls: The class to extract source code from.

    Returns:
        The source code as a string.
    """
    structure = _get_class_structure(cls)
    return _generate_class_source(structure)


class _ClassRenamer(ast.NodeTransformer):
    """AST NodeTransformer for renaming classes."""

    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
        self.in_class = False

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Visits and renames class definitions."""
        if node.name == self.old_name:
            node.name = self.new_name
            self.in_class = True
        else:
            self.in_class = False

        node.bases = [self.visit(base) for base in node.bases]
        node.keywords = [self.visit(kw) for kw in node.keywords]

        original_in_class = self.in_class
        self.generic_visit(node)
        self.in_class = original_in_class
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Handles references in type annotations and other contexts."""
        if node.id == self.old_name and not self.in_class:
            return ast.Name(id=self.new_name, ctx=node.ctx)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Handles class attribute accesses."""
        if isinstance(node.value, ast.Name) and node.value.id == self.old_name:
            node.value = ast.Name(id=self.new_name, ctx=ast.Load())
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Handles class instantiations."""
        if isinstance(node.func, ast.Name) and node.func.id == self.old_name:
            node.func = ast.Name(id=self.new_name, ctx=ast.Load())
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Handles type annotations in arguments."""
        for arg in node.args.args:
            if arg.annotation:
                arg.annotation = self.visit(arg.annotation)
        if node.returns:
            node.returns = self.visit(node.returns)
        self.generic_visit(node)
        return node


def _rename_class(source_code: str, old_name: str, new_name: str) -> str:
    """Renames a class in the source code and adds a property to return the old name.

    Args:
        source_code: The source code containing the class.
        old_name: The original class name.
        new_name: The new class name.

    Returns:
        The modified source code as a string.
    """
    tree = ast.parse(source_code)
    transformer = _ClassRenamer(old_name, new_name)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    modified_code = ast.unparse(new_tree)
    modified_code += f"""\n
    @property
    def _class_name(self) -> str:
        return '{old_name}'"""
    return modified_code


class CustomClassWrapper:
    """Wrapper for dynamically loading and managing custom classes."""

    def __init__(self, code: str, *args: Any, class_name: Optional[str] = None, limit_globals: bool = True, **kwargs: Any):
        if class_name is None:
            pattern = r"class\s+(\w+)\s*\("

            match = re.search(pattern, code)
            if match:
                class_name = match.group(1)
            else:
                raise Exception(
                    "Class name not found in the code or specified manually.")

        self._original_class_name = class_name
        self._code = _rename_class(
            code, class_name, "CustomClass")
        self._cls = None
        self._args = args
        self._kwargs = kwargs
        self._limit_globals = limit_globals
        self.loaded = False

    def _load_class(self):
        if self.loaded:
            return

        # Also removed unnecessary builtins like `breakpoint`, `globals`, `locals`, `help`, `input`
        unsafe_builtins = {
            "__import__",
            "breakpoint",
            "eval",
            "exec",
            "globals",
            "help",
            "input",
            "locals",
            "open"
        }
        safe_builtins = {k: getattr(builtins, k) for k in dir(
            builtins) if k not in unsafe_builtins}

        allowed_globals = {
            "__builtins__": safe_builtins,
            "np": np,
            "torch": torch,
            "Tensor": torch.Tensor,
            "torchvision": torchvision,
            "transforms": torchvision.transforms,
            "ToTensor": torchvision.transforms.ToTensor,
            "Dataset": torch.utils.data.Dataset,
            "Subset": torch.utils.data.Subset,
            "DataLoader": torch.utils.data.DataLoader,
            "tqdm": tqdm,
            "typing": typing,
            "List": typing.List,
            "Tuple": typing.Tuple,
            "Dict": typing.Dict,
            "Callable": typing.Callable,
            "Optional": typing.Optional,
            "Union": typing.Union,
            "BaseDiffusion": BaseDiffusion,
            "VarianceExploding": VarianceExploding,
            "VariancePreserving": VariancePreserving,
            "SubVariancePreserving": SubVariancePreserving,
            "BaseSampler": BaseSampler,
            "EulerMaruyama": EulerMaruyama,
            "PredictorCorrector": PredictorCorrector,
            "ODEProbabilityFlow": ODEProbabilityFlow,
            "ExponentialIntegrator": ExponentialIntegrator,
            "BaseNoiseSchedule": BaseNoiseSchedule,
            "LinearNoiseSchedule": LinearNoiseSchedule,
            "CosineNoiseSchedule": CosineNoiseSchedule,
            "BaseMetric": BaseMetric,
            "BitsPerDimension": BitsPerDimension,
            "FrechetInceptionDistance": FrechetInceptionDistance,
            "InceptionScore": InceptionScore
        }

        # The loaded class will be stored in the `allowed_locals` dictionary to avoid polluting the global namespace
        allowed_locals = {}

        exec(self._code, allowed_globals if self._limit_globals else None, allowed_locals)
        self._cls = allowed_locals["CustomClass"](
            *self._args, **self._kwargs)
        self.loaded = True

    def __getattribute__(self, name: str):
        if name in ["_original_class_name", "_class_name", "_code", "_cls", "_args", "_kwargs", "_limit_globals", "loaded", "_load_class"]:
            return object.__getattribute__(self, name)

        if self._cls is not None:
            return getattr(self._cls, name)

        self._load_class()
        return getattr(self._cls, name)

    def __call__(self, *args, **kwargs):
        if not self.loaded:
            self._load_class()
        return self._cls(*args, **kwargs)

    @property
    def _class_name(self) -> str:
        return f"{self._original_class_name} (Custom Class)"
