# Internal infrastructure for ComfyAPI
from .api_registry import ComfyAPIBase as ComfyAPIBase

import asyncio
from dataclasses import asdict
from typing import Callable, Optional


def first_real_override(cls: type, name: str, *, base: type=None) -> Optional[Callable]:
    """Return the *callable* override of `name` visible on `cls`, or None if every
    implementation up to (and including) `base` is the placeholder defined on `base`.

    If base is not provided, it will assume cls has a GET_BASE_CLASS
    """
    if base is None:
        if not hasattr(cls, "GET_BASE_CLASS"):
            raise ValueError("base is required if cls does not have a GET_BASE_CLASS; is this a valid ComfyNode subclass?")
        base = cls.GET_BASE_CLASS()
    base_attr = getattr(base, name, None)
    if base_attr is None:
        return None
    base_func = base_attr.__func__
    for c in cls.mro():                       # NodeB, NodeA, ComfyNode, object …
        if c is base:                         # reached the placeholder – we're done
            break
        if name in c.__dict__:                # first class that *defines* the attr
            func = getattr(c, name).__func__
            if func is not base_func:         # real override
                return getattr(cls, name)     # bound to *cls*
    return None


class _ComfyNodeInternal:
    """Class that all V3-based APIs inherit from for ComfyNode.

    This is intended to only be referenced within execution.py, as it has to handle all V3 APIs going forward."""
    @classmethod
    def GET_NODE_INFO_V1(cls):
        ...


class _NodeOutputInternal:
    """Class that all V3-based APIs inherit from for NodeOutput.

    This is intended to only be referenced within execution.py, as it has to handle all V3 APIs going forward."""
    ...


def as_pruned_dict(dataclass_obj):
    '''Return dict of dataclass object with pruned None values.'''
    return prune_dict(asdict(dataclass_obj))

def prune_dict(d: dict):
    return {k: v for k,v in d.items() if v is not None}


def is_class(obj):
    '''
    Returns True if is a class type.
    Returns False if is a class instance.
    '''
    return isinstance(obj, type)


def copy_class(cls: type) -> type:
    '''
    Copy a class and its attributes.
    '''
    if cls is None:
        return None
    cls_dict = {
            k: v for k, v in cls.__dict__.items()
            if k not in ('__dict__', '__weakref__', '__module__', '__doc__')
        }
    # new class
    new_cls = type(
        cls.__name__,
        (cls,),
        cls_dict
    )
    # metadata preservation
    new_cls.__module__ = cls.__module__
    new_cls.__doc__ = cls.__doc__
    return new_cls


class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)


# NOTE: this was ai generated and validated by hand
def shallow_clone_class(cls, new_name=None):
    '''
    Shallow clone a class while preserving super() functionality.
    '''
    new_name = new_name or f"{cls.__name__}Clone"
    # Include the original class in the bases to maintain proper inheritance
    new_bases = (cls,) + cls.__bases__
    return type(new_name, new_bases, dict(cls.__dict__))

# NOTE: this was ai generated and validated by hand
def lock_class(cls):
    '''
    Lock a class so that its top-levelattributes cannot be modified.
    '''
    # Locked instance __setattr__
    def locked_instance_setattr(self, name, value):
        raise AttributeError(
            f"Cannot set attribute '{name}' on immutable instance of {type(self).__name__}"
        )
    # Locked metaclass
    class LockedMeta(type(cls)):
        def __setattr__(cls_, name, value):
            raise AttributeError(
                f"Cannot modify class attribute '{name}' on locked class '{cls_.__name__}'"
            )
    # Rebuild class with locked behavior
    locked_dict = dict(cls.__dict__)
    locked_dict['__setattr__'] = locked_instance_setattr

    return LockedMeta(cls.__name__, cls.__bases__, locked_dict)


def make_locked_method_func(type_obj, func, class_clone):
    """
    Returns a function that, when called with **inputs, will execute:
    getattr(type_obj, func).__func__(lock_class(class_clone), **inputs)

    Supports both synchronous and asynchronous methods.
    """
    locked_class = lock_class(class_clone)
    method = getattr(type_obj, func).__func__

    # Check if the original method is async
    if asyncio.iscoroutinefunction(method):
        async def wrapped_async_func(**inputs):
            return await method(locked_class, **inputs)
        return wrapped_async_func
    else:
        def wrapped_func(**inputs):
            return method(locked_class, **inputs)
        return wrapped_func
