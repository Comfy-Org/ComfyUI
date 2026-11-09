import asyncio
import concurrent.futures
import contextvars
import functools
import inspect
import logging
import os
import textwrap
import threading
from enum import Enum
from typing import Optional, get_origin, get_args, get_type_hints


class TypeTracker:
    """Tracks types discovered during stub generation for automatic import generation."""

    def __init__(self):
        self.discovered_types = {}  # type_name -> (module, qualname)
        self.builtin_types = {
            "Any",
            "Dict",
            "List",
            "Optional",
            "Tuple",
            "Union",
            "Set",
            "Sequence",
            "cast",
            "NamedTuple",
            "str",
            "int",
            "float",
            "bool",
            "None",
            "bytes",
            "object",
            "type",
            "dict",
            "list",
            "tuple",
            "set",
        }
        self.already_imported = (
            set()
        )  # Track types already imported to avoid duplicates

    def track_type(self, annotation):
        """Track a type annotation and record its module/import info."""
        if annotation is None or annotation is type(None):
            return

        # Skip builtins and typing module types we already import
        type_name = getattr(annotation, "__name__", None)
        if type_name and (
            type_name in self.builtin_types or type_name in self.already_imported
        ):
            return

        # Get module and qualname
        module = getattr(annotation, "__module__", None)
        qualname = getattr(annotation, "__qualname__", type_name or "")

        # Skip types from typing module (they're already imported)
        if module == "typing":
            return

        # Skip UnionType and GenericAlias from types module as they're handled specially
        if module == "types" and type_name in ("UnionType", "GenericAlias"):
            return

        if module and module not in ["builtins", "__main__"]:
            # Store the type info
            if type_name:
                self.discovered_types[type_name] = (module, qualname)

    def get_imports(self, main_module_name: str) -> list[str]:
        """Generate import statements for all discovered types."""
        imports = []
        imports_by_module = {}

        for type_name, (module, qualname) in sorted(self.discovered_types.items()):
            # Skip types from the main module (they're already imported)
            if main_module_name and module == main_module_name:
                continue

            if module not in imports_by_module:
                imports_by_module[module] = []
            if type_name not in imports_by_module[module]:  # Avoid duplicates
                imports_by_module[module].append(type_name)

        # Generate import statements
        for module, types in sorted(imports_by_module.items()):
            if len(types) == 1:
                imports.append(f"from {module} import {types[0]}")
            else:
                imports.append(f"from {module} import {', '.join(sorted(set(types)))}")

        return imports


class AsyncToSyncConverter:
    """
    Provides utilities to convert async classes to sync classes with proper type hints.
    """

    _thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _thread_pool_lock = threading.Lock()
    _thread_pool_initialized = False

    @classmethod
    def get_thread_pool(cls, max_workers=None) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create the shared thread pool with proper thread-safe initialization."""
        # Fast path - check if already initialized without acquiring lock
        if cls._thread_pool_initialized:
            assert cls._thread_pool is not None, "Thread pool should be initialized"
            return cls._thread_pool

        # Slow path - acquire lock and create pool if needed
        with cls._thread_pool_lock:
            if not cls._thread_pool_initialized:
                cls._thread_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix="async_to_sync_"
                )
                cls._thread_pool_initialized = True

        # This should never be None at this point, but add assertion for type checker
        assert cls._thread_pool is not None
        return cls._thread_pool

    @classmethod
    def run_async_in_thread(cls, coro_func, *args, **kwargs):
        """
        Run an async function in a separate thread from the thread pool.
        Blocks until the async function completes.
        Properly propagates contextvars between threads and manages event loops.
        """
        # Capture current context - this includes all context variables
        context = contextvars.copy_context()

        # Store the result and any exception that occurs
        result_container: dict = {"result": None, "exception": None}

        # Function that runs in the thread pool
        def run_in_thread():
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Create the coroutine within the context
                async def run_with_context():
                    # The coroutine function might access context variables
                    return await coro_func(*args, **kwargs)

                # Run the coroutine with the captured context
                # This ensures all context variables are available in the async function
                result = context.run(loop.run_until_complete, run_with_context())
                result_container["result"] = result
            except Exception as e:
                # Store the exception to re-raise in the calling thread
                result_container["exception"] = e
            finally:
                # Ensure event loop is properly closed to prevent warnings
                try:
                    # Cancel any remaining tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()

                    # Run the loop briefly to handle cancellations
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass  # Ignore errors during cleanup

                # Close the event loop
                loop.close()

                # Clear the event loop from the thread
                asyncio.set_event_loop(None)

        # Submit to thread pool and wait for result
        thread_pool = cls.get_thread_pool()
        future = thread_pool.submit(run_in_thread)
        future.result()  # Wait for completion

        # Re-raise any exception that occurred in the thread
        if result_container["exception"] is not None:
            raise result_container["exception"]

        return result_container["result"]

    @classmethod
    def create_sync_class(cls, async_class: type, thread_pool_size=10) -> type:
        """
        Creates a new class with synchronous versions of all async methods.

        Args:
            async_class: The async class to convert
            thread_pool_size: Size of thread pool to use

        Returns:
            A new class with sync versions of all async methods
        """
        sync_class_name = "ComfyAPISyncStub"
        cls.get_thread_pool(thread_pool_size)

        # Create a proper class with docstrings and proper base classes
        sync_class_dict = {
            "__doc__": async_class.__doc__,
            "__module__": async_class.__module__,
            "__qualname__": sync_class_name,
            "__orig_class__": async_class,  # Store original class for typing references
        }

        # Create __init__ method
        def __init__(self, *args, **kwargs):
            self._async_instance = async_class(*args, **kwargs)

            # Handle annotated class attributes (like execution: Execution)
            # Get all annotations from the class hierarchy and resolve string annotations
            try:
                # get_type_hints resolves string annotations to actual type objects
                # This handles classes using 'from __future__ import annotations'
                all_annotations = get_type_hints(async_class)
            except Exception:
                # Fallback to raw annotations if get_type_hints fails
                # (e.g., for undefined forward references)
                all_annotations = {}
                for base_class in reversed(inspect.getmro(async_class)):
                    if hasattr(base_class, "__annotations__"):
                        all_annotations.update(base_class.__annotations__)

            # For each annotated attribute, check if it needs to be created or wrapped
            for attr_name, attr_type in all_annotations.items():
                if hasattr(self._async_instance, attr_name):
                    # Attribute exists on the instance
                    attr = getattr(self._async_instance, attr_name)
                    # Check if this attribute needs a sync wrapper
                    if hasattr(attr, "__class__"):
                        from comfy_api._impl.internal.singleton import ProxiedSingleton

                        if isinstance(attr, ProxiedSingleton):
                            # Create a sync version of this attribute
                            try:
                                sync_attr_class = cls.create_sync_class(attr.__class__)
                                # Create instance of the sync wrapper with the async instance
                                sync_attr = object.__new__(sync_attr_class)  # type: ignore
                                sync_attr._async_instance = attr
                                setattr(self, attr_name, sync_attr)
                            except Exception:
                                # If we can't create a sync version, keep the original
                                setattr(self, attr_name, attr)
                        else:
                            # Not async, just copy the reference
                            setattr(self, attr_name, attr)
                else:
                    # Attribute doesn't exist, but is annotated - create it
                    # This handles cases like execution: Execution
                    if isinstance(attr_type, type):
                        # Check if the type is defined as an inner class
                        if hasattr(async_class, attr_type.__name__):
                            inner_class = getattr(async_class, attr_type.__name__)
                            from comfy_api._impl.internal.singleton import ProxiedSingleton

                            # Create an instance of the inner class
                            try:
                                # For ProxiedSingleton classes, get or create the singleton instance
                                if issubclass(inner_class, ProxiedSingleton):
                                    async_instance = inner_class.get_instance()
                                else:
                                    async_instance = inner_class()

                                # Create sync wrapper
                                sync_attr_class = cls.create_sync_class(inner_class)
                                sync_attr = object.__new__(sync_attr_class)  # type: ignore
                                sync_attr._async_instance = async_instance
                                setattr(self, attr_name, sync_attr)
                                # Also set on the async instance for consistency
                                setattr(self._async_instance, attr_name, async_instance)
                            except Exception as e:
                                logging.warning(
                                    f"Failed to create instance for {attr_name}: {e}"
                                )

            # Handle other instance attributes that might not be annotated
            for name, attr in inspect.getmembers(self._async_instance):
                if name.startswith("_") or hasattr(self, name):
                    continue

                # If attribute is an instance of a class, and that class is defined in the original class
                # we need to check if it needs a sync wrapper
                if isinstance(attr, object) and not isinstance(
                    attr, (str, int, float, bool, list, dict, tuple)
                ):
                    from comfy_api._impl.internal.singleton import ProxiedSingleton

                    if isinstance(attr, ProxiedSingleton):
                        # Create a sync version of this nested class
                        try:
                            sync_attr_class = cls.create_sync_class(attr.__class__)
                            # Create instance of the sync wrapper with the async instance
                            sync_attr = object.__new__(sync_attr_class)  # type: ignore
                            sync_attr._async_instance = attr
                            setattr(self, name, sync_attr)
                        except Exception:
                            # If we can't create a sync version, keep the original
                            setattr(self, name, attr)

        sync_class_dict["__init__"] = __init__

        # Process methods from the async class
        for name, method in inspect.getmembers(
            async_class, predicate=inspect.isfunction
        ):
            if name.startswith("_"):
                continue

            # Extract the actual return type from a coroutine
            if inspect.iscoroutinefunction(method):
                # Create sync version of async method with proper signature
                @functools.wraps(method)
                def sync_method(self, *args, _method_name=name, **kwargs):
                    async_method = getattr(self._async_instance, _method_name)
                    return AsyncToSyncConverter.run_async_in_thread(
                        async_method, *args, **kwargs
                    )

                # Add to the class dict
                sync_class_dict[name] = sync_method
            else:
                # For regular methods, create a proxy method
                @functools.wraps(method)
                def proxy_method(self, *args, _method_name=name, **kwargs):
                    method = getattr(self._async_instance, _method_name)
                    return method(*args, **kwargs)

                # Add to the class dict
                sync_class_dict[name] = proxy_method

        # Handle property access
        for name, prop in inspect.getmembers(
            async_class, lambda x: isinstance(x, property)
        ):

            def make_property(name, prop_obj):
                def getter(self):
                    value = getattr(self._async_instance, name)
                    if inspect.iscoroutinefunction(value):

                        def sync_fn(*args, **kwargs):
                            return AsyncToSyncConverter.run_async_in_thread(
                                value, *args, **kwargs
                            )

                        return sync_fn
                    return value

                def setter(self, value):
                    setattr(self._async_instance, name, value)

                return property(getter, setter if prop_obj.fset else None)

            sync_class_dict[name] = make_property(name, prop)

        # Create the class
        sync_class = type(sync_class_name, (object,), sync_class_dict)

        return sync_class

    @classmethod
    def _format_type_annotation(
        cls, annotation, type_tracker: Optional[TypeTracker] = None
    ) -> str:
        """Convert a type annotation to its string representation for stub files."""
        if (
            annotation is inspect.Parameter.empty
            or annotation is inspect.Signature.empty
        ):
            return "Any"

        # Handle None type
        if annotation is type(None):
            return "None"

        # Track the type if we have a tracker
        if type_tracker:
            type_tracker.track_type(annotation)

        # Try using typing.get_origin/get_args for Python 3.8+
        try:
            origin = get_origin(annotation)
            args = get_args(annotation)

            if origin is not None:
                # Track the origin type
                if type_tracker:
                    type_tracker.track_type(origin)

                # Get the origin name
                origin_name = getattr(origin, "__name__", str(origin))
                if "." in origin_name:
                    origin_name = origin_name.split(".")[-1]

                # Special handling for types.UnionType (Python 3.10+ pipe operator)
                # Convert to old-style Union for compatibility
                if str(origin) == "<class 'types.UnionType'>" or origin_name == "UnionType":
                    origin_name = "Union"

                # Format arguments recursively
                if args:
                    formatted_args = []
                    for arg in args:
                        # Track each type in the union
                        if type_tracker:
                            type_tracker.track_type(arg)
                        formatted_args.append(cls._format_type_annotation(arg, type_tracker))
                    return f"{origin_name}[{', '.join(formatted_args)}]"
                else:
                    return origin_name
        except (AttributeError, TypeError):
            # Fallback for older Python versions or non-generic types
            pass

        # Handle generic types the old way for compatibility
        if hasattr(annotation, "__origin__") and hasattr(annotation, "__args__"):
            origin = annotation.__origin__
            origin_name = (
                origin.__name__
                if hasattr(origin, "__name__")
                else str(origin).split("'")[1]
            )

            # Format each type argument
            args = []
            for arg in annotation.__args__:
                args.append(cls._format_type_annotation(arg, type_tracker))

            return f"{origin_name}[{', '.join(args)}]"

        # Handle regular types with __name__
        if hasattr(annotation, "__name__"):
            return annotation.__name__

        # Handle special module types (like types from typing module)
        if hasattr(annotation, "__module__") and hasattr(annotation, "__qualname__"):
            # For types like typing.Literal, typing.TypedDict, etc.
            return annotation.__qualname__

        # Last resort: string conversion with cleanup
        type_str = str(annotation)

        # Clean up common patterns more robustly
        if type_str.startswith("<class '") and type_str.endswith("'>"):
            type_str = type_str[8:-2]  # Remove "<class '" and "'>"

        # Remove module prefixes for common modules
        for prefix in ["typing.", "builtins.", "types."]:
            if type_str.startswith(prefix):
                type_str = type_str[len(prefix) :]

        # Handle special cases
        if type_str in ("_empty", "inspect._empty"):
            return "None"

        # Fix NoneType (this should rarely be needed now)
        if type_str == "NoneType":
            return "None"

        return type_str

    @classmethod
    def _extract_coroutine_return_type(cls, annotation):
        """Extract the actual return type from a Coroutine annotation."""
        if hasattr(annotation, "__args__") and len(annotation.__args__) > 2:
            # Coroutine[Any, Any, ReturnType] -> extract ReturnType
            return annotation.__args__[2]
        return annotation

    @classmethod
    def _format_parameter_default(cls, default_value) -> str:
        """Format a parameter's default value for stub files."""
        if default_value is inspect.Parameter.empty:
            return ""
        elif default_value is None:
            return " = None"
        elif isinstance(default_value, bool):
            return f" = {default_value}"
        elif default_value == {}:
            return " = {}"
        elif default_value == []:
            return " = []"
        else:
            return f" = {default_value}"

    @classmethod
    def _format_method_parameters(
        cls,
        sig: inspect.Signature,
        skip_self: bool = True,
        type_hints: Optional[dict] = None,
        type_tracker: Optional[TypeTracker] = None,
    ) -> str:
        """Format method parameters for stub files."""
        params = []
        if type_hints is None:
            type_hints = {}

        for i, (param_name, param) in enumerate(sig.parameters.items()):
            if i == 0 and param_name == "self" and skip_self:
                params.append("self")
            else:
                # Get type annotation from type hints if available, otherwise from signature
                annotation = type_hints.get(param_name, param.annotation)
                type_str = cls._format_type_annotation(annotation, type_tracker)

                # Get default value
                default_str = cls._format_parameter_default(param.default)

                # Combine parameter parts
                if annotation is inspect.Parameter.empty:
                    params.append(f"{param_name}: Any{default_str}")
                else:
                    params.append(f"{param_name}: {type_str}{default_str}")

        return ", ".join(params)

    @classmethod
    def _generate_method_signature(
        cls,
        method_name: str,
        method,
        is_async: bool = False,
        type_tracker: Optional[TypeTracker] = None,
    ) -> str:
        """Generate a complete method signature for stub files."""
        sig = inspect.signature(method)

        # Try to get evaluated type hints to resolve string annotations
        try:
            from typing import get_type_hints
            type_hints = get_type_hints(method)
        except Exception:
            # Fallback to empty dict if we can't get type hints
            type_hints = {}

        # For async methods, extract the actual return type
        return_annotation = type_hints.get('return', sig.return_annotation)
        if is_async and inspect.iscoroutinefunction(method):
            return_annotation = cls._extract_coroutine_return_type(return_annotation)

        # Format parameters with type hints
        params_str = cls._format_method_parameters(sig, type_hints=type_hints, type_tracker=type_tracker)

        # Format return type
        return_type = cls._format_type_annotation(return_annotation, type_tracker)
        if return_annotation is inspect.Signature.empty:
            return_type = "None"

        return f"def {method_name}({params_str}) -> {return_type}: ..."

    @classmethod
    def _generate_imports(
        cls, async_class: type, type_tracker: TypeTracker
    ) -> list[str]:
        """Generate import statements for the stub file."""
        imports = []

        # Add standard typing imports
        imports.append(
            "from typing import Any, Dict, List, Optional, Tuple, Union, Set, Sequence, cast, NamedTuple"
        )

        # Add imports from the original module
        if async_class.__module__ != "builtins":
            module = inspect.getmodule(async_class)
            additional_types = []

            if module:
                # Check if module has __all__ defined
                module_all = getattr(module, "__all__", None)

                for name, obj in sorted(inspect.getmembers(module)):
                    if isinstance(obj, type):
                        # Skip if __all__ is defined and this name isn't in it
                        # unless it's already been tracked as used in type annotations
                        if module_all is not None and name not in module_all:
                            # Check if this type was actually used in annotations
                            if name not in type_tracker.discovered_types:
                                continue

                        # Check for NamedTuple
                        if issubclass(obj, tuple) and hasattr(obj, "_fields"):
                            additional_types.append(name)
                            # Mark as already imported
                            type_tracker.already_imported.add(name)
                        # Check for Enum
                        elif issubclass(obj, Enum) and name != "Enum":
                            additional_types.append(name)
                            # Mark as already imported
                            type_tracker.already_imported.add(name)

            if additional_types:
                type_imports = ", ".join([async_class.__name__] + additional_types)
                imports.append(f"from {async_class.__module__} import {type_imports}")
            else:
                imports.append(
                    f"from {async_class.__module__} import {async_class.__name__}"
                )

        # Add imports for all discovered types
        # Pass the main module name to avoid duplicate imports
        imports.extend(
            type_tracker.get_imports(main_module_name=async_class.__module__)
        )

        # Add base module import if needed
        if hasattr(inspect.getmodule(async_class), "__name__"):
            module_name = inspect.getmodule(async_class).__name__
            if "." in module_name:
                base_module = module_name.split(".")[0]
                # Only add if not already importing from it
                if not any(imp.startswith(f"from {base_module}") for imp in imports):
                    imports.append(f"import {base_module}")

        return imports

    @classmethod
    def _get_class_attributes(cls, async_class: type) -> list[tuple[str, type]]:
        """Extract class attributes that are classes themselves."""
        class_attributes = []

        # Get resolved type hints to handle string annotations
        try:
            type_hints = get_type_hints(async_class)
        except Exception:
            type_hints = {}

        # Look for class attributes that are classes
        for name, attr in sorted(inspect.getmembers(async_class)):
            if isinstance(attr, type) and not name.startswith("_"):
                class_attributes.append((name, attr))
            elif name in type_hints:
                # Use resolved type hint instead of raw annotation
                annotation = type_hints[name]
                if isinstance(annotation, type):
                    class_attributes.append((name, annotation))

        return class_attributes

    @classmethod
    def _generate_inner_class_stub(
        cls,
        name: str,
        attr: type,
        indent: str = "    ",
        type_tracker: Optional[TypeTracker] = None,
    ) -> list[str]:
        """Generate stub for an inner class."""
        stub_lines = []
        stub_lines.append(f"{indent}class {name}Sync:")

        # Add docstring if available
        if hasattr(attr, "__doc__") and attr.__doc__:
            stub_lines.extend(
                cls._format_docstring_for_stub(attr.__doc__, f"{indent}    ")
            )

        # Add __init__ if it exists
        if hasattr(attr, "__init__"):
            try:
                init_method = getattr(attr, "__init__")
                init_sig = inspect.signature(init_method)

                # Try to get type hints
                try:
                    from typing import get_type_hints
                    init_hints = get_type_hints(init_method)
                except Exception:
                    init_hints = {}

                # Format parameters
                params_str = cls._format_method_parameters(
                    init_sig, type_hints=init_hints, type_tracker=type_tracker
                )
                # Add __init__ docstring if available (before the method)
                if hasattr(init_method, "__doc__") and init_method.__doc__:
                    stub_lines.extend(
                        cls._format_docstring_for_stub(
                            init_method.__doc__, f"{indent}    "
                        )
                    )
                stub_lines.append(
                    f"{indent}    def __init__({params_str}) -> None: ..."
                )
            except (ValueError, TypeError):
                stub_lines.append(
                    f"{indent}    def __init__(self, *args, **kwargs) -> None: ..."
                )

        # Add methods to the inner class
        has_methods = False
        for method_name, method in sorted(
            inspect.getmembers(attr, predicate=inspect.isfunction)
        ):
            if method_name.startswith("_"):
                continue

            has_methods = True
            try:
                # Add method docstring if available (before the method signature)
                if method.__doc__:
                    stub_lines.extend(
                        cls._format_docstring_for_stub(method.__doc__, f"{indent}    ")
                    )

                method_sig = cls._generate_method_signature(
                    method_name, method, is_async=True, type_tracker=type_tracker
                )
                stub_lines.append(f"{indent}    {method_sig}")
            except (ValueError, TypeError):
                stub_lines.append(
                    f"{indent}    def {method_name}(self, *args, **kwargs): ..."
                )

        if not has_methods:
            stub_lines.append(f"{indent}    pass")

        return stub_lines

    @classmethod
    def _format_docstring_for_stub(
        cls, docstring: str, indent: str = "    "
    ) -> list[str]:
        """Format a docstring for inclusion in a stub file with proper indentation."""
        if not docstring:
            return []

        # First, dedent the docstring to remove any existing indentation
        dedented = textwrap.dedent(docstring).strip()

        # Split into lines
        lines = dedented.split("\n")

        # Build the properly indented docstring
        result = []
        result.append(f'{indent}"""')

        for line in lines:
            if line.strip():  # Non-empty line
                result.append(f"{indent}{line}")
            else:  # Empty line
                result.append("")

        result.append(f'{indent}"""')
        return result

    @classmethod
    def _post_process_stub_content(cls, stub_content: list[str]) -> list[str]:
        """Post-process stub content to fix any remaining issues."""
        processed = []

        for line in stub_content:
            # Skip processing imports
            if line.startswith(("from ", "import ")):
                processed.append(line)
                continue

            # Fix method signatures missing return types
            if (
                line.strip().startswith("def ")
                and line.strip().endswith(": ...")
                and ") -> " not in line
            ):
                # Add -> None for methods without return annotation
                line = line.replace(": ...", " -> None: ...")

            processed.append(line)

        return processed

    @classmethod
    def generate_stub_file(cls, async_class: type, sync_class: type) -> None:
        """
        Generate a .pyi stub file for the sync class to help IDEs with type checking.
        """
        try:
            # Only generate stub if we can determine module path
            if async_class.__module__ == "__main__":
                return

            module = inspect.getmodule(async_class)
            if not module:
                return

            module_path = module.__file__
            if not module_path:
                return

            # Create stub file path in a 'generated' subdirectory
            module_dir = os.path.dirname(module_path)
            stub_dir = os.path.join(module_dir, "generated")

            # Ensure the generated directory exists
            os.makedirs(stub_dir, exist_ok=True)

            module_name = os.path.basename(module_path)
            if module_name.endswith(".py"):
                module_name = module_name[:-3]

            sync_stub_path = os.path.join(stub_dir, f"{sync_class.__name__}.pyi")

            # Create a type tracker for this stub generation
            type_tracker = TypeTracker()

            stub_content = []

            # We'll generate imports after processing all methods to capture all types
            # Leave a placeholder for imports
            imports_placeholder_index = len(stub_content)
            stub_content.append("")  # Will be replaced with imports later

            # Class definition
            stub_content.append(f"class {sync_class.__name__}:")

            # Docstring
            if async_class.__doc__:
                stub_content.extend(
                    cls._format_docstring_for_stub(async_class.__doc__, "    ")
                )

            # Generate __init__
            try:
                init_method = async_class.__init__
                init_signature = inspect.signature(init_method)

                # Try to get type hints for __init__
                try:
                    from typing import get_type_hints
                    init_hints = get_type_hints(init_method)
                except Exception:
                    init_hints = {}

                # Format parameters
                params_str = cls._format_method_parameters(
                    init_signature, type_hints=init_hints, type_tracker=type_tracker
                )
                # Add __init__ docstring if available (before the method)
                if hasattr(init_method, "__doc__") and init_method.__doc__:
                    stub_content.extend(
                        cls._format_docstring_for_stub(init_method.__doc__, "    ")
                    )
                stub_content.append(f"    def __init__({params_str}) -> None: ...")
            except (ValueError, TypeError):
                stub_content.append(
                    "    def __init__(self, *args, **kwargs) -> None: ..."
                )

            stub_content.append("")  # Add newline after __init__

            # Get class attributes
            class_attributes = cls._get_class_attributes(async_class)

            # Generate inner classes
            for name, attr in class_attributes:
                inner_class_stub = cls._generate_inner_class_stub(
                    name, attr, type_tracker=type_tracker
                )
                stub_content.extend(inner_class_stub)
                stub_content.append("")  # Add newline after the inner class

            # Add methods to the main class
            processed_methods = set()  # Keep track of methods we've processed
            for name, method in sorted(
                inspect.getmembers(async_class, predicate=inspect.isfunction)
            ):
                if name.startswith("_") or name in processed_methods:
                    continue

                processed_methods.add(name)

                try:
                    method_sig = cls._generate_method_signature(
                        name, method, is_async=True, type_tracker=type_tracker
                    )

                    # Add docstring if available (before the method signature for proper formatting)
                    if method.__doc__:
                        stub_content.extend(
                            cls._format_docstring_for_stub(method.__doc__, "    ")
                        )

                    stub_content.append(f"    {method_sig}")

                    stub_content.append("")  # Add newline after each method

                except (ValueError, TypeError):
                    # If we can't get the signature, just add a simple stub
                    stub_content.append(f"    def {name}(self, *args, **kwargs): ...")
                    stub_content.append("")  # Add newline

            # Add properties
            for name, prop in sorted(
                inspect.getmembers(async_class, lambda x: isinstance(x, property))
            ):
                stub_content.append("    @property")
                stub_content.append(f"    def {name}(self) -> Any: ...")
                if prop.fset:
                    stub_content.append(f"    @{name}.setter")
                    stub_content.append(
                        f"    def {name}(self, value: Any) -> None: ..."
                    )
                stub_content.append("")  # Add newline after each property

            # Add placeholders for the nested class instances
            # Check the actual attribute names from class annotations and attributes
            attribute_mappings = {}

            # First check annotations for typed attributes (including from parent classes)
            # Resolve string annotations to actual types
            try:
                all_annotations = get_type_hints(async_class)
            except Exception:
                # Fallback to raw annotations
                all_annotations = {}
                for base_class in reversed(inspect.getmro(async_class)):
                    if hasattr(base_class, "__annotations__"):
                        all_annotations.update(base_class.__annotations__)

            for attr_name, attr_type in sorted(all_annotations.items()):
                for class_name, class_type in class_attributes:
                    # If the class type matches the annotated type
                    if (
                        attr_type == class_type
                        or (hasattr(attr_type, "__name__") and attr_type.__name__ == class_name)
                        or (isinstance(attr_type, str) and attr_type == class_name)
                    ):
                        attribute_mappings[class_name] = attr_name

            # Remove the extra checking - annotations should be sufficient

            # Add the attribute declarations with proper names
            for class_name, class_type in class_attributes:
                # Check if there's a mapping from annotation
                attr_name = attribute_mappings.get(class_name, class_name)
                # Use the annotation name if it exists, even if the attribute doesn't exist yet
                # This is because the attribute might be created at runtime
                stub_content.append(f"    {attr_name}: {class_name}Sync")

            stub_content.append("")  # Add a final newline

            # Now generate imports with all discovered types
            imports = cls._generate_imports(async_class, type_tracker)

            # Deduplicate imports while preserving order
            seen = set()
            unique_imports = []
            for imp in imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_imports.append(imp)
                else:
                    logging.warning(f"Duplicate import detected: {imp}")

            # Replace the placeholder with actual imports
            stub_content[imports_placeholder_index : imports_placeholder_index + 1] = (
                unique_imports
            )

            # Post-process stub content
            stub_content = cls._post_process_stub_content(stub_content)

            # Write stub file
            with open(sync_stub_path, "w") as f:
                f.write("\n".join(stub_content))

            logging.info(f"Generated stub file: {sync_stub_path}")

        except Exception as e:
            # If stub generation fails, log the error but don't break the main functionality
            logging.error(
                f"Error generating stub file for {sync_class.__name__}: {str(e)}"
            )
            import traceback

            logging.error(traceback.format_exc())


def create_sync_class(async_class: type, thread_pool_size=10) -> type:
    """
    Creates a sync version of an async class

    Args:
        async_class: The async class to convert
        thread_pool_size: Size of thread pool to use

    Returns:
        A new class with sync versions of all async methods
    """
    return AsyncToSyncConverter.create_sync_class(async_class, thread_pool_size)
