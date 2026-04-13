"""Unified registry for algorithms, models, datasets, reward_models, pipelines."""

from typing import TypeVar

# Component types that can be registered
COMPONENT_TYPES = (
    "algorithm",
    "model",
    "dataset",
    "reward_model",
    "pipeline",
)

T = TypeVar("T")


class Registry:
    """
    Global registry: name -> class (or factory) per component type.
    
    Register: registry.register(component_type, name, cls).
    Get: registry.get(component_type, name).
    """

    _stores: dict[str, dict[str, type]]

    def __init__(self) -> None:
        self._stores = {ct: {} for ct in COMPONENT_TYPES}

    def register(self, component_type: str, name: str, cls: type[T]) -> type[T]:
        """Register a class for a component type and name. Returns cls for decorator use."""
        if component_type not in self._stores:
            self._stores[component_type] = {}
        self._stores[component_type][name] = cls
        return cls

    def get(self, component_type: str, name: str) -> type:
        """Return the registered class for (component_type, name). Raises if missing."""
        if component_type not in self._stores:
            raise KeyError(f"❌ Unknown component_type: {component_type}")
        store = self._stores[component_type]
        if name not in store:
            raise KeyError(
                f"❌ No {component_type} registered for name '{name}'. "
                f"Available: {list(store.keys())}"
            )
        return store[name]


registry = Registry()
