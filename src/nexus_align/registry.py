"""Component registry: name-to-class lookup for datasets, models, algorithms, trainers, evaluators, and rewards."""

from typing import TypeVar

# Component types that can be registered.
COMPONENT_TYPES = (
    "dataset",
    "model",
    "algorithm",
    "trainer",
    "evaluator",
    "reward",
)

T = TypeVar("T")


class Registry:
    """
    Global registry for project components.

    - Register entries with register(component_type, name, cls).
    - Resolve entries with get(component_type, name).
    """

    _stores: dict[str, dict[str, type]]

    def __init__(self) -> None:
        self._stores = {ct: {} for ct in COMPONENT_TYPES}
    
    def register(self, component_type: str, name: str, cls: type[T]) -> type[T]:
        """Register class under (component_type, name)."""
        if component_type not in COMPONENT_TYPES:
            raise KeyError(
                f"❌ Unknown component_type: {component_type}"
                f"Available {','.join(COMPONENT_TYPES)}"
            )
        self._stores[component_type][name] = cls

    def get(self, component_type: str, name: str) -> type:
        """Return the registered class for (component_type, name)."""
        return self._stores[component_type][name]


registry = Registry()
