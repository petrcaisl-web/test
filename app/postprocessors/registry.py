"""Auto-discovery registry for BasePostProcessor subclasses.

On startup, import all postprocessor modules so their classes are
registered as subclasses of BasePostProcessor, then call
``PostProcessorRegistry.build()`` to collect them.
"""

from __future__ import annotations

import importlib
import pkgutil
import structlog

from app.postprocessors.base import BasePostProcessor

logger = structlog.get_logger(__name__)


class PostProcessorRegistry:
    """Holds all discovered postprocessor instances.

    Usage::

        registry = PostProcessorRegistry.build()
        pp = registry.get("payment-extraction")
        result = await pp.process(ocr_result)
    """

    def __init__(self, processors: list[BasePostProcessor]) -> None:
        self._processors: dict[str, BasePostProcessor] = {
            p.name: p for p in processors
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls) -> "PostProcessorRegistry":
        """Discover and instantiate all BasePostProcessor subclasses.

        Imports every module in the ``app.postprocessors`` package so
        that subclasses register themselves via Python's subclass
        tracking mechanism.  Then collects all concrete (non-abstract)
        subclasses and instantiates them.

        Returns:
            A populated PostProcessorRegistry.
        """
        cls._import_all_submodules()

        processors: list[BasePostProcessor] = []
        for subclass in BasePostProcessor.__subclasses__():
            # Skip abstract subclasses (those that don't define ``name``)
            if not hasattr(subclass, "name") or not hasattr(subclass, "process"):
                continue
            try:
                instance = subclass()
                processors.append(instance)
                logger.debug(
                    "postprocessor_registered",
                    name=subclass.name,
                    version=getattr(subclass, "version", "?"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "postprocessor_instantiation_failed",
                    subclass=subclass.__name__,
                    error=str(exc),
                )

        logger.info("postprocessors_loaded", count=len(processors))
        return cls(processors)

    @staticmethod
    def _import_all_submodules() -> None:
        """Import every module in the app.postprocessors package."""
        import app.postprocessors as pkg

        for module_info in pkgutil.iter_modules(pkg.__path__):
            if module_info.name in ("base", "registry"):
                continue
            module_name = f"app.postprocessors.{module_info.name}"
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "postprocessor_module_import_failed",
                    module=module_name,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def list_all(self) -> list[BasePostProcessor]:
        """Return all registered postprocessor instances."""
        return list(self._processors.values())

    def get(self, name: str) -> BasePostProcessor:
        """Return the postprocessor with the given name.

        Args:
            name: The postprocessor name as defined in its ``name``
                class attribute (e.g. ``"payment-extraction"``).

        Raises:
            KeyError: If no postprocessor with that name is registered.
        """
        try:
            return self._processors[name]
        except KeyError:
            available = sorted(self._processors.keys())
            raise KeyError(
                f"Postprocessor '{name}' not found. "
                f"Available: {available}"
            ) from None
