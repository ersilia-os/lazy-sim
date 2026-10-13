"""Ultra-fast similarity search in large chemical libraries, by the Ersilia Open Source Initiative."""

from importlib.metadata import version

__version__ = version("lazy-sim")

__all__ = ["index", "fetch", "search", "__version__"]


def __getattr__(name):
    # Imported lazily so `lazysim --help` stays fast and the CLI can set
    # thread variables before numpy loads.
    if name in ("index", "fetch", "search"):
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
