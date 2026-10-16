from .commands import lazysim_cli
from .commands.catalog import catalog_cmd
from .commands.compare import compare_cmd
from .commands.fetch import fetch_cmd
from .commands.index import index_cmd
from .commands.info import info_cmd
from .commands.search import search_cmd


def create_lazysim_cli():
    """Register every command on the ``lazysim`` click group and return it."""
    fetch_cmd()
    index_cmd()
    search_cmd()
    catalog_cmd()
    info_cmd()
    compare_cmd()
    return lazysim_cli
