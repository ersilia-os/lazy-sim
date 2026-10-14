import click
from rich.console import Console
from rich.table import Table

from ...io import artifacts, home, remote
from . import lazysim_cli, run_or_fail
from .info import sizes, source


def _gb(files):
    return f"{sum(f['bytes'] for f in files) / 1e9:.2f} GB" if files else "—"


def _local_table():
    libraries = home.list_local()
    if not libraries:
        return None
    table = Table(title=f"Libraries in {home.libraries_dir()}")
    for column in ("Name", "Featurizer", "Similarity", "Molecules", "Index"):
        table.add_column(column, no_wrap=True)
    table.add_column("Embeddings", no_wrap=True)
    table.add_column("Source")
    table.add_column("Created", no_wrap=True)
    for name, path, m in libraries:
        if not artifacts.featurizers_in(path):  # indexing never finished
            unfinished = ("—", "—", f"{m['library']['indexed']:,}", "—", "—")
            table.add_row(name, *unfinished, source(m), m["created"][:10])
        for featurizer in artifacts.featurizers_in(path):
            fm = artifacts.read_featurizer(path, featurizer)
            table.add_row(
                name,
                featurizer,
                fm["similarity"],
                f"{m['library']['indexed']:,}",
                *sizes(path, featurizer, fm),
                source(m),
                fm["created"][:10],
            )
    return table


def _remote_table():
    catalog = run_or_fail(remote.read_catalog)
    local = {
        (name, f)
        for name, path, _ in home.list_local()
        for f in artifacts.featurizers_in(path)
    }
    shared = ", ".join(f"{n} {_gb(e['files'])}" for n, e in sorted(catalog.items()))
    table = Table(
        title="Libraries to fetch with 'lazysim fetch NAME [-f FEATURIZER]'",
        caption=f"Each library also downloads its ids and SMILES once ({shared}).",
    )
    table.add_column("Name", no_wrap=True)
    table.add_column("Title")
    for column in ("Featurizer", "Similarity", "Molecules", "Index", "Embeddings"):
        table.add_column(column, no_wrap=True)
    table.add_column("Fetched", no_wrap=True)
    for name, entry in sorted(catalog.items()):
        for featurizer, published in sorted(entry["featurizers"].items()):
            table.add_row(
                name,
                entry["title"],
                featurizer,
                published["similarity"],
                f"{entry['molecules']:,}",
                _gb(published["files"]),
                _gb(published.get("embeddings", [])),
                "Yes" if (name, featurizer) in local else "No",
            )
    return table


def catalog_cmd():
    @lazysim_cli.command(short_help="List local libraries, or ones to fetch.")
    @click.option(
        "--remote",
        "show_remote",
        is_flag=True,
        help="List the libraries Ersilia publishes, instead of local ones.",
    )
    def catalog(show_remote):
        """List libraries in ~/lazysim/libraries, or with --remote, ones to fetch.

        There is one row per library and featurizer.
        """
        console = Console()
        table = _remote_table() if show_remote else _local_table()
        if table is None:
            console.print(
                f"No libraries in {home.libraries_dir()} yet. "
                "See 'lazysim catalog --remote'."
            )
        else:
            console.print(table)
