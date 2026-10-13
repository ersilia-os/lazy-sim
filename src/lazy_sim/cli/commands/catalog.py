import click
from rich.console import Console
from rich.table import Table

from ...io import artifacts, home, remote
from . import lazysim_cli, run_or_fail
from .info import source


def _gb(files):
    return f"{sum(f['bytes'] for f in files) / 1e9:.2f} GB" if files else "—"


def _local_table():
    libraries = home.list_local()
    if not libraries:
        return None
    table = Table(title=f"Libraries in {home.libraries_dir()}")
    for column in ("Name", "Molecules", "Embeddings", "Size"):
        table.add_column(column, no_wrap=True)
    table.add_column("Source")
    table.add_column("Created", no_wrap=True)
    for name, path, m in libraries:
        feat = m["featurizer"]["name"]
        has_embeddings = (path / artifacts.embeddings_file(feat)).exists()
        table.add_row(
            name,
            f"{m['library']['indexed']:,}",
            "yes" if has_embeddings else "no",
            artifacts.disk_size(path),
            source(m),
            m["created"][:10],
        )
    return table


def _remote_table():
    catalog = run_or_fail(remote.read_catalog)
    local = {name for name, _, _ in home.list_local()}
    table = Table(title="Libraries to fetch with 'lazysim fetch NAME'")
    table.add_column("Name", no_wrap=True)
    table.add_column("Title")
    for column in ("Version", "Molecules", "Download", "Embeddings", "Fetched"):
        table.add_column(column, no_wrap=True)
    for name, entry in sorted(catalog.items()):
        files = entry["files"]
        table.add_row(
            name,
            entry["title"],
            entry["version"],
            f"{entry['molecules']:,}",
            _gb(files["library"]),
            _gb(files.get("embeddings", [])),
            "yes" if name in local else "no",
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
        """List libraries in ~/lazysim/libraries, or with --remote, ones to fetch."""
        console = Console()
        table = _remote_table() if show_remote else _local_table()
        if table is None:
            console.print(
                f"No libraries in {home.libraries_dir()} yet. "
                "See 'lazysim catalog --remote'."
            )
        else:
            console.print(table)
