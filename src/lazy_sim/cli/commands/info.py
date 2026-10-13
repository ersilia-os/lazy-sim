from rich.console import Console
from rich.table import Table

from ...io import artifacts, home
from . import library_option, lazysim_cli, run_or_fail


def source(manifest):
    """Describe where a library comes from: an Ersilia release or a local index."""
    src = manifest.get("source")
    return f"{src['name']} {src['version']}" if src else "indexed locally"


def info_cmd():
    @lazysim_cli.command(short_help="Describe a library.")
    @library_option("Library name or folder.")
    def info(artifacts_dir):
        """Show what a library contains and how it was built."""
        folder = run_or_fail(home.resolve, artifacts_dir, must_exist=True)
        m = artifacts.read_manifest(folder)
        lib, feat, backend = m["library"], m["featurizer"], m["backend"]
        has_embeddings = (folder / artifacts.embeddings_file(feat["name"])).exists()
        table = Table(show_header=False, title=str(folder))
        for key, value in (
            ("Molecules indexed", f"{lib['indexed']:,}"),
            ("Rows skipped", f"{lib['skipped']:,}"),
            ("Source", source(m)),
            ("Featurizer", f"{feat['name']} ({feat['dim']}-d)"),
            ("Index", f"{backend['name']} {backend['version']}, {backend['bits']}-bit"),
            ("Full embeddings", "yes" if has_embeddings else "no"),
            ("Size on disk", artifacts.disk_size(folder)),
            ("Library file", lib["file"]),
            ("Library sha256", lib["sha256"]),
            ("Built with", f"lazy-sim {m['lazysim_version']}"),
            ("Created", m["created"]),
        ):
            table.add_row(key, value)
        Console().print(table)
