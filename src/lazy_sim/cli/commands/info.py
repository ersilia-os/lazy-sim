from rich.console import Console
from rich.table import Table

from ...io import artifacts, home
from . import library_option, lazysim_cli, run_or_fail


def source(manifest):
    """Describe where a library comes from: an Ersilia release or a local index."""
    src = manifest.get("source")
    return f"{src['name']} {src['version']}" if src else "indexed locally"


def _engine(fm):
    if fm["engine"] == "turbovec":
        return f"turbovec {fm['engine_version']}, {fm['bits']}-bit"
    return f"FPSim2 {fm['engine_version']}, {fm['fp_type']} {fm['fp_params']}"


def info_cmd():
    @lazysim_cli.command(short_help="Describe a library.")
    @library_option("Library name or folder.")
    def info(artifacts_dir):
        """Show what a library contains and how each index was built."""
        folder = run_or_fail(home.resolve, artifacts_dir, must_exist=True)
        m = artifacts.read_manifest(folder)
        lib = m["library"]
        console = Console()
        table = Table(show_header=False, title=str(folder))
        for key, value in (
            ("Molecules", f"{lib['indexed']:,}"),
            ("Rows skipped", f"{lib['skipped']:,}"),
            ("Source", source(m)),
            ("Library file", lib["file"]),
            ("Library sha256", lib["sha256"]),
            ("Built with", f"lazy-sim {m['lazysim_version']}"),
            ("Created", m["created"]),
        ):
            table.add_row(key, value)
        console.print(table)
        indexes = Table(title="Indexes")
        for column in ("Featurizer", "Similarity", "Engine", "Embeddings", "Size"):
            indexes.add_column(column)
        for name in artifacts.featurizers_in(folder):
            fm = artifacts.read_featurizer(folder, name)
            emb = (folder / artifacts.embeddings_file(name)).exists()
            indexes.add_row(
                name,
                fm["similarity"],
                _engine(fm),
                "Yes" if emb else "No",
                artifacts.files_size(artifacts.featurizer_files(folder, name)),
            )
        console.print(indexes)
