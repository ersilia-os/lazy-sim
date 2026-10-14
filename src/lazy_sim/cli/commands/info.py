from rich.console import Console
from rich.table import Table

from ...io import artifacts, home
from . import library_option, lazysim_cli, run_or_fail


def source(manifest):
    """Describe where a library comes from: an Ersilia release or a local index."""
    src = manifest.get("source")
    return f"{src['name']} {src['version']}" if src else "indexed locally"


def sizes(folder, featurizer, manifest):
    """Return the index size and the embeddings size (or "No") of a featurizer."""
    emb = folder / artifacts.embeddings_file(featurizer)
    index = artifacts.files_size([folder / manifest["index_file"]])
    return index, artifacts.files_size([emb]) if emb.exists() else "No"


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
            ("Ids and SMILES", artifacts.files_size([folder / artifacts.LIBRARY_DB])),
            ("Library file", lib["file"]),
            ("Library sha256", lib["sha256"]),
            ("Built with", f"lazy-sim {m['lazysim_version']}"),
            ("Created", m["created"]),
        ):
            table.add_row(key, value)
        console.print(table)
        indexes = Table(title="Indexes")
        for column in ("Featurizer", "Similarity", "Engine", "Index", "Embeddings"):
            indexes.add_column(column)
        for name in artifacts.featurizers_in(folder):
            fm = artifacts.read_featurizer(folder, name)
            indexes.add_row(
                name, fm["similarity"], _engine(fm), *sizes(folder, name, fm)
            )
        console.print(indexes)
