import click

from ...featurizers import AVAILABLE
from ...utils.parallel import set_threads
from . import library_option, lazysim_cli, run_or_fail, threads_option


def index_cmd():
    @lazysim_cli.command(short_help="Embed and index a chemical library.")
    @click.option(
        "-i",
        "--input",
        "input_csv",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Library CSV with 'smiles' and 'id' columns.",
    )
    @library_option(
        "Name for the library (kept in ~/lazysim/libraries), or a new folder path."
    )
    @click.option(
        "-f",
        "--featurizer",
        default="clamp",
        show_default=True,
        type=click.Choice(AVAILABLE),
        help="Molecular embedding.",
    )
    @click.option(
        "--embeddings",
        is_flag=True,
        help="Also keep the full float32 embeddings (3 KB per molecule).",
    )
    @threads_option
    def index(input_csv, artifacts_dir, featurizer, embeddings, threads):
        """Embed every molecule of a library and build a searchable index."""
        set_threads(threads)
        from ...api import index as build_index

        run_or_fail(
            build_index,
            input_csv,
            artifacts_dir,
            featurizer=featurizer,
            embeddings=embeddings,
            threads=threads,
        )
