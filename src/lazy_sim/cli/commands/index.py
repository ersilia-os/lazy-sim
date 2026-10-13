import click

from ...utils.parallel import set_threads
from . import (
    embeddings_option,
    featurizers_option,
    library_option,
    lazysim_cli,
    run_or_fail,
    threads_option,
)


def index_cmd():
    @lazysim_cli.command(short_help="Embed and index a chemical library.")
    @click.option(
        "-i",
        "--input",
        "input_csv",
        type=click.Path(exists=True, dir_okay=False),
        help="Library CSV with 'smiles' and 'id' columns. Leave out to add "
        "featurizers to an existing library.",
    )
    @library_option(
        "Name for the library (kept in ~/lazysim/libraries), or a folder path."
    )
    @featurizers_option
    @embeddings_option
    @threads_option
    def index(input_csv, artifacts_dir, featurizers, embeddings, threads):
        """Embed every molecule of a library and build a searchable index."""
        set_threads(threads)
        from ...api import index as build_index

        run_or_fail(
            build_index,
            input_csv,
            artifacts_dir,
            featurizers=featurizers,
            embeddings=embeddings,
            threads=threads,
        )
