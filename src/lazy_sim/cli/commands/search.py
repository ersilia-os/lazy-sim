import click

from ...featurizers import AVAILABLE
from ...utils.parallel import set_threads
from . import library_option, lazysim_cli, run_or_fail, threads_option


def search_cmd():
    @lazysim_cli.command(short_help="Find the nearest library molecules of queries.")
    @click.option(
        "-i",
        "--input",
        "input_csv",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Query CSV with 'smiles' and 'id' columns.",
    )
    @library_option("Library name or folder, from 'lazysim index' or 'fetch'.")
    @click.option(
        "-o",
        "--output",
        "output_csv",
        required=True,
        type=click.Path(dir_okay=False),
        help="Output CSV, one row per hit.",
    )
    @click.option(
        "-k",
        "--top-k",
        "k",
        default=100,
        show_default=True,
        type=click.IntRange(min=1),
        help="Hits per query.",
    )
    @click.option(
        "-f",
        "--featurizer",
        type=click.Choice(AVAILABLE),
        help="Index to search: clamp (cosine) or ecfp (Tanimoto).  "
        "[default: clamp if the library has it]",
    )
    @threads_option
    def search(input_csv, artifacts_dir, output_csv, k, featurizer, threads):
        """Find the k most similar library molecules for each query."""
        set_threads(threads)
        from ...api import search as run_search

        run_or_fail(
            run_search,
            input_csv,
            artifacts_dir,
            output_csv,
            k=k,
            featurizer=featurizer,
            threads=threads,
        )
