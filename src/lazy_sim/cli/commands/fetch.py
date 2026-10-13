import click

from . import lazysim_cli, run_or_fail


def fetch_cmd():
    @lazysim_cli.command(short_help="Download a library indexed by Ersilia.")
    @click.argument("name")
    @click.option(
        "-a",
        "--artifacts",
        "artifacts_dir",
        metavar="NAME|PATH",
        help="Where to put it.  [default: ~/lazysim/libraries/NAME]",
    )
    @click.option(
        "--embeddings",
        is_flag=True,
        help="Also download the full float32 embeddings (large).",
    )
    def fetch(name, artifacts_dir, embeddings):
        """Download library NAME, ready to search (see 'lazysim catalog --remote')."""
        from ...api import fetch as run_fetch

        run_or_fail(run_fetch, name, artifacts_dir, embeddings=embeddings)
