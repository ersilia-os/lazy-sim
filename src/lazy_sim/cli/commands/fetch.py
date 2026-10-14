import click

from . import embeddings_option, featurizers_option, lazysim_cli, run_or_fail


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
    @featurizers_option
    @embeddings_option
    def fetch(name, artifacts_dir, featurizers, embeddings):
        """Download library NAME, ready to search (see 'lazysim catalog --remote').

        Fetching a library you already have adds the featurizers it lacks.
        """
        from ...api import fetch as run_fetch

        run_or_fail(
            run_fetch,
            name,
            featurizers=featurizers,
            artifacts_dir=artifacts_dir,
            embeddings=embeddings,
        )
