import click

from ... import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="lazysim")
def lazysim_cli():
    """Ultra-fast similarity search in large chemical libraries.

    lazy-sim is developed by the Ersilia Open Source Initiative (https://ersilia.io).
    """


def run_or_fail(func, *args, **kwargs):
    """Call `func`, turning expected errors into clean CLI messages."""
    try:
        return func(*args, **kwargs)
    except (ValueError, OSError) as e:
        raise click.ClickException(str(e)) from None


threads_option = click.option(
    "-t",
    "--threads",
    type=click.IntRange(min=1),
    help="Processes and threads to use.  [default: all CPUs]",
)


def library_option(help):
    """The ``-a`` option: a library name (in ~/lazysim/libraries) or a path."""
    return click.option(
        "-a",
        "--artifacts",
        "artifacts_dir",
        required=True,
        metavar="NAME|PATH",
        help=help,
    )
