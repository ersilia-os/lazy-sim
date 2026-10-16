import click
from rich.console import Console
from rich.table import Table

from ...featurizers import AVAILABLE
from ...io import results
from . import lazysim_cli, run_or_fail

MODES = {
    "indexed": "What 'lazysim search' returns",
    "exact": "Exact rankings, from the embeddings",
}
NOTES = (
    "Molecules in both: the share of the top k that both featurizers find.",
    "Similarity: exact, under each featurizer's own measure; random library "
    "molecules give the baseline.",
    "Position: where one featurizer's top k sit in the other's ranking of the whole "
    "library; a random molecule sits in the middle.",
)


def _cell(values, rank=False):
    q1, median, q3 = values
    fmt = "{:,.0f}" if rank else "{:.2f}"
    return f"{fmt.format(median)} [{fmt.format(q1)}–{fmt.format(q3)}]"


def render(stored, entry):
    """Lay out a stored comparison of two featurizers.

    Parameters
    ----------
    stored : dict
        The library's results, from `lazy_sim.io.results.pair`.
    entry : dict
        The comparison of one pair of featurizers.

    Returns
    -------
    tuple
        Header lines, one Rich table per mode (indexed, exact), and closing notes.
    """
    a, b = entry["featurizers"]
    ks, n = [str(k) for k in entry["k"]], stored["molecules"]
    header = (
        f"{stored['library']} ({n:,} molecules): {a} vs {b}",
        f"{entry['queries']:,} library molecules as queries. Each cell: the typical "
        "query (median) [the middle half of the queries].",
    )
    tables = []
    for mode, title in MODES.items():
        m = entry[mode]
        table = Table()
        table.add_column(title)
        for k in ks:
            table.add_column(f"k={int(k):,}", justify="right")
        table.add_row(
            "Molecules in both top-k lists", *(_cell(m["shared"][k]) for k in ks)
        )
        for f, g in ((a, b), (b, a)):
            table.add_row(f"{f} {entry['measures'][f]} to the query", style="bold")
            for h in (f, g):
                sim = m["similarity"][f][h]
                table.add_row(f"  {h}'s top k", *(_cell(sim[k]) for k in ks))
            random = _cell(entry["random"][f])
            table.add_row("  random library molecules", *[random] * len(ks))
        table.add_row(f"Position in the other's ranking (of {n:,})", style="bold")
        for f, g in ((a, b), (b, a)):
            ranks = m["rank"][g][f]
            table.add_row(
                f"  {f}'s top k, in {g}'s ranking",
                *(_cell(ranks[k], rank=True) for k in ks),
            )
        table.add_row("  a random molecule", *[f"{n / 2:,.0f}"] * len(ks))
        tables.append(table)
    return header, tables, NOTES


def show(stored, entry):
    """Print a stored comparison of two featurizers."""
    header, tables, notes = render(stored, entry)
    console = Console()
    for line in header:
        console.print(line, markup=False, highlight=False)
    for table in tables:
        console.print(table)
    for line in notes:
        console.print(line, markup=False, highlight=False)


def compare_cmd():
    @lazysim_cli.command(short_help="Compare two featurizers on an Ersilia library.")
    @click.argument("name")
    @click.option(
        "-f",
        "--featurizer",
        "featurizers",
        multiple=True,
        required=True,
        type=click.Choice(AVAILABLE),
        help="Give it twice: the two featurizers to compare.",
    )
    def compare(name, featurizers):
        """Show how the hits of two featurizers compare on Ersilia library NAME.

        The results were computed by Ersilia and come with lazy-sim, so the library
        need not be fetched.
        """
        if len(featurizers) != 2:
            raise click.UsageError("give -f exactly twice, such as -f clamp -f ecfp")
        show(*run_or_fail(results.pair, name, *featurizers))
