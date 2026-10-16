# Fast similarity search in large chemical libraries

A tool from the [Ersilia Open Source Initiative](https://ersilia.io) that finds the nearest neighbours of query molecules in libraries of millions of compounds, on a CPU.

```bash
conda create -n lazysim python=3.12 -y
conda activate lazysim
git clone https://github.com/ersilia-os/lazy-sim.git
cd lazy-sim
pip install .
```

This covers searching and indexing with every featurizer except `ecfp`, which needs FPSim2: `pip install ".[ecfp]"`. No torch or GPU is needed, for searching or for indexing.

## Usage

| Command | Description |
|---|---|
| `lazysim catalog --remote` | List the libraries Ersilia has indexed |
| `lazysim fetch ersilia_reference_library_1M_v0 -f clamp` | Download one, ready to search |
| `lazysim index -i library.csv -a mylib -f clamp` | Index your own library |
| `lazysim search -i queries.csv -a mylib -o hits.csv -f clamp` | Top 100 hits per query (`-k` to change) |
| `lazysim catalog` | List your local libraries |
| `lazysim info -a mylib` | Describe a library |
| `lazysim compare ersilia_reference_library_1M_v0 -f clamp -f ecfp` | How two featurizers' hits compare on an Ersilia library |

- Libraries need `smiles` and `id` columns. Queries need `smiles`. Without `id`, they are named by InChIKey. SMILES are used exactly as given.
- The hits CSV has one row per hit: `query_id`, `query_smiles`, `rank`, `hit_id`, `hit_smiles`, `similarity`. Invalid queries are skipped with a warning. Library rows that fail to parse are listed in the library's `skipped.csv`.
- Libraries live in `~/lazysim/libraries/`. Set `LAZYSIM_HOME` to move it. `-a` takes a name there or a path.
- `--embeddings` on `fetch` or `index` also keeps the full vectors, for your own analyses.
- Model weights (CLAMP: 174 MB, SAND: 112 MB, Mol-JEPA: 182 MB, Monroe: 235 MB) are downloaded once, to `~/lazysim/models/`.
- lazy-sim is built for libraries of up to 10M compounds. Its indexes and search have only been tested on libraries of 1–10M compounds.

In Python, `index`, `fetch` and `search` take the same arguments as the commands (see their docstrings):

```python
import lazy_sim

lazy_sim.fetch("ersilia_reference_library_1M_v0", ["clamp"])
lazy_sim.search("queries.csv", "ersilia_reference_library_1M_v0", "hits.csv", "clamp", k=100)
lazy_sim.index("library.csv", "mylib", ["clamp", "ecfp"])
```

## Libraries

`fetch` serves only libraries from the [Ersilia Maintained Inputs](https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs), indexed by Ersilia with lazy-sim's featurizers. Only the Ersilia Reference Library is available so far, and only in part. The others are work in progress.

| Library | Compounds | Status |
|---|---|---|
| `ersilia_reference_library_1M_v0` | 1.36M | `clamp`, `ecfp`, `sand` and `moljepa` available. `monroe` in progress |
| `molport_5M_v0` | 5M | Work in progress |
| `enamine_lsc_2M_v0` | 2M | Work in progress |
| `enamine_real_sample_10M_v0` | 10M | Work in progress |
| `coconut_715K_v0` | 715K | Work in progress |
| `enamine_hll_460K_v0` | 460K | Work in progress |

The featurizers are representation models of the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), re-implemented in lazy-sim. Optimisations here and different dependency versions (RDKit in particular) can make their representations differ slightly from the Hub's. The checks are in [docs/featurizers.md](docs/featurizers.md).

## Featurizers

`fetch`, `index` and `search` need `-f`. There is no default. A library can hold several featurizers, and `-f` on `fetch` or `index -a` adds one to a library you have.

| `-f` | Similarity | Engine |
|---|---|---|
| `clamp` | Cosine of [CLAMP](https://github.com/ml-jku/clamp) embeddings, trained on bioactivity data | [turbovec](https://pypi.org/project/turbovec/), 4-bit (approximate) |
| `ecfp` | Tanimoto of ECFP6 (2,048 bits) | [FPSim2](https://github.com/chembl/FPSim2) (exact), needs `[ecfp]` |
| `sand` | Cosine of [SAND](https://github.com/pfizer-opensource/SAND) embeddings, which track 3D shape overlap | turbovec, 4-bit (approximate) |
| `moljepa` | Cosine of [Mol-JEPA](https://github.com/Boehringer-Ingelheim/mol-jepa) embeddings, which carry predicted biological and quantum context. **Non-commercial use only (CC BY-NC 4.0)** | turbovec, 4-bit (approximate) |
| `monroe` | Cosine of [Monroe](https://github.com/blazejba/Monroe) embeddings, from a graph transformer with a 3D conformer | turbovec, 4-bit (approximate) |

Scores can't be compared across featurizers. A random library molecule scores about 0.52 with CLAMP, 0.12 with Mol-JEPA, 0.09 with ECFP and 0.06 with SAND. `lazysim compare` shows how far two featurizers agree.

How each featurizer is computed, where its weights come from, how it was checked against the original code, how well it searches, and its citation and licence: **[docs/featurizers.md](docs/featurizers.md)**.

## Performance

On a 16-thread laptop, 1,000 queries against the Ersilia Reference Library (1.36M compounds) take about 7.5 s with CLAMP, 10 s with ECFP, 18 s with Mol-JEPA and 25 s with SAND, from launch to results. CLAMP, SAND and Mol-JEPA return 95–97% of the exact top 100. ECFP is exact. Monroe isn't indexed yet. Details and how to reproduce them are in [docs/featurizers.md](docs/featurizers.md).

## Maintainers

How Ersilia builds, evaluates and publishes its libraries: [docs/maintainers.md](docs/maintainers.md).

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
