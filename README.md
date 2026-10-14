# Ultra-fast similarity search in large chemical libraries

A tool from the [Ersilia Open Source Initiative](https://ersilia.io) that finds the nearest neighbours of query molecules in libraries of millions of compounds, on a CPU.

```bash
pip install git+https://github.com/ersilia-os/lazy-sim.git                       # CLAMP
pip install "lazy-sim[ecfp] @ git+https://github.com/ersilia-os/lazy-sim.git"   # CLAMP and ECFP
```

## Usage

| Command | Description |
|---|---|
| `lazysim catalog --remote` | List the libraries Ersilia has indexed |
| `lazysim fetch ersilia-reference` | Download one, ready to search |
| `lazysim index -i library.csv -a mylib` | Index your own library |
| `lazysim search -i queries.csv -a mylib -o hits.csv` | Top 100 hits per query (`-k` to change) |
| `lazysim catalog` | List your local libraries |
| `lazysim info -a mylib` | Describe a library |

- `fetch` serves only libraries built from the [Ersilia Maintained Inputs](https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs). For now, that is `ersilia-reference`, the Ersilia Reference Library (1.36M compounds).
- Libraries need `smiles` and `id` columns. Queries need `smiles`; without `id`, they are named by InChIKey. SMILES are used exactly as given.
- Libraries live in `~/lazysim/libraries/` (set `LAZYSIM_HOME` to move it); `-a` takes a name there or a path.
- `--embeddings` on `fetch` or `index` also keeps the full vectors, for your own analyses.

## Featurizers

Choose with `-f`. A library can hold both; `-f ecfp` on `fetch` or `index -a` adds it to one you have.

| `-f` | Similarity | Engine |
|---|---|---|
| `clamp` (default) | Cosine of [CLAMP](https://github.com/ml-jku/clamp) embeddings, approximate | [turbovec](https://pypi.org/project/turbovec/), 4-bit |
| `ecfp` | Tanimoto of ECFP6 (2048 bits), exact | [FPSim2](https://github.com/chembl/FPSim2), needs `[ecfp]` |

CLAMP scores are compressed: random pairs score about 0.5 and close analogues above 0.9.

## Performance

On a 16-thread laptop, 1,000 queries against the Ersilia Reference Library (1.36M compounds) take about 6 s with CLAMP and 8 s with ECFP. CLAMP returns 95% of the exact top 100. To measure these, see `scripts/evaluate_speed.py` and `scripts/evaluate_recall.py`.

## CLAMP

> Seidl, P., Vall, A., Hochreiter, S. & Klambauer, G. Enhancing activity prediction models in drug discovery with the ability to understand human language. *ICML* (2023).

lazy-sim runs CLAMP's original weights with RDKit and numpy, without torch. The weights (174 MB) are downloaded once, to `~/lazysim/models/`. The CLAMP model terms state that it must not be used to exploit or harm.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
