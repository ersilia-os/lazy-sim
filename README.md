# Ultra-fast similarity search in large chemical libraries

lazy-sim is a tool from the [Ersilia Open Source Initiative](https://ersilia.io). It finds the nearest neighbours of query molecules in a fixed library of millions of compounds. It needs no GPU and no torch.

```bash
pip install git+https://github.com/ersilia-os/lazy-sim.git            # CLAMP
pip install "lazy-sim[ecfp] @ git+https://github.com/ersilia-os/lazy-sim.git"   # + ECFP
```

lazy-sim needs Python 3.12 or later.

## Usage

| Command | Description |
|---|---|
| `lazysim catalog --remote` | List the libraries Ersilia has already indexed |
| `lazysim fetch ersilia-reference [-f ecfp]` | Download one, ready to search |
| `lazysim index -i library.csv -a mylib [-f clamp -f ecfp]` | Index your own library, once |
| `lazysim search -i queries.csv -a ersilia-reference -o hits.csv [-f ecfp]` | Top hits for each query (`-k`, default 100) |
| `lazysim catalog` | List your local libraries |
| `lazysim info -a ersilia-reference` | Describe a library |

Library CSVs need `smiles` and `id` columns. Query CSVs need `smiles`; `id` is optional, and queries without one are named by their InChIKey. SMILES are used exactly as given. The output has one row per hit: `query_id, query_smiles, rank, hit_id, hit_smiles, similarity`.

## Featurizers

| `-f` | Molecules as | Similarity | Search engine |
|---|---|---|---|
| `clamp` (default) | [CLAMP](https://github.com/ml-jku/clamp) embeddings | cosine | [turbovec](https://pypi.org/project/turbovec/), 4-bit |
| `ecfp` | ECFP6 (Morgan radius 3, 2048 bits) | exact Tanimoto | [FPSim2](https://github.com/chembl/FPSim2), needs `lazy-sim[ecfp]` |

A library can hold both. `search` uses CLAMP unless you pass `-f ecfp`.

## Libraries

- **Where they live.** A library name such as `ersilia-reference` refers to a folder in `~/lazysim/libraries/` (set `LAZYSIM_HOME` to move it). A path such as `-a ./mylib` works anywhere.
- **What's inside.** The ids and SMILES (`library.db`) and a `manifest.json`, plus one index per featurizer (`clamp.tv`, `ecfp.h5`), each described by its own `.json`.
- **Adding a featurizer.** `lazysim fetch NAME -f ecfp` or `lazysim index -a mylib -f ecfp` adds it to a library you already have.
- **CLAMP weights.** The weights (174 MB) are downloaded once, the first time a command needs them, into `~/lazysim/models/`. All libraries share them.
- **Full embeddings.** `--embeddings`, on `fetch` or `index`, also keeps the full vectors for your own analyses: float32 CLAMP embeddings (3 KB per molecule) and packed ECFP bits (256 B, `np.unpackbits`). `search` does not need them.

## Performance

Tested on a laptop with 16 threads:
- **CLAMP on ChEMBL (2.9M compounds):** 1,000 queries take about 12 s from start to finish, using 1.5 GB of memory. The index keeps 4 bits per dimension, and the results include 96% of the exact top-100 (recall@100 0.96).
- **ECFP on the Ersilia Reference Library (1.36M compounds):** 1,000 queries take about 8 s, and the Tanimoto values are exact.
- **Indexing** runs at about 2,300 molecules per second for CLAMP.

## About the similarity score

For CLAMP, `similarity` is the cosine between embeddings, not a Tanimoto coefficient. CLAMP was trained on bioactivity data, so the scores are compressed: random pairs score around 0.5, and close analogues score above 0.9. For ECFP, it is the usual Tanimoto coefficient.

## CLAMP

> Seidl, P., Vall, A., Hochreiter, S. & Klambauer, G. Enhancing activity prediction models in drug discovery with the ability to understand human language. *ICML* (2023).

lazy-sim reimplements CLAMP's fingerprint and compound encoder with RDKit and numpy, using the original weights. The embeddings match the original code to within 1e-6. The CLAMP model terms state that it must not be used to exploit or harm.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
